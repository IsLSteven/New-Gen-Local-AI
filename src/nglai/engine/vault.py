"""Hot/Cold Model Vault (256GB SSD <-> 1TB HDD Tiering) & Parallel Multi-Stream GGUF Downloader."""

from __future__ import annotations

import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import httpx

from ..config import NGLAIConfig
from .catalog import MODEL_CATALOG, ModelSpec, get_model_spec


@dataclass
class VaultEntry:
    model_id: str
    name: str
    filename: str
    expected_gb: float
    in_hot_ssd: bool
    in_cold_hdd: bool
    hot_path: Optional[str]
    cold_path: Optional[str]

    @property
    def is_downloaded(self) -> bool:
        return self.in_hot_ssd or self.in_cold_hdd


class ModelVault:
    """
    Manages GGUF weights across a fast Hot SSD cache (keeping only the active model on the 256GB SSD)
    and a large Cold HDD archive (storing all downloaded models on the 1TB HDD).
    """

    def __init__(self, config: NGLAIConfig) -> None:
        self.config = config
        self.config.ensure_directories()
        self.hot_dir = Path(self.config.hot_ssd_dir)
        self.cold_dir = Path(self.config.cold_hdd_dir)

    def _same_storage_root(self) -> bool:
        return self.hot_dir.resolve() == self.cold_dir.resolve()

    def hot_file(self, spec: ModelSpec) -> Path:
        return self.hot_dir / spec.filename

    def cold_file(self, spec: ModelSpec) -> Path:
        return self.cold_dir / spec.filename

    def inspect_catalog(self) -> List[VaultEntry]:
        entries: List[VaultEntry] = []
        for spec in MODEL_CATALOG.values():
            hp = self.hot_file(spec)
            cp = self.cold_file(spec)
            entries.append(
                VaultEntry(
                    model_id=spec.id,
                    name=spec.name,
                    filename=spec.filename,
                    expected_gb=spec.size_gb,
                    in_hot_ssd=hp.exists() and hp.stat().st_size > 0,
                    in_cold_hdd=cp.exists() and cp.stat().st_size > 0,
                    hot_path=str(hp) if hp.exists() else None,
                    cold_path=str(cp) if cp.exists() else None,
                )
            )
        return entries

    def resolve_active_path(self, model_id: str, promote_to_hot: bool = True) -> Optional[Path]:
        """Return path on Hot SSD (promoting from Cold HDD and evicting inactive Hot models if needed)."""
        spec = get_model_spec(model_id)
        if not spec:
            return None
        hp = self.hot_file(spec)
        cp = self.cold_file(spec)

        if hp.exists() and hp.stat().st_size > 0:
            return hp

        if cp.exists() and cp.stat().st_size > 0:
            if not promote_to_hot or self._same_storage_root():
                return cp
            self.evict_inactive_from_hot(keep_model_id=model_id)
            actual_gb = cp.stat().st_size / (1024**3)
            free_gb = shutil.disk_usage(self.hot_dir).free / (1024**3)
            if free_gb >= (actual_gb + 1.0):
                tmp_target = hp.with_suffix(".gguf.promoting")
                shutil.copy2(cp, tmp_target)
                tmp_target.replace(hp)
                return hp
            return cp

        return None

    def evict_inactive_from_hot(self, keep_model_id: Optional[str] = None) -> List[str]:
        """Move inactive GGUF models out of the 256GB Hot SSD back to the 1TB Cold HDD."""
        if self._same_storage_root():
            return []

        keep_filename: Optional[str] = None
        if keep_model_id and keep_model_id in MODEL_CATALOG:
            keep_filename = MODEL_CATALOG[keep_model_id].filename

        evicted: List[str] = []
        for gguf in self.hot_dir.glob("*.gguf"):
            if keep_filename and gguf.name == keep_filename:
                continue
            cold_dest = self.cold_dir / gguf.name
            if not cold_dest.exists() or cold_dest.stat().st_size != gguf.stat().st_size:
                shutil.copy2(gguf, cold_dest)
            gguf.unlink()
            evicted.append(gguf.name)
        return evicted

    def download_model(
        self,
        model_id: str,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        activate_hot: bool = True,
        parallel_workers: int = 6,
    ) -> Path:
        """
        Multi-connection parallel HTTP Range downloader with seamless resume from existing .part files.
        Avoids single-stream CDN throttling on large 12GB-18GB GGUF downloads.
        """
        spec = get_model_spec(model_id)
        if not spec:
            raise ValueError(f"Modelo desconhecido no catálogo: {model_id}")

        self.cold_dir.mkdir(parents=True, exist_ok=True)
        final_cold = self.cold_file(spec)
        part_file = final_cold.with_suffix(".gguf.part")

        if not final_cold.exists():
            base_headers: Dict[str, str] = {"User-Agent": "New-Gen-Local-AI-Vault/0.1"}
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                base_headers["Authorization"] = f"Bearer {hf_token}"

            existing_bytes = part_file.stat().st_size if part_file.exists() else 0

            # Probe final redirect URL and total size via stream headers ONLY (never buffer body!)
            probe_headers = {**base_headers, "Range": "bytes=0-0"}
            with httpx.Client(follow_redirects=True, timeout=30.0) as probe_client:
                with probe_client.stream("GET", spec.hf_download_url, headers=probe_headers) as probe:
                    resolved_url = str(probe.url)
                    content_range = probe.headers.get("Content-Range", "")
                    supports_range = probe.status_code == 206 and "/" in content_range
                    total_bytes = (
                        int(content_range.split("/")[-1])
                        if supports_range
                        else int(probe.headers.get("Content-Length") or 0)
                    )

            if total_bytes > 0 and existing_bytes >= total_bytes and existing_bytes > 0:
                part_file.replace(final_cold)
            elif supports_range and total_bytes > (existing_bytes + 8 * 1024 * 1024) and parallel_workers > 1:
                # Pre-extend sparse random-access file preserving existing_bytes [0 .. existing_bytes-1]
                mode = "r+b" if part_file.exists() else "w+b"
                with open(part_file, mode) as pre_fh:
                    if os.name == "nt" and existing_bytes == 0:
                        import subprocess as _sp

                        _sp.run(["fsutil", "sparse", "setflag", str(part_file)], capture_output=True, check=False)
                    pre_fh.truncate(total_bytes)

                remaining = total_bytes - existing_bytes
                chunk_span = max(4 * 1024 * 1024, (remaining + parallel_workers - 1) // parallel_workers)
                ranges: List[tuple[int, int]] = []
                cursor = existing_bytes
                while cursor < total_bytes:
                    end = min(total_bytes - 1, cursor + chunk_span - 1)
                    ranges.append((cursor, end))
                    cursor = end + 1

                lock = threading.Lock()
                state = {"done": existing_bytes}

                def _fetch_slice(start_b: int, end_b: int) -> None:
                    hdrs = {**base_headers, "Range": f"bytes={start_b}-{end_b}"}
                    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=300.0)) as w_client:
                        with w_client.stream("GET", resolved_url, headers=hdrs) as r_stream:
                            r_stream.raise_for_status()
                            offset = start_b
                            with open(part_file, "r+b") as seg_fh:
                                seg_fh.seek(offset)
                                for chunk in r_stream.iter_bytes(chunk_size=1024 * 512):
                                    if chunk:
                                        seg_fh.write(chunk)
                                        with lock:
                                            state["done"] += len(chunk)
                                            cur_done = state["done"]
                                        if progress_cb:
                                            progress_cb(cur_done, total_bytes)

                with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
                    futures = [pool.submit(_fetch_slice, s, e) for s, e in ranges]
                    for fut in as_completed(futures):
                        fut.result()

                part_file.replace(final_cold)
            else:
                # Single-stream fallback
                headers = dict(base_headers)
                if existing_bytes > 0:
                    headers["Range"] = f"bytes={existing_bytes}-"
                with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=300.0)) as client:
                    with client.stream("GET", spec.hf_download_url, headers=headers) as resp:
                        resp.raise_for_status()
                        f_mode = "ab" if resp.status_code == 206 and existing_bytes > 0 else "wb"
                        if f_mode == "wb":
                            existing_bytes = 0
                        downloaded = existing_bytes
                        with open(part_file, f_mode) as fh:
                            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                                if chunk:
                                    fh.write(chunk)
                                    downloaded += len(chunk)
                                    if progress_cb:
                                        progress_cb(downloaded, total_bytes)
                        part_file.replace(final_cold)

        if activate_hot:
            resolved = self.resolve_active_path(model_id, promote_to_hot=True)
            if resolved:
                return resolved
        return final_cold
