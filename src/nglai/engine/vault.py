"""Hot/Cold Model Vault (256GB SSD <-> 1TB HDD Tiering) & Resumable GGUF Downloader."""

from __future__ import annotations

import os
import shutil
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
            # Not enough room on SSD -> run directly from Cold HDD gracefully
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
    ) -> Path:
        """
        Download GGUF from HuggingFace with HTTP Range resume support directly into Cold Vault,
        then optionally stage into Hot SSD. Never touches the OS C: drive cache.
        """
        spec = get_model_spec(model_id)
        if not spec:
            raise ValueError(f"Modelo desconhecido no catálogo: {model_id}")

        self.cold_dir.mkdir(parents=True, exist_ok=True)
        final_cold = self.cold_file(spec)
        part_file = final_cold.with_suffix(".gguf.part")

        if not final_cold.exists():
            existing_bytes = part_file.stat().st_size if part_file.exists() else 0
            headers: Dict[str, str] = {}
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            if existing_bytes > 0:
                headers["Range"] = f"bytes={existing_bytes}-"

            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=300.0)) as client:
                with client.stream("GET", spec.hf_download_url, headers=headers) as resp:
                    if resp.status_code == 416:
                        # Range already satisfied
                        part_file.replace(final_cold)
                    else:
                        resp.raise_for_status()
                        mode = "ab" if resp.status_code == 206 and existing_bytes > 0 else "wb"
                        if mode == "wb":
                            existing_bytes = 0
                        total_header = resp.headers.get("Content-Length")
                        total_bytes = (existing_bytes + int(total_header)) if total_header else 0

                        downloaded = existing_bytes
                        with open(part_file, mode) as fh:
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
