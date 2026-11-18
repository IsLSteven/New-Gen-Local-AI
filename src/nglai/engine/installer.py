"""Cross-Platform Automated llama.cpp Binary Installer (CUDA / Vulkan / Metal / CPU)."""

from __future__ import annotations

import os
import platform
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import httpx

from ..config import NGLAIConfig
from .hardware import detect_hardware

LLAMA_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"


def _select_release_assets(
    releases: List[Dict],
    system_name: str,
    arch: str,
    backend: str,
) -> Tuple[str, List[Dict]]:
    """Find the most recent llama.cpp release containing matching prebuilt binary archives."""
    is_arm = arch in ("arm64", "aarch64")

    for rel in releases:
        assets = rel.get("assets") or []
        if not assets:
            continue
        tag = rel.get("tag_name", "unknown")

        def find_asset(pattern: str) -> Optional[Dict]:
            rx = re.compile(pattern, re.IGNORECASE)
            for a in assets:
                if rx.search(a.get("name", "")):
                    return a
            return None

        chosen: List[Dict] = []

        if system_name == "windows":
            arch_tok = "arm64" if is_arm else "x64"
            if backend == "cuda":
                cudart = find_asset(rf"^cudart-llama-.*bin-win-cuda-12\..*-{arch_tok}\.zip$") or find_asset(
                    rf"^cudart-llama-.*bin-win-cuda-.*-{arch_tok}\.zip$"
                )
                main_zip = find_asset(rf"^llama-.*-bin-win-cuda-12\..*-{arch_tok}\.zip$") or find_asset(
                    rf"^llama-.*-bin-win-cuda-.*-{arch_tok}\.zip$"
                )
                if main_zip:
                    if cudart:
                        chosen.append(cudart)
                    chosen.append(main_zip)
            elif backend == "vulkan":
                vk = find_asset(rf"^llama-.*-bin-win-vulkan-{arch_tok}\.zip$")
                if vk:
                    chosen.append(vk)
            else:
                cpu = find_asset(rf"^llama-.*-bin-win-cpu-{arch_tok}\.zip$")
                if cpu:
                    chosen.append(cpu)

        elif system_name == "darwin":
            arch_tok = "arm64" if is_arm else "x64"
            mac = find_asset(rf"^llama-.*-bin-macos-{arch_tok}\.(zip|tar\.gz)$")
            if mac:
                chosen.append(mac)

        else:  # Linux
            arch_tok = "arm64" if is_arm else "x64"
            if backend == "cuda":
                cudart = find_asset(rf"^cudart-llama-.*bin-ubuntu-cuda-12\..*-{arch_tok}\.tar\.gz$") or find_asset(
                    rf"^cudart-llama-.*bin-ubuntu-cuda-.*-{arch_tok}\.tar\.gz$"
                )
                main_tar = find_asset(rf"^llama-.*-bin-ubuntu-cuda-12\..*-{arch_tok}\.tar\.gz$") or find_asset(
                    rf"^llama-.*-bin-ubuntu-cuda-.*-{arch_tok}\.tar\.gz$"
                )
                if main_tar:
                    if cudart:
                        chosen.append(cudart)
                    chosen.append(main_tar)
            elif backend == "vulkan":
                vk = find_asset(rf"^llama-.*-bin-ubuntu-vulkan-{arch_tok}\.tar\.gz$")
                if vk:
                    chosen.append(vk)
            else:
                cpu = find_asset(rf"^llama-.*-bin-ubuntu-{arch_tok}\.tar\.gz$")
                if cpu:
                    chosen.append(cpu)

        if chosen:
            return tag, chosen

    raise RuntimeError("Nenhum binário pré-compilado compatível encontrado nas últimas releases do llama.cpp.")


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {archive_path}")


def install_llama_engine(
    config: NGLAIConfig,
    backend: str = "auto",
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Path:
    """
    Download and extract official llama.cpp prebuilt server binaries into config.engine_bin_dir.
    Supports Windows (CUDA/Vulkan/CPU), Linux (CUDA/Vulkan/CPU), and macOS (Metal/CPU).
    """
    config.ensure_directories()
    bin_dir = Path(config.engine_bin_dir)
    tmp_dir = Path(config.cache_dir) / "tmp"
    bin_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sys_name = platform.system().lower()
    arch = platform.machine().lower()
    hw = detect_hardware(config.hot_ssd_dir, config.cold_hdd_dir)

    resolved_backend = backend.lower()
    if resolved_backend == "auto":
        if sys_name == "darwin":
            resolved_backend = "metal"
        elif hw.has_cuda and hw.has_tensor_cores:
            resolved_backend = "cuda"
        else:
            resolved_backend = "vulkan" if sys_name in ("windows", "linux") else "cpu"

    headers = {"User-Agent": "New-Gen-Local-AI-Installer"}
    gh_token = os.environ.get("GITHUB_TOKEN")
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"

    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=300.0), headers=headers) as client:
        resp = client.get(LLAMA_RELEASES_API)
        resp.raise_for_status()
        releases = resp.json()

        # Fallback gracefully if vulkan not found -> cpu
        try:
            _tag, assets = _select_release_assets(releases, sys_name, arch, resolved_backend)
        except RuntimeError:
            _tag, assets = _select_release_assets(releases, sys_name, arch, "cpu")

        for asset in assets:
            fname = asset["name"]
            url = asset["browser_download_url"]
            archive_file = tmp_dir / fname

            with client.stream("GET", url) as dl:
                dl.raise_for_status()
                total = int(dl.headers.get("Content-Length") or asset.get("size") or 0)
                done = 0
                with open(archive_file, "wb") as fh:
                    for chunk in dl.iter_bytes(chunk_size=1024 * 512):
                        if chunk:
                            fh.write(chunk)
                            done += len(chunk)
                            if progress_cb:
                                progress_cb(fname, done, total)

            _extract_archive(archive_file, bin_dir)
            try:
                archive_file.unlink()
            except OSError:
                pass

    # Ensure executable permissions on Unix
    if sys_name != "windows":
        for item in bin_dir.rglob("llama-*"):
            if item.is_file():
                item.chmod(item.stat().st_mode | 0o111)

    exe_name = "llama-server.exe" if sys_name == "windows" else "llama-server"
    for candidate in bin_dir.rglob(exe_name):
        return candidate

    raise FileNotFoundError(f"Extração concluída em {bin_dir}, mas '{exe_name}' não foi localizado.")
