"""Central configuration & Zero-C-Drive storage isolation for New-Gen-Local-AI."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


def _default_base_dir() -> Path:
    """Resolve base project/runtime directory, preferring env override or workspace root."""
    env_root = os.environ.get("NGLAI_HOME")
    if env_root:
        return Path(env_root).expanduser().resolve()
    # Default to repository root when running from source, otherwise ~/.nglai
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").exists():
        return repo_root
    return Path.home() / ".nglai"


@dataclass
class NGLAIConfig:
    """Persistent configuration for engine, Hot/Cold storage vault, and Compound AI agent."""

    base_dir: str = field(default_factory=lambda: str(_default_base_dir()))
    hot_ssd_dir: str = ""
    cold_hdd_dir: str = ""
    cache_dir: str = ""
    engine_bin_dir: str = ""

    # Default model selection (30B-35B Compressed MoE tier)
    active_model_id: str = "qwen3.6-35b-a3b-ud-iq3"
    draft_model_id: Optional[str] = None

    # Engine runtime settings
    engine_host: str = "127.0.0.1"
    engine_port: int = 8081
    web_host: str = "127.0.0.1"
    web_port: int = 7860
    context_size: int = 32768
    kv_cache_mode: str = "auto"  # "auto" (probes turbo3 -> q4_0), "turbo3", "turbo4", "q4_0", "q8_0", "f16"
    flash_attention: bool = True
    enable_thinking: bool = True

    # Compound AI Agent settings (Opus/Argon-style pipeline)
    max_agent_steps: int = 25
    auto_verify_edits: bool = True
    auto_approve_safe_commands: bool = True
    repomap_max_tokens: int = 2200
    context_compact_threshold: float = 0.78

    def __post_init__(self) -> None:
        root = Path(self.base_dir).resolve()
        self.base_dir = str(root)
        if not self.hot_ssd_dir:
            self.hot_ssd_dir = str(root / "hot_ssd")
        if not self.cold_hdd_dir:
            self.cold_hdd_dir = str(root / "vault_hdd")
        if not self.cache_dir:
            self.cache_dir = str(root / ".cache")
        if not self.engine_bin_dir:
            self.engine_bin_dir = str(root / "engine_bin")

    @property
    def config_file(self) -> Path:
        return Path(self.base_dir) / "nglai.local.json"

    def ensure_directories(self) -> None:
        """Create required directories and enforce Zero-C-Drive cache environment variables."""
        for folder in (
            self.hot_ssd_dir,
            self.cold_hdd_dir,
            self.cache_dir,
            self.engine_bin_dir,
            str(Path(self.cache_dir) / "hf"),
            str(Path(self.cache_dir) / "pip"),
            str(Path(self.cache_dir) / "cuda_jit"),
            str(Path(self.cache_dir) / "tmp"),
        ):
            Path(folder).mkdir(parents=True, exist_ok=True)

        # Isolate HuggingFace, Pip, CUDA JIT, and temporary caches away from OS C: drive
        os.environ.setdefault("HF_HOME", str(Path(self.cache_dir) / "hf"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(self.cache_dir) / "hf" / "hub"))
        os.environ.setdefault("PIP_CACHE_DIR", str(Path(self.cache_dir) / "pip"))
        os.environ.setdefault("CUDA_CACHE_PATH", str(Path(self.cache_dir) / "cuda_jit"))

    def save(self, path: Optional[Path] = None) -> Path:
        target = path or self.config_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "NGLAIConfig":
        default_cfg = cls()
        target = path or default_cfg.config_file
        if target.exists():
            try:
                raw = json.loads(target.read_text(encoding="utf-8"))
                valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
                filtered = {k: v for k, v in raw.items() if k in valid_keys}
                cfg = cls(**filtered)
                cfg.ensure_directories()
                return cfg
            except Exception:
                pass
        default_cfg.ensure_directories()
        return default_cfg
