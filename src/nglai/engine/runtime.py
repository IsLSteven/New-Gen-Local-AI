"""llama-server Process Manager with Automatic TurboQuant KV Probe & Health Monitor."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from ..config import NGLAIConfig
from .catalog import get_model_spec
from .hardware import SplitPlan, calculate_surgical_split, detect_hardware
from .vault import ModelVault


@dataclass
class EngineStatus:
    running: bool
    endpoint: str
    pid: Optional[int]
    active_model_id: str
    turboquant_enabled: bool
    kv_cache_type: str
    execution_mode: str
    message: str


class EngineRuntime:
    """Controls the local compressed inference server (llama-server / llama-cpp-turboquant)."""

    def __init__(self, config: NGLAIConfig) -> None:
        self.config = config
        self.vault = ModelVault(config)
        self._process: Optional[subprocess.Popen] = None
        self._active_plan: Optional[SplitPlan] = None
        self._turboquant_active: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.config.engine_host}:{self.config.engine_port}"

    def find_llama_server_binary(self) -> Optional[Path]:
        """Locate llama-server binary in engine_bin/ or system PATH."""
        exe_name = "llama-server.exe" if platform.system().lower() == "windows" else "llama-server"
        bin_dir = Path(self.config.engine_bin_dir)

        if bin_dir.exists():
            direct = bin_dir / exe_name
            if direct.exists():
                return direct
            for candidate in bin_dir.rglob(exe_name):
                return candidate

        which_hit = shutil.which(exe_name) or shutil.which("llama-server")
        return Path(which_hit) if which_hit else None

    def probe_turboquant_support(self, binary: Path) -> bool:
        """Check whether the installed llama-server build supports TurboQuant ('turbo3'/'turbo4')."""
        try:
            proc = subprocess.run(
                [str(binary), "--help"],
                capture_output=True,
                text=True,
                timeout=6,
                check=False,
            )
            help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            return "turbo3" in help_text.lower() or "turboquant" in help_text.lower()
        except Exception:
            return False

    def is_healthy(self) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def status(self) -> EngineStatus:
        healthy = self.is_healthy()
        pid = self._process.pid if (self._process and self._process.poll() is None) else None
        kv_used = (
            self._active_plan.kv_cache_type_k
            if (self._active_plan and self._turboquant_active)
            else ("q4_0" if self._active_plan else self.config.kv_cache_mode)
        )
        mode = self._active_plan.execution_mode if self._active_plan else "standby"
        msg = "Motor Triple-Stack online e pronto" if healthy else "Motor parado (aguardando inicialização)"
        return EngineStatus(
            running=healthy,
            endpoint=f"{self.base_url}/v1",
            pid=pid,
            active_model_id=self.config.active_model_id,
            turboquant_enabled=self._turboquant_active,
            kv_cache_type=kv_used,
            execution_mode=mode,
            message=msg,
        )

    def build_launch_command(self, model_id: Optional[str] = None) -> tuple[List[str], SplitPlan, bool]:
        """Compute surgical hardware split and assemble exact llama-server command line."""
        target_id = model_id or self.config.active_model_id
        spec = get_model_spec(target_id)
        if not spec:
            raise ValueError(f"Modelo '{target_id}' não encontrado no catálogo.")

        binary = self.find_llama_server_binary()
        if not binary:
            raise FileNotFoundError(
                f"Binário 'llama-server' não encontrado em '{self.config.engine_bin_dir}'. "
                "Execute 'scripts/setup_engine.ps1' ou 'nglai engine install'."
            )

        model_path = self.vault.resolve_active_path(target_id, promote_to_hot=True)
        if not model_path:
            raise FileNotFoundError(
                f"Pesos do modelo '{spec.name}' ainda não baixados. Use 'nglai vault pull {target_id}'."
            )

        hw = detect_hardware(self.config.hot_ssd_dir, self.config.cold_hdd_dir)
        plan = calculate_surgical_split(
            hw=hw,
            model=spec,
            requested_ctx=self.config.context_size,
            kv_mode=self.config.kv_cache_mode,
        )
        plan.flash_attention = self.config.flash_attention and plan.flash_attention
        has_turbo = self.probe_turboquant_support(binary)

        cmd = [
            str(binary),
            "-m",
            str(model_path),
            "--host",
            self.config.engine_host,
            "--port",
            str(self.config.engine_port),
            *plan.to_llama_args(turboquant_supported=has_turbo),
        ]

        # Optional Draft Model for Speculative Decoding if configured and available
        if self.config.draft_model_id:
            draft_path = self.vault.resolve_active_path(self.config.draft_model_id, promote_to_hot=False)
            if draft_path:
                cmd.extend(["-md", str(draft_path), "--draft-max", "8"])

        return cmd, plan, has_turbo

    def start(self, model_id: Optional[str] = None, wait_seconds: int = 90) -> EngineStatus:
        """Launch llama-server in the background and wait until /health responds 200."""
        if self.is_healthy() and (not model_id or model_id == self.config.active_model_id):
            return self.status()

        self.stop()
        if model_id:
            self.config.active_model_id = model_id

        cmd, plan, has_turbo = self.build_launch_command(self.config.active_model_id)
        self._active_plan = plan
        self._turboquant_active = has_turbo

        log_dir = Path(self.config.base_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / "llama_server.log", "a", encoding="utf-8")

        env = os.environ.copy()
        env["TMP"] = str(Path(self.config.cache_dir) / "tmp")
        env["TEMP"] = str(Path(self.config.cache_dir) / "tmp")

        creationflags = subprocess.CREATE_NO_WINDOW if platform.system().lower() == "windows" else 0
        self._process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=creationflags,
        )

        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"llama-server encerrou inesperadamente (código {self._process.returncode}). "
                    f"Verifique logs em {log_dir / 'llama_server.log'}."
                )
            if self.is_healthy():
                return self.status()
            time.sleep(1.0)

        return self.status()

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
