"""Hardware Auto-Profiler & Surgical GPU+RAM Tensor Split Calculator."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import psutil

from .catalog import ModelSpec, get_model_spec


@dataclass
class GPUInfo:
    index: int
    name: str
    vram_total_mb: int
    vram_free_mb: int
    driver_version: str

    @property
    def vram_total_gb(self) -> float:
        return round(self.vram_total_mb / 1024.0, 2)

    @property
    def vram_free_gb(self) -> float:
        return round(self.vram_free_mb / 1024.0, 2)


@dataclass
class DiskTierInfo:
    path: str
    total_gb: float
    free_gb: float


@dataclass
class HardwareProfile:
    os_name: str
    cpu_name: str
    cpu_physical_cores: int
    cpu_logical_threads: int
    ram_total_gb: float
    ram_available_gb: float
    gpus: List[GPUInfo] = field(default_factory=list)
    hot_ssd_disk: Optional[DiskTierInfo] = None
    cold_hdd_disk: Optional[DiskTierInfo] = None

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        return self.gpus[0] if self.gpus else None

    @property
    def has_cuda(self) -> bool:
        return len(self.gpus) > 0

    @property
    def total_vram_gb(self) -> float:
        return round(sum(g.vram_total_gb for g in self.gpus), 2)

    @property
    def combined_memory_pool_gb(self) -> float:
        return round(self.total_vram_gb + self.ram_total_gb, 2)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SplitPlan:
    """Computed execution & memory allocation plan for llama-server."""

    model_id: str
    feasible: bool
    execution_mode: str  # "100%-gpu", "surgical-hybrid-moe", "partial-gpu-dense", "cpu-only"
    n_gpu_layers: int
    cpu_moe_layers: int
    override_tensor_regex: Optional[str]
    recommended_ctx: int
    kv_cache_type_k: str
    kv_cache_type_v: str
    flash_attention: bool
    cpu_threads: int
    estimated_vram_gb: float
    estimated_ram_gb: float
    estimated_tok_per_sec: str
    recommended_model_id: str
    notes_pt: List[str] = field(default_factory=list)

    def to_llama_args(self, turboquant_supported: bool = False) -> List[str]:
        """Build exact CLI flags for llama-server."""
        args: List[str] = [
            "-c",
            str(self.recommended_ctx),
            "-ngl",
            str(self.n_gpu_layers),
            "-t",
            str(self.cpu_threads),
            "--jinja",
        ]
        if self.flash_attention:
            args.extend(["-fa", "on"])

        k_type = self.kv_cache_type_k
        v_type = self.kv_cache_type_v
        if not turboquant_supported and k_type.startswith("turbo"):
            k_type = "q4_0"
            v_type = "q4_0"

        args.extend(["--cache-type-k", k_type, "--cache-type-v", v_type])

        if self.override_tensor_regex:
            args.extend(["-ot", self.override_tensor_regex])

        return args


def _probe_nvidia_gpus() -> List[GPUInfo]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
        gpus: List[GPUInfo] = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        vram_total_mb=int(float(parts[2])),
                        vram_free_mb=int(float(parts[3])),
                        driver_version=parts[4],
                    )
                )
        return gpus
    except Exception:
        return []


def _probe_cpu_name() -> str:
    if platform.system().lower() == "windows":
        cpu_env = os.environ.get("PROCESSOR_IDENTIFIER")
        try:
            proc = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
                ],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except Exception:
            pass
        if cpu_env:
            return cpu_env
    return platform.processor() or "Generic CPU"


def _disk_info(path_str: Optional[str]) -> Optional[DiskTierInfo]:
    if not path_str:
        return None
    try:
        p = Path(path_str)
        probe_target = p if p.exists() else p.anchor or str(Path.cwd())
        usage = shutil.disk_usage(probe_target)
        return DiskTierInfo(
            path=str(p),
            total_gb=round(usage.total / (1024**3), 2),
            free_gb=round(usage.free / (1024**3), 2),
        )
    except Exception:
        return None


def detect_hardware(hot_ssd_dir: Optional[str] = None, cold_hdd_dir: Optional[str] = None) -> HardwareProfile:
    """Inspect live CPU, RAM, GPU VRAM, and Hot/Cold storage tiers."""
    vm = psutil.virtual_memory()
    phys_cores = psutil.cpu_count(logical=False) or 4
    log_threads = psutil.cpu_count(logical=True) or phys_cores

    return HardwareProfile(
        os_name=f"{platform.system()} {platform.release()}",
        cpu_name=_probe_cpu_name(),
        cpu_physical_cores=phys_cores,
        cpu_logical_threads=log_threads,
        ram_total_gb=round(vm.total / (1024**3), 2),
        ram_available_gb=round(vm.available / (1024**3), 2),
        gpus=_probe_nvidia_gpus(),
        hot_ssd_disk=_disk_info(hot_ssd_dir),
        cold_hdd_disk=_disk_info(cold_hdd_dir),
    )


def _build_layer_range_regex(start_layer: int, total_layers: int) -> str:
    """Build compact regex matchingblk.<N>.ffn_.*_exps.weight=CPU for N in [start_layer, total_layers-1]."""
    if start_layer <= 0:
        return r"\.ffn_(up|down|gate)_exps\.weight=CPU"
    indices = "|".join(str(i) for i in range(start_layer, total_layers))
    return rf"blk\.({indices})\.ffn_(up|down|gate)_exps\.weight=CPU"


def _kv_gb_per_1k_ctx(kv_mode: str, is_hybrid_linear: bool) -> float:
    """Estimate KV cache footprint per 1,024 tokens."""
    base = {
        "turbo2": 0.013,
        "turbo3": 0.018,
        "turbo4": 0.024,
        "q4_0": 0.026,
        "q8_0": 0.048,
        "f16": 0.095,
    }.get(kv_mode, 0.024)
    # Qwen3.6 hybrid linear-attention slashes KV growth even further
    return base * (0.65 if is_hybrid_linear else 1.0)


def recommend_best_model_id(hw: HardwareProfile) -> str:
    """Select optimal flagship model ID from catalog for the detected rig."""
    vram = hw.total_vram_gb
    ram = hw.ram_total_gb
    if vram >= 16 or ram >= 28:
        return "qwen3-coder-30b-a3b-ud-q4kxl" if ram >= 30 and vram >= 8 else "qwen3.6-35b-a3b-ud-iq3"
    if vram >= 7.5 and ram >= 15.0:
        # Target Rig: RTX 3050 8GB + 16GB RAM -> Qwen3.6-35B-A3B UD-IQ3_XXS (13.1 GB) or Qwen3-Coder-30B-A3B (12.8 GB)
        return "qwen3.6-35b-a3b-ud-iq3"
    if ram >= 24.0:
        return "qwen3.6-35b-a3b-ud-iq3"
    if vram >= 7.5:
        return "qwen2.5-coder-14b-ud-iq3m"
    return "qwen3.6-35b-a3b-ud-q2kxl"


def calculate_surgical_split(
    hw: HardwareProfile,
    model: ModelSpec,
    requested_ctx: int = 32768,
    kv_mode: str = "auto",
) -> SplitPlan:
    """
    Compute exact VRAM + System RAM tensor split so 30B-35B compressed MoE models run
    safely without Windows pagefile thrashing on 8GB VRAM + 16GB RAM (and any other rig).
    """
    notes: List[str] = []
    best_id = recommend_best_model_id(hw)

    # Optimal CPU thread count: physical cores + half SMT headroom (sweet spot for DDR4 bandwidth)
    optimal_threads = max(2, min(hw.cpu_logical_threads - 1, hw.cpu_physical_cores + 2))

    chosen_kv = "turbo3" if kv_mode == "auto" else kv_mode
    is_linear_hybrid = model.family == "qwen3.6"
    ctx = min(requested_ctx, model.max_native_ctx)

    # Reserve safe OS headroom
    vram_os_reserve = 1.15 if hw.total_vram_gb >= 7.5 else 0.85
    usable_vram = max(0.0, hw.total_vram_gb - vram_os_reserve) if hw.has_cuda else 0.0
    ram_os_reserve = 5.2 if hw.ram_total_gb <= 16.5 else 6.5
    usable_ram = max(1.0, hw.ram_total_gb - ram_os_reserve)

    kv_gb = round(_kv_gb_per_1k_ctx(chosen_kv, is_linear_hybrid) * (ctx / 1024.0), 2)
    # Shrink context automatically on ultra-tight VRAM so attention never OOMs
    if hw.has_cuda and usable_vram < 3.4 and ctx > 16384:
        ctx = 16384
        kv_gb = round(_kv_gb_per_1k_ctx(chosen_kv, is_linear_hybrid) * (ctx / 1024.0), 2)
        notes.append("Contexto ajustado para 16K para caber com segurança nos 4 GB de VRAM.")

    # Case 1: Model + compressed KV cache fits 100% inside GPU VRAM!
    if hw.has_cuda and (model.size_gb + kv_gb) <= usable_vram:
        notes.append(
            f"100% GPU Offload ativo: modelo ({model.size_gb} GB) + KV Cache comprimido ({kv_gb} GB) cabem inteiramente na VRAM!"
        )
        return SplitPlan(
            model_id=model.id,
            feasible=True,
            execution_mode="100%-gpu",
            n_gpu_layers=99,
            cpu_moe_layers=0,
            override_tensor_regex=None,
            recommended_ctx=ctx,
            kv_cache_type_k=chosen_kv,
            kv_cache_type_v=chosen_kv,
            flash_attention=True,
            cpu_threads=optimal_threads,
            estimated_vram_gb=round(model.size_gb + kv_gb, 2),
            estimated_ram_gb=0.6,
            estimated_tok_per_sec="32 - 58 tok/s",
            recommended_model_id=best_id,
            notes_pt=notes,
        )

    # Case 2: Flagship MoE (30B-35B-A3B) -> Surgical Attention-on-GPU + Partial/Full MoE Expert Split!
    if hw.has_cuda and model.is_moe:
        shared_attn_gb = round(model.size_gb * 0.18, 2)  # Attention + Norms + Embeddings (~18%)
        moe_experts_total_gb = round(model.size_gb - shared_attn_gb, 2)  # Routed FFN experts (~82%)
        per_layer_moe_gb = moe_experts_total_gb / max(1, model.num_layers)

        vram_for_attn_and_kv = shared_attn_gb + kv_gb
        if vram_for_attn_and_kv <= usable_vram:
            extra_vram_for_moe = max(0.0, usable_vram - vram_for_attn_and_kv)
            gpu_moe_layers = min(model.num_layers, int(extra_vram_for_moe // per_layer_moe_gb))
            cpu_moe_layers = model.num_layers - gpu_moe_layers

            est_vram = round(vram_for_attn_and_kv + (gpu_moe_layers * per_layer_moe_gb), 2)
            est_ram = round((cpu_moe_layers * per_layer_moe_gb) + 0.5, 2)
            feasible = est_ram <= (usable_ram + 1.2)
            regex = _build_layer_range_regex(gpu_moe_layers, model.num_layers) if cpu_moe_layers > 0 else None

            notes.append(
                f"Tensor Split Cirúrgico MoE: 100% da Atenção + KV Cache {chosen_kv} ({kv_gb} GB) + "
                f"{gpu_moe_layers}/{model.num_layers} blocos MoE na GPU ({est_vram} GB VRAM), e "
                f"{cpu_moe_layers} blocos de Experts na RAM ({est_ram} GB RAM)."
            )
            if not feasible:
                notes.append(
                    f"⚠️ Atenção: {est_ram} GB em RAM fica apertado para {hw.ram_total_gb} GB totais. "
                    "Feche abas pesadas do navegador ou use a variante UD-Q2_K_XL / UD-IQ3_XXS."
                )

            speed = "15 - 26 tok/s" if gpu_moe_layers >= 12 else "8 - 16 tok/s"
            return SplitPlan(
                model_id=model.id,
                feasible=feasible,
                execution_mode="surgical-hybrid-moe",
                n_gpu_layers=99,
                cpu_moe_layers=cpu_moe_layers,
                override_tensor_regex=regex,
                recommended_ctx=ctx,
                kv_cache_type_k=chosen_kv,
                kv_cache_type_v=chosen_kv,
                flash_attention=True,
                cpu_threads=optimal_threads,
                estimated_vram_gb=est_vram,
                estimated_ram_gb=est_ram,
                estimated_tok_per_sec=speed,
                recommended_model_id=best_id,
                notes_pt=notes,
            )

    # Case 3: Partial GPU layer offload (dense model larger than VRAM, or tiny <3GB VRAM)
    if hw.has_cuda and usable_vram > 1.0:
        vram_for_weights = max(0.5, usable_vram - kv_gb)
        per_layer_gb = model.size_gb / max(1, model.num_layers)
        gpu_layers = max(1, min(model.num_layers, int(vram_for_weights // per_layer_gb)))
        est_vram = round((gpu_layers * per_layer_gb) + kv_gb, 2)
        est_ram = round(max(0.5, model.size_gb - (gpu_layers * per_layer_gb)) + 0.5, 2)
        feasible = est_ram <= (usable_ram + 1.0)
        notes.append(
            f"Offload Parcial de Camadas: {gpu_layers}/{model.num_layers} camadas na GPU ({est_vram} GB VRAM) "
            f"e restante na RAM ({est_ram} GB)."
        )
        return SplitPlan(
            model_id=model.id,
            feasible=feasible,
            execution_mode="partial-gpu-dense",
            n_gpu_layers=gpu_layers,
            cpu_moe_layers=0,
            override_tensor_regex=None,
            recommended_ctx=ctx,
            kv_cache_type_k=chosen_kv,
            kv_cache_type_v=chosen_kv,
            flash_attention=True,
            cpu_threads=optimal_threads,
            estimated_vram_gb=est_vram,
            estimated_ram_gb=est_ram,
            estimated_tok_per_sec="6 - 14 tok/s",
            recommended_model_id=best_id,
            notes_pt=notes,
        )

    # Case 4: CPU-Only Fallback
    est_ram = round(model.size_gb + kv_gb + 0.5, 2)
    feasible = est_ram <= usable_ram
    notes.append("Modo somente CPU (nenhuma GPU CUDA com VRAM suficiente detectada).")
    return SplitPlan(
        model_id=model.id,
        feasible=feasible,
        execution_mode="cpu-only",
        n_gpu_layers=0,
        cpu_moe_layers=model.num_layers if model.is_moe else 0,
        override_tensor_regex=None,
        recommended_ctx=min(ctx, 16384),
        kv_cache_type_k="q4_0",
        kv_cache_type_v="q4_0",
        flash_attention=False,
        cpu_threads=optimal_threads,
        estimated_vram_gb=0.0,
        estimated_ram_gb=est_ram,
        estimated_tok_per_sec="4 - 10 tok/s" if model.is_moe else "2 - 5 tok/s",
        recommended_model_id=best_id,
        notes_pt=notes,
    )
