"""Tests for Hardware Profiler & Surgical GPU+RAM Tensor Split (including Target RTX 3050 8GB + 16GB RAM rig)."""

from nglai.engine.catalog import get_model_spec
from nglai.engine.hardware import GPUInfo, HardwareProfile, calculate_surgical_split


def test_target_rig_rtx3050_8gb_16gb_ram_qwen36_35b_moe() -> None:
    """Verify Qwen3.6-35B-A3B (13.1 GB) fits safely across 8GB VRAM + 16GB RAM via surgical split."""
    target_hw = HardwareProfile(
        os_name="Windows 11",
        cpu_name="Intel Core i5-10400F",
        cpu_physical_cores=6,
        cpu_logical_threads=12,
        ram_total_gb=16.0,
        ram_available_gb=11.2,
        gpus=[
            GPUInfo(
                index=0,
                name="NVIDIA GeForce RTX 3050",
                vram_total_mb=8192,
                vram_free_mb=7200,
                driver_version="572.16",
            )
        ],
    )
    model = get_model_spec("qwen3.6-35b-a3b-ud-iq3")
    assert model is not None

    plan = calculate_surgical_split(target_hw, model, requested_ctx=32768, kv_mode="auto")

    assert plan.feasible is True
    assert plan.execution_mode == "surgical-hybrid-moe"
    assert plan.estimated_vram_gb <= 7.0
    assert plan.estimated_ram_gb <= 8.5
    assert 0 < plan.cpu_moe_layers < model.num_layers
    assert plan.override_tensor_regex is not None
    assert "ffn_(up|down|gate)_exps" in plan.override_tensor_regex

    # TurboQuant supported vs fallback CLI flags
    turbo_args = plan.to_llama_args(turboquant_supported=True)
    assert "turbo3" in turbo_args
    fallback_args = plan.to_llama_args(turboquant_supported=False)
    assert "q4_0" in fallback_args


def test_dev_laptop_gtx1650_4gb_32gb_ram_qwen3_coder_30b() -> None:
    """Verify 4GB VRAM + 32GB RAM keeps attention on GPU and offloads MoE experts cleanly to 32GB RAM."""
    dev_hw = HardwareProfile(
        os_name="Windows 11",
        cpu_name="Intel Core i5-9300H",
        cpu_physical_cores=4,
        cpu_logical_threads=8,
        ram_total_gb=31.83,
        ram_available_gb=22.0,
        gpus=[
            GPUInfo(
                index=0,
                name="NVIDIA GeForce GTX 1650",
                vram_total_mb=4096,
                vram_free_mb=3400,
                driver_version="617.42",
            )
        ],
    )
    model = get_model_spec("qwen3-coder-30b-a3b-ud-iq3")
    assert model is not None

    plan = calculate_surgical_split(dev_hw, model, requested_ctx=32768)
    assert plan.feasible is True
    assert plan.execution_mode == "surgical-hybrid-moe"
    assert plan.estimated_vram_gb <= 3.3
    assert plan.estimated_ram_gb <= 13.0


def test_100_percent_gpu_mode_when_vram_allows() -> None:
    hw = HardwareProfile(
        os_name="Windows 11",
        cpu_name="Intel Core i5-10400F",
        cpu_physical_cores=6,
        cpu_logical_threads=12,
        ram_total_gb=16.0,
        ram_available_gb=11.0,
        gpus=[GPUInfo(0, "RTX 3050", 8192, 7400, "572.16")],
    )
    draft = get_model_spec("qwen2.5-coder-1.5b-draft")
    assert draft is not None
    plan = calculate_surgical_split(hw, draft, requested_ctx=16384)
    assert plan.execution_mode == "100%-gpu"
    assert plan.override_tensor_regex is None
