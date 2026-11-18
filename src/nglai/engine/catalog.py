"""Triple-Stack Compressed Model Catalog (Unsloth Dynamic 2.0 GGUF + MoE Metadata)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for a curated coding LLM optimized for local compressed execution."""

    id: str
    name: str
    family: str
    hf_repo: str
    filename: str
    size_gb: float
    total_params_b: float
    active_params_b: float
    is_moe: bool
    num_layers: int
    quant_method: str
    bits_per_weight: float
    max_native_ctx: int
    supports_thinking: bool
    supports_tools: bool
    tier: str  # "flagship-moe", "dense-specialist", "draft-accelerator"
    description_pt: str

    @property
    def hf_download_url(self) -> str:
        return f"https://huggingface.co/{self.hf_repo}/resolve/main/{self.filename}"


MODEL_CATALOG: Dict[str, ModelSpec] = {
    "qwen3.6-35b-a3b-ud-iq3": ModelSpec(
        id="qwen3.6-35b-a3b-ud-iq3",
        name="Qwen3.6-35B-A3B (Unsloth UD-IQ3_XXS)",
        family="qwen3.6",
        hf_repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        filename="Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
        size_gb=13.1,
        total_params_b=35.0,
        active_params_b=3.0,
        is_moe=True,
        num_layers=48,
        quant_method="Unsloth Dynamic 2.0 (KLD Mixed-Precision ~3.06 bpw)",
        bits_per_weight=3.06,
        max_native_ctx=262144,
        supports_thinking=True,
        supports_tools=True,
        tier="flagship-moe",
        description_pt=(
            "Cérebro Principal Recomendado p/ 8GB VRAM + 16GB RAM: 35B totais (apenas 3B ativos/token), "
            "Atenção Linear Híbrida + MoE esparso + modo Deep Thinking. Cabe dividindo ~6.5GB na GPU e ~6.6GB na RAM."
        ),
    ),
    "qwen3.6-35b-a3b-ud-q2kxl": ModelSpec(
        id="qwen3.6-35b-a3b-ud-q2kxl",
        name="Qwen3.6-35B-A3B (Unsloth UD-Q2_K_XL)",
        family="qwen3.6",
        hf_repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        filename="Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf",
        size_gb=12.1,
        total_params_b=35.0,
        active_params_b=3.0,
        is_moe=True,
        num_layers=48,
        quant_method="Unsloth Dynamic 2.0 (UD-Q2_K_XL ~2.82 bpw)",
        bits_per_weight=2.82,
        max_native_ctx=262144,
        supports_thinking=True,
        supports_tools=True,
        tier="flagship-moe",
        description_pt=(
            "Versão ultra-compacta de 35B MoE: deixa ainda mais folga de RAM no Windows mantendo "
            "camadas críticas de atenção e tool-calling protegidas em alta precisão."
        ),
    ),
    "qwen3-coder-30b-a3b-ud-iq3": ModelSpec(
        id="qwen3-coder-30b-a3b-ud-iq3",
        name="Qwen3-Coder-30B-A3B-Instruct (Unsloth UD-IQ3_XXS)",
        family="qwen3-coder",
        hf_repo="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        filename="Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf",
        size_gb=12.8,
        total_params_b=30.5,
        active_params_b=3.3,
        is_moe=True,
        num_layers=48,
        quant_method="Unsloth Dynamic 2.0 (UD-IQ3_XXS ~3.15 bpw)",
        bits_per_weight=3.15,
        max_native_ctx=262144,
        supports_thinking=False,
        supports_tools=True,
        tier="flagship-moe",
        description_pt=(
            "Especialista Puro em Engenharia de Repositórios (SWE-bench / Aider Polyglot): 30.5B totais "
            "(3.3B ativos), contexto nativo de 256K e execução cirúrgica de ferramentas."
        ),
    ),
    "qwen3-coder-30b-a3b-ud-q3kxl": ModelSpec(
        id="qwen3-coder-30b-a3b-ud-q3kxl",
        name="Qwen3-Coder-30B-A3B-Instruct (Unsloth UD-Q3_K_XL)",
        family="qwen3-coder",
        hf_repo="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        filename="Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf",
        size_gb=13.8,
        total_params_b=30.5,
        active_params_b=3.3,
        is_moe=True,
        num_layers=48,
        quant_method="Unsloth Dynamic 2.0 (UD-Q3_K_XL ~3.45 bpw)",
        bits_per_weight=3.45,
        max_native_ctx=262144,
        supports_thinking=False,
        supports_tools=True,
        tier="flagship-moe",
        description_pt=(
            "Fidelidade extra em 3-bit XL: ideal para 8GB VRAM + 16GB RAM (modo focado) ou sistemas com 24GB-32GB RAM."
        ),
    ),
    "qwen3-coder-30b-a3b-ud-q4kxl": ModelSpec(
        id="qwen3-coder-30b-a3b-ud-q4kxl",
        name="Qwen3-Coder-30B-A3B-Instruct (Unsloth UD-Q4_K_XL)",
        family="qwen3-coder",
        hf_repo="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        filename="Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
        size_gb=17.7,
        total_params_b=30.5,
        active_params_b=3.3,
        is_moe=True,
        num_layers=48,
        quant_method="Unsloth Dynamic 2.0 (UD-Q4_K_XL ~4.50 bpw)",
        bits_per_weight=4.50,
        max_native_ctx=262144,
        supports_thinking=False,
        supports_tools=True,
        tier="flagship-moe",
        description_pt=(
            "Perfil 32GB RAM / Upgrade: fidelidade praticamente idêntica ao BF16 original para máquinas com >=32GB RAM."
        ),
    ),
    "qwen2.5-coder-14b-ud-iq3m": ModelSpec(
        id="qwen2.5-coder-14b-ud-iq3m",
        name="Qwen2.5-Coder-14B-Instruct (Unsloth IQ3_M)",
        family="qwen2.5-coder",
        hf_repo="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        filename="qwen2.5-coder-14b-instruct-q3_k_m.gguf",
        size_gb=6.9,
        total_params_b=14.7,
        active_params_b=14.7,
        is_moe=False,
        num_layers=48,
        quant_method="Compressed 3-bit K-Quant",
        bits_per_weight=3.70,
        max_native_ctx=131072,
        supports_thinking=False,
        supports_tools=True,
        tier="dense-specialist",
        description_pt=(
            "Alternativa 100% GPU para placas de 8GB VRAM: roda inteiramente dentro da RTX 3050 com TurboQuant."
        ),
    ),
    "qwen2.5-coder-1.5b-draft": ModelSpec(
        id="qwen2.5-coder-1.5b-draft",
        name="Qwen2.5-Coder-1.5B-Instruct (Draft Speculative)",
        family="qwen2.5-coder",
        hf_repo="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        filename="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        size_gb=1.1,
        total_params_b=1.5,
        active_params_b=1.5,
        is_moe=False,
        num_layers=28,
        quant_method="Q4_K_M",
        bits_per_weight=4.8,
        max_native_ctx=32768,
        supports_thinking=False,
        supports_tools=False,
        tier="draft-accelerator",
        description_pt=(
            "Micro-modelo rascunho opcional para Speculative Decoding ou testes rápidos de sanidade (<1.1 GB)."
        ),
    ),
}


def get_model_spec(model_id: str) -> Optional[ModelSpec]:
    return MODEL_CATALOG.get(model_id)


def list_models(tier: Optional[str] = None) -> List[ModelSpec]:
    specs = list(MODEL_CATALOG.values())
    if tier:
        specs = [s for s in specs if s.tier == tier]
    return specs
