"""Triple-Stack Inference Engine package."""

from .catalog import MODEL_CATALOG, ModelSpec, get_model_spec, list_models
from .hardware import HardwareProfile, SplitPlan, calculate_surgical_split, detect_hardware

__all__ = [
    "MODEL_CATALOG",
    "ModelSpec",
    "get_model_spec",
    "list_models",
    "HardwareProfile",
    "SplitPlan",
    "calculate_surgical_split",
    "detect_hardware",
]
