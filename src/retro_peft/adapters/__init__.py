"""
Parameter-efficient adapters with retrieval augmentation.

This module implements various adapter types enhanced with retrieval capabilities:
- RetroLoRA: Low-rank adaptation with retrieval integration
- RetroAdaLoRA: Adaptive rank allocation based on retrieval importance
- RetroIA3: Lightweight adapters with retrieval scaling
"""

from .base_adapter import BaseRetroAdapter
from .retro_lora import RetroLoRA

__all__ = [
    "BaseRetroAdapter",
    "RetroLoRA",
]


# Lazy imports for adapters that will be implemented
def __getattr__(name):
    """Lazy import for adapters that may not be implemented yet."""
    if name == "RetroAdaLoRA":
        from .retro_adalora import RetroAdaLoRA

        return RetroAdaLoRA
    elif name == "RetroIA3":
        from .retro_ia3 import RetroIA3

        return RetroIA3
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
