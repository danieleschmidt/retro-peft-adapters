"""
Parameter-efficient adapters with retrieval augmentation.

This module implements various adapter types enhanced with retrieval capabilities:
- RetroLoRA: Low-rank adaptation with retrieval integration
- RetroAdaLoRA: Adaptive rank allocation based on retrieval importance
- RetroIA3: Lightweight adapters with retrieval scaling
"""

from .retro_lora import RetroLoRA
from .retro_adalora import RetroAdaLoRA
from .retro_ia3 import RetroIA3
from .base_adapter import BaseRetroAdapter

__all__ = [
    "RetroLoRA",
    "RetroAdaLoRA", 
    "RetroIA3",
    "BaseRetroAdapter",
]