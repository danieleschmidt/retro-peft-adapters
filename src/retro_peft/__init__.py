"""
retro-peft-adapters
===================
Retrieval-Augmented Parameter-Efficient Fine-Tuning.

Core components:
    KeyValueCache   — frozen key/value memory store
    CacheBuilder    — build caches from (key, value) corpora
    RetroAdapter    — LoRA + retrieval adapter for nn.Linear
    AdapterBank     — multi-domain adapter with hot-swappable caches
    RetroPEFT       — full model wrapper; injects RetroAdapters at target layers
"""

from .cache import KeyValueCache
from .cache_builder import CacheBuilder
from .adapter import RetroAdapter
from .adapter_bank import AdapterBank
from .retro_peft import RetroPEFT

__version__ = "0.2.0"
__author__ = "Daniel Schmidt"

__all__ = [
    "KeyValueCache",
    "CacheBuilder",
    "RetroAdapter",
    "AdapterBank",
    "RetroPEFT",
]
