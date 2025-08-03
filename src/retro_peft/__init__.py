"""
Retro-PEFT-Adapters: Retrieval-Augmented Parameter-Efficient Fine-Tuning

This package provides retrieval-augmented parameter-efficient adapters that combine
frozen key/value caching with local vector databases for instant domain adaptation.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .adapters import RetroLoRA, RetroAdaLoRA, RetroIA3
from .retrieval import VectorIndexBuilder, HybridRetriever, ContextualRetriever
from .training import ContrastiveRetrievalTrainer, MultiTaskRetroTrainer
from .fusion import CrossAttentionFusion, GatedFusion
from .caching import FrozenKVCache, HierarchicalCache
from .serving import RetroAdapterServer, AdapterRouter

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "RetroLoRA",
    "RetroAdaLoRA", 
    "RetroIA3",
    "VectorIndexBuilder",
    "HybridRetriever",
    "ContextualRetriever",
    "ContrastiveRetrievalTrainer",
    "MultiTaskRetroTrainer",
    "CrossAttentionFusion",
    "GatedFusion",
    "FrozenKVCache",
    "HierarchicalCache",
    "RetroAdapterServer",
    "AdapterRouter",
]