"""
Retro-PEFT-Adapters: Retrieval-Augmented Parameter-Efficient Fine-Tuning

This package provides retrieval-augmented parameter-efficient adapters that combine
frozen key/value caching with local vector databases for instant domain adaptation.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

# Import only what exists currently (lazy loading to avoid import errors)
def __getattr__(name):
    """Lazy import for optional components that may not be available."""
    if name == "BaseRetroAdapter":
        from .adapters import BaseRetroAdapter
        return BaseRetroAdapter
    elif name == "RetroLoRA":
        from .adapters import RetroLoRA
        return RetroLoRA
    elif name == "RetroAdaLoRA":
        from .adapters import RetroAdaLoRA
        return RetroAdaLoRA
    elif name == "RetroIA3":
        from .adapters import RetroIA3
        return RetroIA3
    elif name == "VectorIndexBuilder":
        from .retrieval import VectorIndexBuilder
        return VectorIndexBuilder
    elif name == "HybridRetriever":
        from .retrieval import HybridRetriever
        return HybridRetriever
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Export what's available
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "BaseRetroAdapter", 
    "RetroLoRA",
    "RetroAdaLoRA",
    "RetroIA3",
    "VectorIndexBuilder",
    "HybridRetriever",
]
