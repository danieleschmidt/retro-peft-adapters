"""
Retro-PEFT-Adapters: Retrieval-Augmented Parameter-Efficient Fine-Tuning

This package provides retrieval-augmented parameter-efficient adapters that combine
frozen key/value caching with local vector databases for instant domain adaptation.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

# Core exports will be defined as implementation progresses
# from .adapters import RetroLoRA, RetroAdaLoRA, RetroIA3
# from .retrieval import VectorIndexBuilder, HybridRetriever
# from .training import ContrastiveRetrievalTrainer

__all__ = [
    "__version__",
    "__author__",
    "__email__",
]