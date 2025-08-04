"""
Retrieval system for augmenting adapters with external knowledge.

This module provides:
- Vector index builders for document corpora
- Multiple retrieval backends (FAISS, Qdrant, Weaviate)
- Hybrid retrieval strategies
- Contextual and conversation-aware retrieval
"""

from .index_builder import VectorIndexBuilder

__all__ = [
    "VectorIndexBuilder",
]

# Lazy imports for components that will be implemented
def __getattr__(name):
    """Lazy import for retrieval components that may not be implemented yet."""
    if name == "BaseRetriever":
        from .retrievers import BaseRetriever
        return BaseRetriever
    elif name == "FAISSRetriever":
        from .retrievers import FAISSRetriever
        return FAISSRetriever
    elif name == "HybridRetriever":
        from .retrievers import HybridRetriever
        return HybridRetriever
    elif name == "ContextualRetriever":
        from .contextual import ContextualRetriever
        return ContextualRetriever
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")