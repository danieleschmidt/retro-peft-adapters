"""
Retrieval system for augmenting adapters with external knowledge.

This module provides:
- Vector index builders for document corpora
- Multiple retrieval backends (FAISS, Qdrant, Weaviate)
- Hybrid retrieval strategies
- Contextual and conversation-aware retrieval
"""

from .index_builder import VectorIndexBuilder
from .retrievers import (
    BaseRetriever, 
    FAISSRetriever, 
    QdrantRetriever, 
    WeaviateRetriever,
    HybridRetriever
)
from .contextual import ContextualRetriever
from .rerankers import CrossEncoderReranker, ListwiseReranker

__all__ = [
    "VectorIndexBuilder",
    "BaseRetriever",
    "FAISSRetriever", 
    "QdrantRetriever",
    "WeaviateRetriever", 
    "HybridRetriever",
    "ContextualRetriever",
    "CrossEncoderReranker",
    "ListwiseReranker",
]