"""
Research Module for Retro-PEFT-Adapters

Advanced research implementations for academic publication and novel algorithm development.
Contains cutting-edge approaches to parameter-efficient fine-tuning with retrieval augmentation.

Modules:
- cross_modal_adaptive_retrieval: CARN model with multi-modal embeddings
- experimental_framework: Comprehensive benchmarking and validation
- statistical_analysis: Statistical significance testing and analysis
- publication_tools: Academic paper preparation utilities
"""

from .cross_modal_adaptive_retrieval import (
    CARNConfig,
    CrossModalAdaptiveRetrievalNetwork,
    MultiModalEmbeddingAligner,
    AdaptiveRetrievalWeighter,
    CrossDomainKnowledgeDistiller,
    ReinforcementBasedAdapterRanker,
    HierarchicalAttentionFusion,
    create_research_benchmark,
    run_carn_research_validation,
    demonstrate_carn_research
)

__all__ = [
    "CARNConfig",
    "CrossModalAdaptiveRetrievalNetwork", 
    "MultiModalEmbeddingAligner",
    "AdaptiveRetrievalWeighter",
    "CrossDomainKnowledgeDistiller",
    "ReinforcementBasedAdapterRanker",
    "HierarchicalAttentionFusion",
    "create_research_benchmark",
    "run_carn_research_validation",
    "demonstrate_carn_research"
]