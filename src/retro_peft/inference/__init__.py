"""
Inference utilities for retrieval-augmented adapters.

Provides optimized inference pipelines, caching, and deployment utilities
for production use of retro-PEFT adapters.
"""

from .pipeline import RetrievalInferencePipeline
from .batch_processor import BatchProcessor

__all__ = [
    "RetrievalInferencePipeline",
    "BatchProcessor",
]

# Lazy imports for optional components
def __getattr__(name):
    """Lazy import for optional inference components."""
    if name == "StreamingPipeline":
        from .streaming import StreamingPipeline
        return StreamingPipeline
    elif name == "AsyncProcessor":
        from .async_processor import AsyncProcessor
        return AsyncProcessor
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")