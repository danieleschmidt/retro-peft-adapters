"""
Scaling and performance optimization components.

Provides async processing, high-performance caching, and distributed capabilities.
"""

# Import only modules that work without heavy dependencies
try:
    from .high_performance_cache import (
        MemoryCache, MultiLevelCache, EmbeddingCache, CacheManager
    )
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False

try:
    from .async_processing import (
        AsyncBatchProcessor, AsyncRetriever, ConcurrentAdapterPool
    )
    _ASYNC_AVAILABLE = True
except ImportError:
    _ASYNC_AVAILABLE = False

# Basic exports
__all__ = []

if _CACHE_AVAILABLE:
    __all__.extend([
        "MemoryCache", "MultiLevelCache", "EmbeddingCache", "CacheManager"
    ])

if _ASYNC_AVAILABLE:
    __all__.extend([
        "AsyncBatchProcessor", "AsyncRetriever", "ConcurrentAdapterPool"
    ])
