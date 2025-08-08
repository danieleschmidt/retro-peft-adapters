"""
Scaling and optimization components for retro-peft-adapters.

Provides high-performance, production-ready scaling features including
caching, async processing, load balancing, and distributed computing support.
"""

from .async_pipeline import AsyncBatchProcessor, AsyncRetrievalPipeline
from .cache import MultiLevelCache, VectorCache, get_cache_manager
from .load_balancer import LoadBalancer, RequestRouter
from .metrics import PerformanceAnalyzer, ScalingMetrics
from .resource_pool import ConnectionPool, ModelPool, get_resource_manager

__all__ = [
    "MultiLevelCache",
    "VectorCache",
    "get_cache_manager",
    "AsyncRetrievalPipeline",
    "AsyncBatchProcessor",
    "LoadBalancer",
    "RequestRouter",
    "ModelPool",
    "ConnectionPool",
    "get_resource_manager",
    "ScalingMetrics",
    "PerformanceAnalyzer",
]
