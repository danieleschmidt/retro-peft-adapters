"""
Database layer for retro-peft adapters.

Provides data persistence for:
- Training metadata and configurations
- Model checkpoints and versioning
- Retrieval indices and embeddings
- Usage analytics and performance metrics
"""

from .connection import DatabaseManager
from .models import AdapterConfig, PerformanceMetrics, RetrievalIndex, TrainingRun
from .repositories import (
    AdapterRepository,
    MetricsRepository,
    RetrievalRepository,
    TrainingRepository,
)

__all__ = [
    "DatabaseManager",
    "AdapterConfig",
    "TrainingRun",
    "RetrievalIndex",
    "PerformanceMetrics",
    "AdapterRepository",
    "TrainingRepository",
    "RetrievalRepository",
    "MetricsRepository",
]
