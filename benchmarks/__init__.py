"""
Advanced Benchmarking Framework for Retro-PEFT-Adapters

This module provides comprehensive benchmarking capabilities for evaluating
retrieval-augmented parameter-efficient fine-tuning methods across multiple
dimensions: performance, accuracy, efficiency, and scalability.
"""

__version__ = "1.0.0"
__author__ = "Terragon Labs Research Team"

from .comparative_study import ComparativeAdapterStudy
from .performance_profiler import PerformanceProfiler
from .research_metrics import ResearchMetricsCollector
from .statistical_validator import StatisticalValidator
from .benchmark_runner import BenchmarkRunner

__all__ = [
    "ComparativeAdapterStudy",
    "PerformanceProfiler", 
    "ResearchMetricsCollector",
    "StatisticalValidator",
    "BenchmarkRunner"
]