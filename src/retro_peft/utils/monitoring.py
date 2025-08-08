"""
Monitoring and metrics collection for retro-peft-adapters.

Provides comprehensive performance monitoring, health checks,
and metrics collection for production deployments.
"""

import collections
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

import psutil

from .logging import get_global_logger


@dataclass
class Metric:
    """Individual metric data point"""

    name: str
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class HealthCheck:
    """Health check result"""

    name: str
    status: str  # "healthy", "unhealthy", "degraded"
    message: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Comprehensive metrics collection system.

    Collects performance metrics, system resources, and application-specific
    metrics with time-series storage and aggregation capabilities.
    """

    def __init__(
        self,
        max_points: int = 10000,
        retention_hours: int = 24,
        collection_interval: float = 60.0,  # seconds
    ):
        """
        Initialize metrics collector.

        Args:
            max_points: Maximum number of data points to keep per metric
            retention_hours: How long to retain metrics data
            collection_interval: Interval for automatic system metrics collection
        """
        self.max_points = max_points
        self.retention_hours = retention_hours
        self.collection_interval = collection_interval

        # Storage for metrics
        self._metrics: Dict[str, List[Metric]] = collections.defaultdict(list)
        self._lock = threading.RLock()

        # System metrics collection
        self._collection_thread = None
        self._stop_collection = threading.Event()

        # Logger
        self.logger = get_global_logger()

        # Start automatic collection
        self.start_collection()

    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None,
        unit: str = "",
        timestamp: Optional[float] = None,
    ):
        """
        Record a metric data point.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for grouping/filtering
            unit: Unit of measurement
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()

        metric = Metric(name=name, value=value, timestamp=timestamp, tags=tags or {}, unit=unit)

        with self._lock:
            # Add metric
            self._metrics[name].append(metric)

            # Enforce max points limit
            if len(self._metrics[name]) > self.max_points:
                self._metrics[name] = self._metrics[name][-self.max_points :]

            # Clean old metrics
            self._cleanup_old_metrics()

    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Record a counter metric"""
        self.record_metric(name, value, tags, unit="count")

    def record_gauge(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = ""
    ):
        """Record a gauge metric"""
        self.record_metric(name, value, tags, unit=unit)

    def record_timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric"""
        self.record_metric(name, duration_ms, tags, unit="milliseconds")

    def get_metrics(
        self,
        name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        since: Optional[float] = None,
    ) -> Dict[str, List[Metric]]:
        """
        Get metrics with optional filtering.

        Args:
            name: Optional metric name filter
            tags: Optional tags filter
            since: Optional timestamp filter (get metrics since this time)

        Returns:
            Dictionary of metric name to list of metric points
        """
        with self._lock:
            result = {}

            metrics_to_check = [name] if name else self._metrics.keys()

            for metric_name in metrics_to_check:
                if metric_name not in self._metrics:
                    continue

                filtered_metrics = []
                for metric in self._metrics[metric_name]:
                    # Time filter
                    if since and metric.timestamp < since:
                        continue

                    # Tags filter
                    if tags:
                        if not all(metric.tags.get(key) == value for key, value in tags.items()):
                            continue

                    filtered_metrics.append(metric)

                if filtered_metrics:
                    result[metric_name] = filtered_metrics

            return result

    def get_metric_summary(self, name: str, since: Optional[float] = None) -> Dict[str, Any]:
        """
        Get summary statistics for a metric.

        Args:
            name: Metric name
            since: Optional timestamp filter

        Returns:
            Summary statistics
        """
        metrics = self.get_metrics(name, since=since)

        if name not in metrics or not metrics[name]:
            return {}

        values = [m.value for m in metrics[name]]

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values),
            "latest": values[-1] if values else None,
            "latest_timestamp": metrics[name][-1].timestamp if metrics[name] else None,
        }

    def start_collection(self):
        """Start automatic system metrics collection"""
        if self._collection_thread is None or not self._collection_thread.is_alive():
            self._stop_collection.clear()
            self._collection_thread = threading.Thread(
                target=self._collect_system_metrics, daemon=True
            )
            self._collection_thread.start()
            self.logger.info("Started system metrics collection")

    def stop_collection(self):
        """Stop automatic system metrics collection"""
        self._stop_collection.set()
        if self._collection_thread and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=5.0)
        self.logger.info("Stopped system metrics collection")

    def _collect_system_metrics(self):
        """Collect system metrics in background thread"""
        while not self._stop_collection.wait(self.collection_interval):
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_gauge("system.cpu.usage_percent", cpu_percent)

                # Memory metrics
                memory = psutil.virtual_memory()
                self.record_gauge("system.memory.usage_percent", memory.percent)
                self.record_gauge("system.memory.available_mb", memory.available / 1024 / 1024)
                self.record_gauge("system.memory.used_mb", memory.used / 1024 / 1024)

                # Disk metrics
                disk = psutil.disk_usage("/")
                self.record_gauge("system.disk.usage_percent", (disk.used / disk.total) * 100)
                self.record_gauge("system.disk.free_gb", disk.free / 1024 / 1024 / 1024)

                # Process metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                self.record_gauge("process.memory.rss_mb", process_memory.rss / 1024 / 1024)
                self.record_gauge("process.memory.vms_mb", process_memory.vms / 1024 / 1024)
                self.record_gauge("process.cpu.usage_percent", process.cpu_percent())

                # GPU metrics (if available)
                try:
                    import torch

                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            # Memory usage
                            memory_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                            memory_reserved = torch.cuda.memory_reserved(i) / 1024 / 1024

                            tags = {"device": f"cuda:{i}"}
                            self.record_gauge("gpu.memory.allocated_mb", memory_allocated, tags)
                            self.record_gauge("gpu.memory.reserved_mb", memory_reserved, tags)
                except ImportError:
                    pass  # PyTorch not available

            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")

    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = time.time() - (self.retention_hours * 3600)

        for metric_name in list(self._metrics.keys()):
            original_count = len(self._metrics[metric_name])

            # Filter out old metrics
            self._metrics[metric_name] = [
                m for m in self._metrics[metric_name] if m.timestamp >= cutoff_time
            ]

            # Remove empty metric series
            if not self._metrics[metric_name]:
                del self._metrics[metric_name]
            else:
                cleaned_count = original_count - len(self._metrics[metric_name])
                if cleaned_count > 0:
                    self.logger.debug(f"Cleaned {cleaned_count} old metrics for {metric_name}")

    def export_metrics(self, format: str = "json") -> str:
        """
        Export all metrics in specified format.

        Args:
            format: Export format ("json", "prometheus")

        Returns:
            Formatted metrics string
        """
        with self._lock:
            if format == "json":
                data = {}
                for name, metrics in self._metrics.items():
                    data[name] = [
                        {"value": m.value, "timestamp": m.timestamp, "tags": m.tags, "unit": m.unit}
                        for m in metrics
                    ]
                return json.dumps(data, indent=2)

            elif format == "prometheus":
                lines = []
                for name, metrics in self._metrics.items():
                    if not metrics:
                        continue

                    # Use latest value for Prometheus
                    latest = metrics[-1]

                    # Convert name to Prometheus format
                    prom_name = name.replace(".", "_").replace("-", "_")

                    # Add help and type
                    lines.append(f"# HELP {prom_name} {name}")
                    lines.append(f"# TYPE {prom_name} gauge")

                    # Add metric with tags
                    if latest.tags:
                        tag_str = ",".join(f'{k}="{v}"' for k, v in latest.tags.items())
                        lines.append(f"{prom_name}{{{tag_str}}} {latest.value}")
                    else:
                        lines.append(f"{prom_name} {latest.value}")

                return "\n".join(lines)

            else:
                raise ValueError(f"Unsupported export format: {format}")


class PerformanceMonitor:
    """
    Performance monitoring with automatic timing and resource tracking.
    """

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize performance monitor.

        Args:
            metrics_collector: Optional metrics collector instance
        """
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.logger = get_global_logger()

        # Active timers
        self._active_timers: Dict[str, float] = {}
        self._lock = threading.RLock()

    def start_timer(self, name: str) -> str:
        """
        Start a named timer.

        Args:
            name: Timer name

        Returns:
            Timer ID for stopping
        """
        timer_id = f"{name}_{int(time.time() * 1000000)}"

        with self._lock:
            self._active_timers[timer_id] = time.time()

        return timer_id

    def stop_timer(self, timer_id: str, tags: Optional[Dict[str, str]] = None) -> float:
        """
        Stop a timer and record the duration.

        Args:
            timer_id: Timer ID from start_timer
            tags: Optional tags for the metric

        Returns:
            Duration in milliseconds
        """
        end_time = time.time()

        with self._lock:
            if timer_id not in self._active_timers:
                self.logger.warning(f"Timer {timer_id} not found")
                return 0.0

            start_time = self._active_timers.pop(timer_id)

        duration_ms = (end_time - start_time) * 1000

        # Extract metric name from timer ID
        metric_name = timer_id.rsplit("_", 1)[0]

        # Record timing metric
        self.metrics_collector.record_timer(f"performance.{metric_name}", duration_ms, tags)

        return duration_ms

    def time_function(self, name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Decorator to time function execution.

        Args:
            name: Optional custom name (defaults to function name)
            tags: Optional tags for the metric
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                func_name = name or f"{func.__module__}.{func.__name__}"
                timer_id = self.start_timer(func_name)

                try:
                    result = func(*args, **kwargs)
                    duration_ms = self.stop_timer(timer_id, tags)

                    self.logger.debug(f"Function {func_name} completed in {duration_ms:.3f}ms")

                    return result

                except Exception as e:
                    duration_ms = self.stop_timer(timer_id, tags)

                    self.logger.error(f"Function {func_name} failed after {duration_ms:.3f}ms: {e}")
                    raise

            return wrapper

        return decorator

    def monitor_memory_usage(self, operation: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager to monitor memory usage during an operation.

        Args:
            operation: Operation name
            tags: Optional tags for metrics
        """
        return MemoryMonitorContext(self, operation, tags)

    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """
        Get performance summary for the last N hours.

        Args:
            hours: Number of hours to include

        Returns:
            Performance summary
        """
        since = time.time() - (hours * 3600)

        # Get all performance metrics
        perf_metrics = self.metrics_collector.get_metrics(since=since)

        summary = {"period_hours": hours, "operations": {}, "system": {}}

        # Analyze performance metrics
        for metric_name, metrics in perf_metrics.items():
            if metric_name.startswith("performance."):
                op_name = metric_name.replace("performance.", "")
                summary["operations"][op_name] = self.metrics_collector.get_metric_summary(
                    metric_name, since
                )
            elif metric_name.startswith("system."):
                system_metric = metric_name.replace("system.", "")
                summary["system"][system_metric] = self.metrics_collector.get_metric_summary(
                    metric_name, since
                )

        return summary


class MemoryMonitorContext:
    """Context manager for monitoring memory usage"""

    def __init__(
        self, monitor: PerformanceMonitor, operation: str, tags: Optional[Dict[str, str]] = None
    ):
        self.monitor = monitor
        self.operation = operation
        self.tags = tags or {}
        self.start_memory = 0.0
        self.process = psutil.Process()

    def __enter__(self):
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = end_memory - self.start_memory

        # Record memory metrics
        self.monitor.metrics_collector.record_gauge(
            f"memory.{self.operation}.delta_mb", memory_delta, self.tags
        )

        self.monitor.metrics_collector.record_gauge(
            f"memory.{self.operation}.peak_mb", end_memory, self.tags
        )


class HealthMonitor:
    """
    Health monitoring system for checking system and application health.
    """

    def __init__(self):
        self.checks: Dict[str, Callable[[], HealthCheck]] = {}
        self.logger = get_global_logger()

    def register_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """
        Register a health check.

        Args:
            name: Check name
            check_func: Function that returns HealthCheck result
        """
        self.checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")

    def run_checks(self) -> Dict[str, HealthCheck]:
        """
        Run all registered health checks.

        Returns:
            Dictionary of check results
        """
        results = {}

        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results[name] = result

                self.logger.debug(f"Health check {name}: {result.status} - {result.message}")

            except Exception as e:
                results[name] = HealthCheck(
                    name=name,
                    status="unhealthy",
                    message=f"Check failed: {e}",
                    timestamp=time.time(),
                    details={"error": str(e)},
                )

                self.logger.error(f"Health check {name} failed: {e}")

        return results

    def get_overall_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.

        Returns:
            Overall health summary
        """
        check_results = self.run_checks()

        if not check_results:
            return {
                "status": "unknown",
                "message": "No health checks configured",
                "timestamp": time.time(),
                "checks": {},
            }

        # Determine overall status
        statuses = [check.status for check in check_results.values()]

        if all(status == "healthy" for status in statuses):
            overall_status = "healthy"
            message = "All checks passing"
        elif any(status == "unhealthy" for status in statuses):
            overall_status = "unhealthy"
            unhealthy_count = sum(1 for s in statuses if s == "unhealthy")
            message = f"{unhealthy_count} checks failing"
        else:
            overall_status = "degraded"
            degraded_count = sum(1 for s in statuses if s == "degraded")
            message = f"{degraded_count} checks degraded"

        return {
            "status": overall_status,
            "message": message,
            "timestamp": time.time(),
            "checks": {
                name: {
                    "status": check.status,
                    "message": check.message,
                    "timestamp": check.timestamp,
                }
                for name, check in check_results.items()
            },
        }


# Global instances
_global_metrics_collector = None
_global_performance_monitor = None
_global_health_monitor = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _global_metrics_collector

    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()

    return _global_metrics_collector


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _global_performance_monitor

    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor(get_metrics_collector())

    return _global_performance_monitor


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance"""
    global _global_health_monitor

    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()

    return _global_health_monitor
