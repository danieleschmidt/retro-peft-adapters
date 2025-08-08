"""
Advanced metrics and performance analytics for scaling infrastructure.

Provides comprehensive performance monitoring, analytics, and
observability for production deployments.
"""

import json
import os
import statistics
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils.logging import get_global_logger
from ..utils.monitoring import MetricsCollector, get_metrics_collector


@dataclass
class PerformanceMetric:
    """Performance metric data point"""

    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    percentiles: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "percentiles": self.percentiles,
        }


@dataclass
class ThroughputMetric:
    """Throughput measurement"""

    requests_per_second: float
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float
    concurrent_requests: int
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "requests_per_second": self.requests_per_second,
            "avg_response_time": self.avg_response_time,
            "p95_response_time": self.p95_response_time,
            "p99_response_time": self.p99_response_time,
            "error_rate": self.error_rate,
            "concurrent_requests": self.concurrent_requests,
            "timestamp": self.timestamp,
        }


class LatencyTracker:
    """
    High-performance latency tracking with percentile calculations.
    """

    def __init__(self, window_size: int = 1000, percentiles: List[float] = None):
        """
        Initialize latency tracker.

        Args:
            window_size: Size of sliding window for calculations
            percentiles: Percentiles to calculate (e.g., [50, 95, 99])
        """
        self.window_size = window_size
        self.percentiles = percentiles or [50, 90, 95, 99]

        # Sliding window of latencies
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._sorted_latencies: List[float] = []
        self._dirty = False

        # Statistics
        self._total_requests = 0
        self._sum_latencies = 0.0

        self._lock = threading.RLock()

    def record_latency(self, latency_ms: float):
        """
        Record a latency measurement.

        Args:
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            self._latencies.append(latency_ms)
            self._total_requests += 1
            self._sum_latencies += latency_ms
            self._dirty = True

    def get_percentiles(self) -> Dict[str, float]:
        """
        Get latency percentiles.

        Returns:
            Dictionary of percentile -> latency_ms
        """
        with self._lock:
            if not self._latencies:
                return {f"p{p}": 0.0 for p in self.percentiles}

            # Update sorted list if needed
            if self._dirty:
                self._sorted_latencies = sorted(self._latencies)
                self._dirty = False

            result = {}
            n = len(self._sorted_latencies)

            for percentile in self.percentiles:
                # Calculate percentile index
                index = (percentile / 100.0) * (n - 1)

                if index == int(index):
                    # Exact index
                    value = self._sorted_latencies[int(index)]
                else:
                    # Interpolate between two values
                    lower_index = int(index)
                    upper_index = min(lower_index + 1, n - 1)

                    lower_value = self._sorted_latencies[lower_index]
                    upper_value = self._sorted_latencies[upper_index]

                    # Linear interpolation
                    fraction = index - lower_index
                    value = lower_value + fraction * (upper_value - lower_value)

                result[f"p{percentile}"] = value

            return result

    def get_avg_latency(self) -> float:
        """Get average latency"""
        with self._lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive latency statistics"""
        with self._lock:
            if not self._latencies:
                return {
                    "count": 0,
                    "avg": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "percentiles": {f"p{p}": 0.0 for p in self.percentiles},
                }

            percentiles = self.get_percentiles()

            return {
                "count": len(self._latencies),
                "avg": self.get_avg_latency(),
                "min": min(self._latencies),
                "max": max(self._latencies),
                "percentiles": percentiles,
                "total_requests": self._total_requests,
            }


class ThroughputTracker:
    """
    Throughput tracking with sliding window calculations.
    """

    def __init__(self, window_duration: float = 60.0):
        """
        Initialize throughput tracker.

        Args:
            window_duration: Duration of sliding window in seconds
        """
        self.window_duration = window_duration

        # Request timestamps
        self._request_times: deque[float] = deque()
        self._error_times: deque[float] = deque()

        # Concurrent request tracking
        self._active_requests = 0
        self._max_concurrent = 0

        self._lock = threading.RLock()

    def record_request_start(self):
        """Record request start"""
        with self._lock:
            now = time.time()
            self._request_times.append(now)
            self._active_requests += 1
            self._max_concurrent = max(self._max_concurrent, self._active_requests)

            # Clean old entries
            self._cleanup_old_entries(now)

    def record_request_end(self, error: bool = False):
        """Record request end"""
        with self._lock:
            now = time.time()

            if error:
                self._error_times.append(now)

            self._active_requests = max(0, self._active_requests - 1)

            # Clean old entries
            self._cleanup_old_entries(now)

    def get_throughput(self) -> Dict[str, float]:
        """
        Get current throughput metrics.

        Returns:
            Dictionary with throughput statistics
        """
        with self._lock:
            now = time.time()
            self._cleanup_old_entries(now)

            # Calculate requests per second
            total_requests = len(self._request_times)
            rps = total_requests / self.window_duration if self.window_duration > 0 else 0.0

            # Calculate error rate
            total_errors = len(self._error_times)
            error_rate = total_errors / total_requests if total_requests > 0 else 0.0

            return {
                "requests_per_second": rps,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": error_rate,
                "active_requests": self._active_requests,
                "max_concurrent": self._max_concurrent,
            }

    def _cleanup_old_entries(self, now: float):
        """Remove entries outside the window"""
        cutoff_time = now - self.window_duration

        # Remove old request times
        while self._request_times and self._request_times[0] < cutoff_time:
            self._request_times.popleft()

        # Remove old error times
        while self._error_times and self._error_times[0] < cutoff_time:
            self._error_times.popleft()


class ResourceUsageTracker:
    """
    Track resource usage patterns and identify bottlenecks.
    """

    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize resource usage tracker.

        Args:
            sampling_interval: How often to sample resource usage
        """
        self.sampling_interval = sampling_interval

        # Resource metrics
        self._cpu_usage: deque[float] = deque(maxlen=300)  # 5 minutes at 1s intervals
        self._memory_usage: deque[float] = deque(maxlen=300)
        self._gpu_usage: deque[float] = deque(maxlen=300)
        self._gpu_memory: deque[float] = deque(maxlen=300)

        # Threading
        self._sampling_thread = None
        self._sampling_stop = threading.Event()

        self.logger = get_global_logger()
        self._lock = threading.RLock()

        # Start sampling
        self._start_sampling()

    def _start_sampling(self):
        """Start background resource sampling"""
        if self._sampling_thread is None or not self._sampling_thread.is_alive():
            self._sampling_stop.clear()
            self._sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
            self._sampling_thread.start()

    def _sampling_loop(self):
        """Background sampling loop"""
        while not self._sampling_stop.wait(self.sampling_interval):
            try:
                self._sample_resources()
            except Exception as e:
                self.logger.error(f"Resource sampling error: {e}")

    def _sample_resources(self):
        """Sample current resource usage"""
        with self._lock:
            try:
                import psutil

                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self._cpu_usage.append(cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                self._memory_usage.append(memory.percent)

                # GPU usage (if available)
                try:
                    import torch

                    if torch.cuda.is_available():
                        # GPU utilization (approximation)
                        gpu_memory_allocated = torch.cuda.memory_allocated(0)
                        gpu_memory_reserved = torch.cuda.memory_reserved(0)

                        if gpu_memory_reserved > 0:
                            gpu_usage = (gpu_memory_allocated / gpu_memory_reserved) * 100
                            self._gpu_usage.append(gpu_usage)
                            self._gpu_memory.append(gpu_memory_allocated / (1024**3))  # GB

                except ImportError:
                    pass

            except Exception as e:
                self.logger.error(f"Error sampling resources: {e}")

    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics"""
        with self._lock:
            stats = {}

            # CPU stats
            if self._cpu_usage:
                stats["cpu"] = {
                    "current": self._cpu_usage[-1],
                    "avg": statistics.mean(self._cpu_usage),
                    "max": max(self._cpu_usage),
                    "min": min(self._cpu_usage),
                }

            # Memory stats
            if self._memory_usage:
                stats["memory"] = {
                    "current": self._memory_usage[-1],
                    "avg": statistics.mean(self._memory_usage),
                    "max": max(self._memory_usage),
                    "min": min(self._memory_usage),
                }

            # GPU stats
            if self._gpu_usage:
                stats["gpu"] = {
                    "usage_current": self._gpu_usage[-1],
                    "usage_avg": statistics.mean(self._gpu_usage),
                    "memory_gb": self._gpu_memory[-1] if self._gpu_memory else 0,
                }

            return stats

    def stop_sampling(self):
        """Stop resource sampling"""
        self._sampling_stop.set()
        if self._sampling_thread and self._sampling_thread.is_alive():
            self._sampling_thread.join(timeout=5.0)


class ScalingMetrics:
    """
    Comprehensive scaling metrics collector.
    """

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        latency_window: int = 1000,
        throughput_window: float = 60.0,
    ):
        """
        Initialize scaling metrics.

        Args:
            metrics_collector: Base metrics collector
            latency_window: Latency tracking window size
            throughput_window: Throughput tracking window duration
        """
        self.metrics_collector = metrics_collector or get_metrics_collector()

        # Specialized trackers
        self.latency_tracker = LatencyTracker(window_size=latency_window)
        self.throughput_tracker = ThroughputTracker(window_duration=throughput_window)
        self.resource_tracker = ResourceUsageTracker()

        # Service-specific metrics
        self._service_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)

        self.logger = get_global_logger()
        self._lock = threading.RLock()

    def record_request(
        self,
        service: str,
        latency_ms: float,
        success: bool = True,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Record a service request.

        Args:
            service: Service name
            latency_ms: Request latency in milliseconds
            success: Whether request was successful
            labels: Optional labels for grouping
        """
        # Record in latency tracker
        self.latency_tracker.record_latency(latency_ms)

        # Record in throughput tracker
        self.throughput_tracker.record_request_end(error=not success)

        # Record in base metrics collector
        self.metrics_collector.record_timer(f"service.{service}.latency", latency_ms, tags=labels)

        self.metrics_collector.record_counter(
            f"service.{service}.requests_total", 1, tags={**(labels or {}), "success": str(success)}
        )

        # Update service-specific metrics
        with self._lock:
            if service not in self._service_metrics:
                self._service_metrics[service] = {
                    "latency_tracker": LatencyTracker(),
                    "throughput_tracker": ThroughputTracker(),
                }

            self._service_metrics[service]["latency_tracker"].record_latency(latency_ms)
            self._service_metrics[service]["throughput_tracker"].record_request_end(
                error=not success
            )

    def start_request(self, service: str) -> str:
        """
        Start tracking a request.

        Args:
            service: Service name

        Returns:
            Request ID for tracking
        """
        request_id = f"{service}_{int(time.time() * 1000000)}"

        # Record start in throughput tracker
        self.throughput_tracker.record_request_start()

        # Record start in service-specific tracker
        with self._lock:
            if service not in self._service_metrics:
                self._service_metrics[service] = {
                    "latency_tracker": LatencyTracker(),
                    "throughput_tracker": ThroughputTracker(),
                }

            self._service_metrics[service]["throughput_tracker"].record_request_start()

        return request_id

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            "timestamp": time.time(),
            "latency": self.latency_tracker.get_stats(),
            "throughput": self.throughput_tracker.get_throughput(),
            "resources": self.resource_tracker.get_resource_stats(),
            "services": {},
        }

        # Add service-specific metrics
        with self._lock:
            for service, trackers in self._service_metrics.items():
                summary["services"][service] = {
                    "latency": trackers["latency_tracker"].get_stats(),
                    "throughput": trackers["throughput_tracker"].get_throughput(),
                }

        return summary

    def get_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get scaling recommendations based on current metrics.

        Returns:
            List of recommendations
        """
        recommendations = []
        performance = self.get_performance_summary()

        # High latency recommendation
        latency_stats = performance["latency"]
        if latency_stats["percentiles"]["p95"] > 1000:  # > 1 second
            recommendations.append(
                {
                    "type": "scale_up",
                    "reason": "High P95 latency",
                    "metric": "p95_latency",
                    "value": latency_stats["percentiles"]["p95"],
                    "threshold": 1000,
                    "priority": "high",
                }
            )

        # High error rate recommendation
        throughput_stats = performance["throughput"]
        if throughput_stats["error_rate"] > 0.05:  # > 5% error rate
            recommendations.append(
                {
                    "type": "investigate_errors",
                    "reason": "High error rate",
                    "metric": "error_rate",
                    "value": throughput_stats["error_rate"],
                    "threshold": 0.05,
                    "priority": "critical",
                }
            )

        # High resource usage recommendation
        resource_stats = performance["resources"]
        if "cpu" in resource_stats and resource_stats["cpu"]["current"] > 80:
            recommendations.append(
                {
                    "type": "scale_out",
                    "reason": "High CPU usage",
                    "metric": "cpu_usage",
                    "value": resource_stats["cpu"]["current"],
                    "threshold": 80,
                    "priority": "medium",
                }
            )

        if "memory" in resource_stats and resource_stats["memory"]["current"] > 85:
            recommendations.append(
                {
                    "type": "scale_up",
                    "reason": "High memory usage",
                    "metric": "memory_usage",
                    "value": resource_stats["memory"]["current"],
                    "threshold": 85,
                    "priority": "high",
                }
            )

        # Low utilization recommendation
        if (
            throughput_stats["requests_per_second"] < 1.0
            and "cpu" in resource_stats
            and resource_stats["cpu"]["avg"] < 20
        ):
            recommendations.append(
                {
                    "type": "scale_down",
                    "reason": "Low utilization",
                    "metric": "requests_per_second",
                    "value": throughput_stats["requests_per_second"],
                    "threshold": 1.0,
                    "priority": "low",
                }
            )

        return recommendations

    def export_metrics(self, format: str = "prometheus") -> str:
        """
        Export metrics in specified format.

        Args:
            format: Export format ("prometheus", "json")

        Returns:
            Formatted metrics string
        """
        if format == "prometheus":
            return self._export_prometheus()
        elif format == "json":
            return self._export_json()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        performance = self.get_performance_summary()

        # Latency metrics
        latency_stats = performance["latency"]
        for percentile, value in latency_stats["percentiles"].items():
            lines.append(f"# TYPE latency_percentile gauge")
            lines.append(f'latency_percentile{{percentile="{percentile}"}} {value}')

        # Throughput metrics
        throughput_stats = performance["throughput"]
        lines.append(f"# TYPE requests_per_second gauge")
        lines.append(f"requests_per_second {throughput_stats['requests_per_second']}")

        lines.append(f"# TYPE error_rate gauge")
        lines.append(f"error_rate {throughput_stats['error_rate']}")

        # Resource metrics
        resource_stats = performance["resources"]
        if "cpu" in resource_stats:
            lines.append(f"# TYPE cpu_usage gauge")
            lines.append(f"cpu_usage {resource_stats['cpu']['current']}")

        if "memory" in resource_stats:
            lines.append(f"# TYPE memory_usage gauge")
            lines.append(f"memory_usage {resource_stats['memory']['current']}")

        return "\n".join(lines)

    def _export_json(self) -> str:
        """Export metrics in JSON format"""
        performance = self.get_performance_summary()
        return json.dumps(performance, indent=2)

    def stop(self):
        """Stop all tracking"""
        self.resource_tracker.stop_sampling()


class PerformanceAnalyzer:
    """
    Advanced performance analyzer with trend detection and anomaly detection.
    """

    def __init__(self, history_size: int = 1000):
        """
        Initialize performance analyzer.

        Args:
            history_size: Number of historical data points to keep
        """
        self.history_size = history_size

        # Historical data
        self._performance_history: deque[Dict[str, Any]] = deque(maxlen=history_size)

        self.logger = get_global_logger()
        self._lock = threading.RLock()

    def record_performance_snapshot(self, metrics: Dict[str, Any]):
        """
        Record a performance snapshot.

        Args:
            metrics: Performance metrics dictionary
        """
        with self._lock:
            snapshot = {"timestamp": time.time(), **metrics}
            self._performance_history.append(snapshot)

    def detect_trends(self, metric_path: str, window_size: int = 10) -> Dict[str, Any]:
        """
        Detect trends in a specific metric.

        Args:
            metric_path: Dot-separated path to metric (e.g., "latency.avg")
            window_size: Size of window for trend analysis

        Returns:
            Trend analysis results
        """
        with self._lock:
            if len(self._performance_history) < window_size:
                return {"trend": "insufficient_data", "confidence": 0.0}

            # Extract metric values
            values = []
            timestamps = []

            for snapshot in list(self._performance_history)[-window_size:]:
                value = self._get_nested_value(snapshot, metric_path)
                if value is not None:
                    values.append(value)
                    timestamps.append(snapshot["timestamp"])

            if len(values) < 3:
                return {"trend": "insufficient_data", "confidence": 0.0}

            # Calculate trend using linear regression
            n = len(values)
            x = list(range(n))

            # Calculate slope
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(x[i] * values[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

            # Calculate correlation coefficient for confidence
            mean_x = sum_x / n
            mean_y = sum_y / n

            numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
            denominator = (
                sum((x[i] - mean_x) ** 2 for i in range(n))
                * sum((values[i] - mean_y) ** 2 for i in range(n))
            ) ** 0.5

            correlation = numerator / denominator if denominator != 0 else 0
            confidence = abs(correlation)

            # Determine trend direction
            if abs(slope) < 0.001:  # Very small slope
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"

            return {
                "trend": trend,
                "slope": slope,
                "confidence": confidence,
                "data_points": n,
                "latest_value": values[-1],
                "change_rate": slope / mean_y if mean_y != 0 else 0,
            }

    def detect_anomalies(self, metric_path: str, sensitivity: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metric using statistical methods.

        Args:
            metric_path: Dot-separated path to metric
            sensitivity: Number of standard deviations for anomaly threshold

        Returns:
            List of detected anomalies
        """
        with self._lock:
            if len(self._performance_history) < 10:
                return []

            # Extract metric values
            values = []
            timestamps = []

            for snapshot in self._performance_history:
                value = self._get_nested_value(snapshot, metric_path)
                if value is not None:
                    values.append(value)
                    timestamps.append(snapshot["timestamp"])

            if len(values) < 10:
                return []

            # Calculate statistics
            mean_value = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0

            if std_dev == 0:
                return []

            # Detect anomalies
            anomalies = []
            threshold = sensitivity * std_dev

            for i, (value, timestamp) in enumerate(zip(values, timestamps)):
                deviation = abs(value - mean_value)

                if deviation > threshold:
                    anomalies.append(
                        {
                            "timestamp": timestamp,
                            "value": value,
                            "expected_value": mean_value,
                            "deviation": deviation,
                            "severity": min(deviation / threshold, 5.0),  # Cap at 5x
                            "type": "high" if value > mean_value else "low",
                        }
                    )

            return anomalies

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        with self._lock:
            if not self._performance_history:
                return {"error": "No performance data available"}

            latest = self._performance_history[-1]

            report = {
                "timestamp": time.time(),
                "latest_metrics": latest,
                "trends": {},
                "anomalies": {},
                "recommendations": [],
            }

            # Analyze key metrics
            key_metrics = [
                "latency.avg",
                "latency.percentiles.p95",
                "throughput.requests_per_second",
                "throughput.error_rate",
                "resources.cpu.current",
                "resources.memory.current",
            ]

            for metric in key_metrics:
                # Trend analysis
                trend = self.detect_trends(metric)
                if trend["trend"] != "insufficient_data":
                    report["trends"][metric] = trend

                # Anomaly detection
                anomalies = self.detect_anomalies(metric)
                if anomalies:
                    report["anomalies"][metric] = anomalies[-5:]  # Last 5 anomalies

            return report

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Optional[float]:
        """Get nested value from dictionary using dot notation"""
        keys = path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        # Convert to float if possible
        try:
            return float(current)
        except (ValueError, TypeError):
            return None
