"""
Health monitoring and metrics collection for retro-peft components.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Optional, Any
import logging


@dataclass
class HealthStatus:
    """Represents the health status of a component"""
    is_healthy: bool
    status: str
    message: str
    timestamp: float
    metrics: Dict[str, Any]


class MetricsCollector:
    """
    Thread-safe metrics collector with time-windowed statistics.
    """
    
    def __init__(self, window_size: int = 1000, time_window: int = 300):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Maximum number of recent events to keep
            time_window: Time window in seconds for metrics
        """
        self.window_size = window_size
        self.time_window = time_window
        self._lock = Lock()
        
        # Metrics storage
        self._counters = defaultdict(int)
        self._timers = defaultdict(deque)
        self._gauges = {}
        self._events = deque()
        
        # Rate tracking
        self._rates = defaultdict(deque)
        
        self.logger = logging.getLogger(__name__)
    
    def increment(self, metric_name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self._lock:
            full_name = self._build_metric_name(metric_name, tags)
            self._counters[full_name] += value
            self._add_event("counter", metric_name, value, tags)
    
    def timer(self, metric_name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric"""
        with self._lock:
            full_name = self._build_metric_name(metric_name, tags)
            current_time = time.time()
            
            # Add timing data
            self._timers[full_name].append((current_time, duration))
            
            # Clean old data
            cutoff = current_time - self.time_window
            while self._timers[full_name] and self._timers[full_name][0][0] < cutoff:
                self._timers[full_name].popleft()
            
            # Limit size
            if len(self._timers[full_name]) > self.window_size:
                self._timers[full_name].popleft()
            
            self._add_event("timer", metric_name, duration, tags)
    
    def gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        with self._lock:
            full_name = self._build_metric_name(metric_name, tags)
            self._gauges[full_name] = (time.time(), value)
            self._add_event("gauge", metric_name, value, tags)
    
    def rate(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        """Record a rate event"""
        with self._lock:
            full_name = self._build_metric_name(metric_name, tags)
            current_time = time.time()
            
            self._rates[full_name].append(current_time)
            
            # Clean old data
            cutoff = current_time - self.time_window
            while self._rates[full_name] and self._rates[full_name][0] < cutoff:
                self._rates[full_name].popleft()
    
    def get_counter(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value"""
        with self._lock:
            full_name = self._build_metric_name(metric_name, tags)
            return self._counters.get(full_name, 0)
    
    def get_timer_stats(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timer statistics"""
        with self._lock:
            full_name = self._build_metric_name(metric_name, tags)
            timings = [duration for _, duration in self._timers.get(full_name, [])]
            
            if not timings:
                return {"count": 0, "avg": 0, "min": 0, "max": 0, "p95": 0}
            
            timings_sorted = sorted(timings)
            count = len(timings)
            
            return {
                "count": count,
                "avg": sum(timings) / count,
                "min": min(timings),
                "max": max(timings),
                "p50": timings_sorted[count // 2],
                "p95": timings_sorted[int(count * 0.95)] if count > 1 else timings[0],
                "p99": timings_sorted[int(count * 0.99)] if count > 1 else timings[0]
            }
    
    def get_gauge(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current gauge value"""
        with self._lock:
            full_name = self._build_metric_name(metric_name, tags)
            gauge_data = self._gauges.get(full_name)
            return gauge_data[1] if gauge_data else None
    
    def get_rate(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get current rate (events per second)"""
        with self._lock:
            full_name = self._build_metric_name(metric_name, tags)
            events = self._rates.get(full_name, deque())
            
            if not events:
                return 0.0
            
            # Calculate rate over time window
            return len(events) / self.time_window
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        with self._lock:
            metrics = {}
            
            # Counters
            for name, value in self._counters.items():
                metrics[f"counter.{name}"] = value
            
            # Timers
            for name in self._timers.keys():
                stats = self.get_timer_stats("", {})  # Will be recalculated
                for stat_name, stat_value in stats.items():
                    metrics[f"timer.{name}.{stat_name}"] = stat_value
            
            # Gauges
            for name, (timestamp, value) in self._gauges.items():
                metrics[f"gauge.{name}"] = value
                metrics[f"gauge.{name}.timestamp"] = timestamp
            
            # Rates
            for name in self._rates.keys():
                metrics[f"rate.{name}"] = self.get_rate("", {})
            
            return metrics
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self._lock:
            self._counters.clear()
            self._timers.clear()
            self._gauges.clear()
            self._rates.clear()
            self._events.clear()
    
    def _build_metric_name(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Build full metric name with tags"""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def _add_event(self, event_type: str, name: str, value: Any, tags: Optional[Dict[str, str]]):
        """Add event to event log"""
        event = {
            "type": event_type,
            "name": name,
            "value": value,
            "tags": tags or {},
            "timestamp": time.time()
        }
        
        self._events.append(event)
        
        # Limit event log size
        if len(self._events) > self.window_size:
            self._events.popleft()


class HealthMonitor:
    """
    Monitors health of retro-peft components.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics = metrics_collector or MetricsCollector()
        self.logger = logging.getLogger(__name__)
        
        # Health check functions
        self.health_checks = {}
        
        # Component status tracking
        self.component_status = {}
        self.last_health_check = 0
        
    def register_health_check(self, component: str, check_func: callable):
        """
        Register a health check function for a component.
        
        Args:
            component: Component name
            check_func: Function that returns HealthStatus
        """
        self.health_checks[component] = check_func
        self.logger.info(f"Registered health check for {component}")
    
    def check_health(self, component: Optional[str] = None) -> Dict[str, HealthStatus]:
        """
        Run health checks for components.
        
        Args:
            component: Specific component to check, or None for all
            
        Returns:
            Dictionary of component health statuses
        """
        current_time = time.time()
        results = {}
        
        # Determine which components to check
        if component:
            components = [component] if component in self.health_checks else []
        else:
            components = list(self.health_checks.keys())
        
        # Run health checks
        for comp_name in components:
            check_func = self.health_checks[comp_name]
            
            try:
                start_time = time.time()
                status = check_func()
                check_duration = time.time() - start_time
                
                # Record metrics
                self.metrics.timer("health_check_duration", check_duration, {"component": comp_name})
                
                if status.is_healthy:
                    self.metrics.increment("health_check_success", tags={"component": comp_name})
                else:
                    self.metrics.increment("health_check_failure", tags={"component": comp_name})
                    self.logger.warning(f"Health check failed for {comp_name}: {status.message}")
                
                results[comp_name] = status
                self.component_status[comp_name] = status
                
            except Exception as e:
                error_status = HealthStatus(
                    is_healthy=False,
                    status="ERROR",
                    message=f"Health check exception: {e}",
                    timestamp=current_time,
                    metrics={}
                )
                
                results[comp_name] = error_status
                self.component_status[comp_name] = error_status
                
                self.metrics.increment("health_check_error", tags={"component": comp_name})
                self.logger.error(f"Health check error for {comp_name}: {e}")
        
        self.last_health_check = current_time
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status"""
        current_time = time.time()
        
        if not self.component_status:
            return HealthStatus(
                is_healthy=False,
                status="UNKNOWN",
                message="No health checks registered",
                timestamp=current_time,
                metrics={}
            )
        
        # Check if any components are unhealthy
        unhealthy_components = [
            name for name, status in self.component_status.items()
            if not status.is_healthy
        ]
        
        if unhealthy_components:
            return HealthStatus(
                is_healthy=False,
                status="UNHEALTHY",
                message=f"Unhealthy components: {', '.join(unhealthy_components)}",
                timestamp=current_time,
                metrics={"unhealthy_count": len(unhealthy_components)}
            )
        
        return HealthStatus(
            is_healthy=True,
            status="HEALTHY",
            message="All components healthy",
            timestamp=current_time,
            metrics={"total_components": len(self.component_status)}
        )
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        metrics = self.metrics.get_all_metrics()
        
        # Add system-level metrics
        current_time = time.time()
        metrics.update({
            "system.uptime": current_time - (self.last_health_check or current_time),
            "system.healthy_components": len([
                s for s in self.component_status.values() if s.is_healthy
            ]),
            "system.total_components": len(self.component_status),
            "system.last_health_check": self.last_health_check,
        })
        
        return metrics


# Global health monitor instance
_global_health_monitor = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance"""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor


def monitor_performance(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """
    Decorator for monitoring function performance.
    
    Args:
        metric_name: Name of the metric to record
        tags: Optional tags for the metric
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            health_monitor = get_health_monitor()
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record successful execution
                health_monitor.metrics.timer(f"{metric_name}.duration", duration, tags)
                health_monitor.metrics.increment(f"{metric_name}.success", tags=tags)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failed execution
                health_monitor.metrics.timer(f"{metric_name}.duration", duration, tags)
                health_monitor.metrics.increment(f"{metric_name}.error", tags=tags)
                
                raise e
        
        return wrapper
    return decorator