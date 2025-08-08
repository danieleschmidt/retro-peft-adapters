"""
Load balancing and request routing for distributed inference.

Provides intelligent load distribution, circuit breakers,
and fault tolerance for production deployments.
"""

import asyncio
import heapq
import random
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..utils.logging import get_global_logger


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""

    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    CONSISTENT_HASH = "consistent_hash"


class BackendStatus(Enum):
    """Backend health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"


@dataclass
class Backend:
    """Backend server configuration"""

    id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    timeout: float = 30.0
    status: BackendStatus = BackendStatus.HEALTHY
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime statistics
    active_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: float = 0.0
    consecutive_failures: int = 0

    def __post_init__(self):
        self.endpoint = f"{self.host}:{self.port}"

    def get_load_score(self) -> float:
        """Calculate current load score (lower is better)"""
        if self.status != BackendStatus.HEALTHY:
            return float("inf")

        # Base load from active connections
        connection_load = self.active_connections / self.max_connections

        # Response time factor
        response_time_factor = min(self.avg_response_time / 1000.0, 2.0)  # Cap at 2s

        # Failure rate factor
        failure_rate = 0.0
        if self.total_requests > 0:
            failure_rate = self.failed_requests / self.total_requests

        # Combined score (0-1 scale, lower is better)
        load_score = connection_load * 0.4 + response_time_factor * 0.3 + failure_rate * 0.3

        # Apply weight (higher weight = lower effective load)
        return load_score / self.weight

    def update_stats(self, success: bool, response_time: float):
        """Update backend statistics"""
        self.total_requests += 1

        if success:
            self.successful_requests += 1
            self.consecutive_failures = 0
        else:
            self.failed_requests += 1
            self.consecutive_failures += 1

        # Update average response time (exponential moving average)
        alpha = 0.1
        self.avg_response_time = alpha * response_time + (1 - alpha) * self.avg_response_time


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""

    failure_threshold: int = 5
    timeout_duration: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self._lock = threading.RLock()
        self.logger = get_global_logger()

    def can_execute(self) -> bool:
        """Check if request can be executed"""
        with self._lock:
            now = time.time()

            if self.state == CircuitBreakerState.CLOSED:
                return True

            elif self.state == CircuitBreakerState.OPEN:
                # Check if timeout period has passed
                if now - self.last_failure_time >= self.config.timeout_duration:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    self.logger.info("Circuit breaker moving to HALF_OPEN state")
                    return True
                return False

            elif self.state == CircuitBreakerState.HALF_OPEN:
                return self.half_open_calls < self.config.half_open_max_calls

            return False

    def record_success(self):
        """Record successful execution"""
        with self._lock:
            self.failure_count = 0

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.success_count = 0
                    self.logger.info("Circuit breaker moving to CLOSED state")

    def record_failure(self):
        """Record failed execution"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    self.logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")

            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
                self.logger.warning("Circuit breaker back to OPEN state")

            self.half_open_calls += 1

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        with self._lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "half_open_calls": self.half_open_calls,
                "last_failure_time": self.last_failure_time,
            }


class RequestRouter:
    """
    Intelligent request router with multiple routing strategies.
    """

    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.logger = get_global_logger()

        # Strategy-specific state
        self._round_robin_index = 0
        self._consistent_hash_ring = {}
        self._lock = threading.RLock()

    def select_backend(
        self, backends: List[Backend], request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Backend]:
        """
        Select best backend for request.

        Args:
            backends: Available backends
            request_context: Optional request context for routing decisions

        Returns:
            Selected backend or None if none available
        """
        with self._lock:
            # Filter healthy backends
            healthy_backends = [
                b
                for b in backends
                if b.status == BackendStatus.HEALTHY and b.active_connections < b.max_connections
            ]

            if not healthy_backends:
                # Try degraded backends as fallback
                healthy_backends = [
                    b
                    for b in backends
                    if b.status == BackendStatus.DEGRADED
                    and b.active_connections < b.max_connections
                ]

            if not healthy_backends:
                return None

            # Apply routing strategy
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_select(healthy_backends)

            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_select(healthy_backends)

            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_select(healthy_backends)

            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time_select(healthy_backends)

            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                return self._resource_based_select(healthy_backends)

            elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                return self._consistent_hash_select(healthy_backends, request_context)

            else:
                # Fallback to round robin
                return self._round_robin_select(healthy_backends)

    def _round_robin_select(self, backends: List[Backend]) -> Backend:
        """Round-robin selection"""
        selected = backends[self._round_robin_index % len(backends)]
        self._round_robin_index += 1
        return selected

    def _weighted_round_robin_select(self, backends: List[Backend]) -> Backend:
        """Weighted round-robin selection"""
        # Build weighted list
        weighted_backends = []
        for backend in backends:
            count = max(1, int(backend.weight * 10))
            weighted_backends.extend([backend] * count)

        if not weighted_backends:
            return backends[0]

        selected = weighted_backends[self._round_robin_index % len(weighted_backends)]
        self._round_robin_index += 1
        return selected

    def _least_connections_select(self, backends: List[Backend]) -> Backend:
        """Select backend with least active connections"""
        return min(backends, key=lambda b: b.active_connections)

    def _least_response_time_select(self, backends: List[Backend]) -> Backend:
        """Select backend with lowest average response time"""
        return min(backends, key=lambda b: b.avg_response_time)

    def _resource_based_select(self, backends: List[Backend]) -> Backend:
        """Select backend based on comprehensive load score"""
        return min(backends, key=lambda b: b.get_load_score())

    def _consistent_hash_select(
        self, backends: List[Backend], request_context: Optional[Dict[str, Any]]
    ) -> Backend:
        """Consistent hash selection"""
        if not request_context or "hash_key" not in request_context:
            # Fallback to round robin if no hash key
            return self._round_robin_select(backends)

        hash_key = request_context["hash_key"]
        hash_value = hash(hash_key) % (2**32)

        # Simple consistent hashing
        backend_hashes = [(hash(b.id) % (2**32), b) for b in backends]
        backend_hashes.sort()

        for backend_hash, backend in backend_hashes:
            if hash_value <= backend_hash:
                return backend

        # Wrap around to first backend
        return backend_hashes[0][1]


class LoadBalancer:
    """
    Production-grade load balancer with health checking and circuit breakers.
    """

    def __init__(
        self,
        backends: List[Backend],
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS,
        health_check_interval: float = 30.0,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize load balancer.

        Args:
            backends: List of backend servers
            strategy: Load balancing strategy
            health_check_interval: Health check interval in seconds
            circuit_breaker_config: Circuit breaker configuration
        """
        self.backends = {b.id: b for b in backends}
        self.router = RequestRouter(strategy)
        self.health_check_interval = health_check_interval

        # Circuit breakers per backend
        cb_config = circuit_breaker_config or CircuitBreakerConfig()
        self.circuit_breakers = {b.id: CircuitBreaker(cb_config) for b in backends}

        # Health checking
        self._health_check_thread = None
        self._health_check_stop = threading.Event()

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "load_balancer_errors": 0,
        }

        self.logger = get_global_logger()
        self._lock = threading.RLock()

        # Start health checking
        self.start_health_checking()

    def route_request(
        self, request_context: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None
    ) -> Optional[Tuple[Backend, Any]]:
        """
        Route request to best available backend.

        Args:
            request_context: Optional request context
            timeout: Request timeout

        Returns:
            Tuple of (backend, connection) or None if no backend available
        """
        with self._lock:
            self.stats["total_requests"] += 1

            # Get available backends
            available_backends = []
            for backend in self.backends.values():
                circuit_breaker = self.circuit_breakers[backend.id]

                if circuit_breaker.can_execute():
                    available_backends.append(backend)

            if not available_backends:
                self.stats["load_balancer_errors"] += 1
                self.logger.warning("No available backends for request routing")
                return None

            # Select backend
            selected_backend = self.router.select_backend(available_backends, request_context)

            if not selected_backend:
                self.stats["load_balancer_errors"] += 1
                return None

            # Update connection count
            selected_backend.active_connections += 1

            # Create connection context (mock for now)
            connection = BackendConnection(self, selected_backend, timeout)

            return selected_backend, connection

    def record_request_result(
        self,
        backend: Backend,
        success: bool,
        response_time: float,
        error: Optional[Exception] = None,
    ):
        """
        Record request result for statistics and circuit breaker.

        Args:
            backend: Backend that handled the request
            success: Whether request was successful
            response_time: Response time in milliseconds
            error: Optional error if request failed
        """
        with self._lock:
            # Update backend stats
            backend.update_stats(success, response_time)
            backend.active_connections = max(0, backend.active_connections - 1)

            # Update circuit breaker
            circuit_breaker = self.circuit_breakers[backend.id]
            if success:
                circuit_breaker.record_success()
                self.stats["successful_requests"] += 1
            else:
                circuit_breaker.record_failure()
                self.stats["failed_requests"] += 1

                if error:
                    self.logger.warning(f"Request to backend {backend.id} failed: {error}")

    def add_backend(self, backend: Backend):
        """Add new backend"""
        with self._lock:
            self.backends[backend.id] = backend
            self.circuit_breakers[backend.id] = CircuitBreaker(CircuitBreakerConfig())
            self.logger.info(f"Added backend: {backend.id}")

    def remove_backend(self, backend_id: str):
        """Remove backend"""
        with self._lock:
            if backend_id in self.backends:
                del self.backends[backend_id]
                del self.circuit_breakers[backend_id]
                self.logger.info(f"Removed backend: {backend_id}")

    def get_backend_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all backends"""
        with self._lock:
            status = {}

            for backend_id, backend in self.backends.items():
                circuit_breaker = self.circuit_breakers[backend_id]

                status[backend_id] = {
                    "endpoint": backend.endpoint,
                    "status": backend.status.value,
                    "active_connections": backend.active_connections,
                    "max_connections": backend.max_connections,
                    "total_requests": backend.total_requests,
                    "successful_requests": backend.successful_requests,
                    "failed_requests": backend.failed_requests,
                    "success_rate": (
                        backend.successful_requests / backend.total_requests
                        if backend.total_requests > 0
                        else 0.0
                    ),
                    "avg_response_time": backend.avg_response_time,
                    "load_score": backend.get_load_score(),
                    "circuit_breaker": circuit_breaker.get_state(),
                    "last_health_check": backend.last_health_check,
                }

            return status

    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        with self._lock:
            backend_stats = self.get_backend_status()

            return {
                "load_balancer": self.stats.copy(),
                "backends": backend_stats,
                "strategy": self.router.strategy.value,
                "total_backends": len(self.backends),
                "healthy_backends": sum(
                    1 for b in self.backends.values() if b.status == BackendStatus.HEALTHY
                ),
            }

    def start_health_checking(self):
        """Start background health checking"""
        if self._health_check_thread is None or not self._health_check_thread.is_alive():
            self._health_check_stop.clear()
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop, daemon=True
            )
            self._health_check_thread.start()
            self.logger.info("Started health checking")

    def stop_health_checking(self):
        """Stop background health checking"""
        self._health_check_stop.set()
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5.0)
        self.logger.info("Stopped health checking")

    def _health_check_loop(self):
        """Background health checking loop"""
        while not self._health_check_stop.wait(self.health_check_interval):
            try:
                self._perform_health_checks()
            except Exception as e:
                self.logger.error(f"Health check error: {e}")

    def _perform_health_checks(self):
        """Perform health checks on all backends"""
        for backend in self.backends.values():
            try:
                # Simple health check (would be implemented based on actual backend)
                health_check_result = self._check_backend_health(backend)

                backend.last_health_check = time.time()

                if health_check_result:
                    if backend.status == BackendStatus.UNHEALTHY:
                        backend.status = BackendStatus.HEALTHY
                        self.logger.info(f"Backend {backend.id} recovered")
                else:
                    if backend.status == BackendStatus.HEALTHY:
                        backend.status = BackendStatus.DEGRADED
                        self.logger.warning(f"Backend {backend.id} degraded")
                    elif backend.status == BackendStatus.DEGRADED:
                        backend.status = BackendStatus.UNHEALTHY
                        self.logger.error(f"Backend {backend.id} unhealthy")

            except Exception as e:
                self.logger.error(f"Health check failed for backend {backend.id}: {e}")
                backend.status = BackendStatus.UNHEALTHY

    def _check_backend_health(self, backend: Backend) -> bool:
        """
        Check health of individual backend.

        Args:
            backend: Backend to check

        Returns:
            True if healthy, False otherwise
        """
        # Mock health check - would implement actual HTTP/gRPC health check
        try:
            # Simulate health check
            import random

            return random.random() > 0.1  # 90% healthy
        except Exception:
            return False


class BackendConnection:
    """
    Context manager for backend connections.
    """

    def __init__(
        self, load_balancer: LoadBalancer, backend: Backend, timeout: Optional[float] = None
    ):
        self.load_balancer = load_balancer
        self.backend = backend
        self.timeout = timeout or backend.timeout
        self.start_time = time.time()
        self.connection = None  # Would be actual connection object

    def __enter__(self):
        """Enter connection context"""
        # Would establish actual connection here
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit connection context"""
        # Calculate response time
        response_time = (time.time() - self.start_time) * 1000  # ms

        # Record result
        success = exc_type is None
        error = exc_val if exc_val else None

        self.load_balancer.record_request_result(self.backend, success, response_time, error)

        # Would close actual connection here
        return False  # Don't suppress exceptions

    async def __aenter__(self):
        """Async enter"""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit"""
        return self.__exit__(exc_type, exc_val, exc_tb)
