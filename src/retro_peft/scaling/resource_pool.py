"""
Resource pooling and management for efficient resource utilization.

Provides connection pools, model instance pools, and resource lifecycle
management for high-performance deployments.
"""

import asyncio
import os
import threading
import time
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

import psutil

from ..utils.logging import get_global_logger

T = TypeVar("T")


@dataclass
class PoolConfig:
    """Resource pool configuration"""

    min_size: int = 1
    max_size: int = 10
    idle_timeout: float = 300.0  # 5 minutes
    max_lifetime: float = 3600.0  # 1 hour
    validation_interval: float = 60.0  # 1 minute
    creation_timeout: float = 30.0
    acquire_timeout: float = 10.0


@dataclass
class PooledResource(Generic[T]):
    """Wrapper for pooled resources"""

    resource: T
    created_at: float
    last_used: float
    use_count: int = 0
    is_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.last_used == 0:
            self.last_used = self.created_at

    def is_expired(self, max_lifetime: float, idle_timeout: float) -> bool:
        """Check if resource is expired"""
        now = time.time()

        # Check maximum lifetime
        if now - self.created_at > max_lifetime:
            return True

        # Check idle timeout
        if now - self.last_used > idle_timeout:
            return True

        return False

    def mark_used(self):
        """Mark resource as used"""
        self.last_used = time.time()
        self.use_count += 1


class ResourcePool(Generic[T]):
    """
    Generic resource pool with lifecycle management.
    """

    def __init__(
        self,
        name: str,
        factory: Callable[[], T],
        validator: Optional[Callable[[T], bool]] = None,
        destroyer: Optional[Callable[[T], None]] = None,
        config: Optional[PoolConfig] = None,
    ):
        """
        Initialize resource pool.

        Args:
            name: Pool name for logging
            factory: Function to create new resources
            validator: Optional function to validate resources
            destroyer: Optional function to cleanup resources
            config: Pool configuration
        """
        self.name = name
        self.factory = factory
        self.validator = validator or (lambda x: True)
        self.destroyer = destroyer or (lambda x: None)
        self.config = config or PoolConfig()

        # Pool state
        self._available: deque[PooledResource[T]] = deque()
        self._in_use: Dict[int, PooledResource[T]] = {}
        self._total_created = 0
        self._total_destroyed = 0

        # Synchronization
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

        # Background maintenance
        self._maintenance_thread = None
        self._maintenance_stop = threading.Event()

        self.logger = get_global_logger()

        # Start maintenance
        self._start_maintenance()

        # Create minimum resources
        self._ensure_min_resources()

    def acquire(self, timeout: Optional[float] = None) -> Optional[T]:
        """
        Acquire resource from pool.

        Args:
            timeout: Timeout in seconds

        Returns:
            Resource or None if timeout
        """
        timeout = timeout or self.config.acquire_timeout
        deadline = time.time() + timeout

        with self._condition:
            while True:
                # Try to get available resource
                resource = self._get_available_resource()
                if resource:
                    return resource

                # Check if we can create new resource
                total_resources = len(self._available) + len(self._in_use)
                if total_resources < self.config.max_size:
                    try:
                        new_resource = self._create_resource()
                        if new_resource:
                            return new_resource
                    except Exception as e:
                        self.logger.error(f"Failed to create resource in pool {self.name}: {e}")

                # Wait for resource to become available
                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    self.logger.warning(f"Timeout acquiring resource from pool {self.name}")
                    return None

                self._condition.wait(timeout=remaining_time)

    def release(self, resource: T):
        """
        Release resource back to pool.

        Args:
            resource: Resource to release
        """
        with self._lock:
            resource_id = id(resource)

            if resource_id not in self._in_use:
                self.logger.warning(f"Releasing unknown resource from pool {self.name}")
                return

            pooled_resource = self._in_use.pop(resource_id)

            # Validate resource before returning to pool
            try:
                if self.validator(resource) and pooled_resource.is_valid:
                    pooled_resource.mark_used()
                    self._available.append(pooled_resource)
                    self._condition.notify()
                else:
                    # Resource is invalid, destroy it
                    self._destroy_resource(pooled_resource)

            except Exception as e:
                self.logger.error(f"Error validating resource in pool {self.name}: {e}")
                self._destroy_resource(pooled_resource)

    def _get_available_resource(self) -> Optional[T]:
        """Get available resource from pool"""
        while self._available:
            pooled_resource = self._available.popleft()

            # Check if resource is expired
            if pooled_resource.is_expired(self.config.max_lifetime, self.config.idle_timeout):
                self._destroy_resource(pooled_resource)
                continue

            # Validate resource
            try:
                if not self.validator(pooled_resource.resource):
                    self._destroy_resource(pooled_resource)
                    continue

                # Resource is good, mark as in use
                pooled_resource.mark_used()
                self._in_use[id(pooled_resource.resource)] = pooled_resource
                return pooled_resource.resource

            except Exception as e:
                self.logger.error(f"Error validating resource in pool {self.name}: {e}")
                self._destroy_resource(pooled_resource)
                continue

        return None

    def _create_resource(self) -> Optional[T]:
        """Create new resource"""
        try:
            start_time = time.time()
            resource = self.factory()
            creation_time = time.time() - start_time

            if creation_time > self.config.creation_timeout:
                self.logger.warning(
                    f"Resource creation took {creation_time:.2f}s in pool {self.name}"
                )

            pooled_resource = PooledResource(resource=resource, created_at=time.time(), last_used=0)

            pooled_resource.mark_used()
            self._in_use[id(resource)] = pooled_resource
            self._total_created += 1

            self.logger.debug(f"Created new resource in pool {self.name}")
            return resource

        except Exception as e:
            self.logger.error(f"Failed to create resource in pool {self.name}: {e}")
            return None

    def _destroy_resource(self, pooled_resource: PooledResource[T]):
        """Destroy resource"""
        try:
            self.destroyer(pooled_resource.resource)
            self._total_destroyed += 1
            self.logger.debug(f"Destroyed resource in pool {self.name}")
        except Exception as e:
            self.logger.error(f"Error destroying resource in pool {self.name}: {e}")

    def _ensure_min_resources(self):
        """Ensure minimum number of resources"""
        with self._lock:
            total_resources = len(self._available) + len(self._in_use)
            needed = self.config.min_size - total_resources

            for _ in range(needed):
                try:
                    resource = self.factory()
                    pooled_resource = PooledResource(
                        resource=resource, created_at=time.time(), last_used=time.time()
                    )
                    self._available.append(pooled_resource)
                    self._total_created += 1
                except Exception as e:
                    self.logger.error(f"Failed to create minimum resource in pool {self.name}: {e}")
                    break

    def _start_maintenance(self):
        """Start background maintenance thread"""
        if self._maintenance_thread is None or not self._maintenance_thread.is_alive():
            self._maintenance_stop.clear()
            self._maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
            self._maintenance_thread.start()

    def _maintenance_loop(self):
        """Background maintenance loop"""
        while not self._maintenance_stop.wait(self.config.validation_interval):
            try:
                self._cleanup_expired_resources()
                self._ensure_min_resources()
            except Exception as e:
                self.logger.error(f"Maintenance error in pool {self.name}: {e}")

    def _cleanup_expired_resources(self):
        """Clean up expired resources"""
        with self._lock:
            # Clean up available resources
            expired_resources = []
            remaining_resources = deque()

            while self._available:
                pooled_resource = self._available.popleft()

                if pooled_resource.is_expired(self.config.max_lifetime, self.config.idle_timeout):
                    expired_resources.append(pooled_resource)
                else:
                    remaining_resources.append(pooled_resource)

            self._available = remaining_resources

            # Destroy expired resources
            for pooled_resource in expired_resources:
                self._destroy_resource(pooled_resource)

            if expired_resources:
                self.logger.debug(
                    f"Cleaned up {len(expired_resources)} expired resources in pool {self.name}"
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                "name": self.name,
                "available": len(self._available),
                "in_use": len(self._in_use),
                "total_created": self._total_created,
                "total_destroyed": self._total_destroyed,
                "min_size": self.config.min_size,
                "max_size": self.config.max_size,
                "utilization": (
                    len(self._in_use) / self.config.max_size if self.config.max_size > 0 else 0
                ),
            }

    def close(self):
        """Close pool and cleanup resources"""
        # Stop maintenance
        self._maintenance_stop.set()
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            self._maintenance_thread.join(timeout=5.0)

        with self._lock:
            # Destroy all resources
            while self._available:
                pooled_resource = self._available.popleft()
                self._destroy_resource(pooled_resource)

            # Note: In-use resources will be cleaned up when released
            self.logger.info(f"Closed resource pool {self.name}")

    @contextmanager
    def get_resource(self, timeout: Optional[float] = None):
        """Context manager for resource acquisition"""
        resource = self.acquire(timeout)
        if resource is None:
            raise TimeoutError(f"Failed to acquire resource from pool {self.name}")

        try:
            yield resource
        finally:
            self.release(resource)


class ModelPool(ResourcePool):
    """
    Specialized pool for ML model instances.
    """

    def __init__(
        self,
        model_name: str,
        model_factory: Callable[[], Any],
        device: str = "cpu",
        config: Optional[PoolConfig] = None,
    ):
        """
        Initialize model pool.

        Args:
            model_name: Name of the model
            model_factory: Function to create model instances
            device: Device to load models on
            config: Pool configuration
        """
        self.model_name = model_name
        self.device = device

        # Model-specific validator
        def validate_model(model):
            try:
                # Basic validation - could be enhanced
                return hasattr(model, "forward") or hasattr(model, "generate")
            except Exception:
                return False

        # Model-specific destroyer
        def destroy_model(model):
            try:
                # Move to CPU and clear cache if CUDA model
                if hasattr(model, "cpu"):
                    model.cpu()

                # Clear CUDA cache
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                # Delete model
                del model

            except Exception as e:
                self.logger.error(f"Error destroying model: {e}")

        super().__init__(
            name=f"ModelPool-{model_name}",
            factory=model_factory,
            validator=validate_model,
            destroyer=destroy_model,
            config=config,
        )


class ConnectionPool(ResourcePool):
    """
    Generic connection pool for database/API connections.
    """

    def __init__(
        self, name: str, connection_factory: Callable[[], Any], config: Optional[PoolConfig] = None
    ):
        """
        Initialize connection pool.

        Args:
            name: Pool name
            connection_factory: Function to create connections
            config: Pool configuration
        """

        # Connection-specific validator
        def validate_connection(conn):
            try:
                # Basic validation - could be enhanced per connection type
                return hasattr(conn, "close") and not getattr(conn, "closed", False)
            except Exception:
                return False

        # Connection-specific destroyer
        def destroy_connection(conn):
            try:
                if hasattr(conn, "close"):
                    conn.close()
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")

        super().__init__(
            name=f"ConnectionPool-{name}",
            factory=connection_factory,
            validator=validate_connection,
            destroyer=destroy_connection,
            config=config,
        )


class AsyncResourcePool(Generic[T]):
    """
    Async version of resource pool.
    """

    def __init__(
        self,
        name: str,
        factory: Callable[[], T],
        validator: Optional[Callable[[T], bool]] = None,
        destroyer: Optional[Callable[[T], None]] = None,
        config: Optional[PoolConfig] = None,
    ):
        self.name = name
        self.factory = factory
        self.validator = validator or (lambda x: True)
        self.destroyer = destroyer or (lambda x: None)
        self.config = config or PoolConfig()

        # Pool state
        self._available: deque[PooledResource[T]] = deque()
        self._in_use: Dict[int, PooledResource[T]] = {}
        self._total_created = 0
        self._total_destroyed = 0

        # Async synchronization
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

        # Background tasks
        self._maintenance_task = None
        self._running = False

        self.logger = get_global_logger()

    async def start(self):
        """Start the async pool"""
        if self._running:
            return

        self._running = True
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

        # Create minimum resources
        await self._ensure_min_resources()

    async def stop(self):
        """Stop the async pool"""
        if not self._running:
            return

        self._running = False

        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass

        await self._close_all_resources()

    async def acquire(self, timeout: Optional[float] = None) -> Optional[T]:
        """Acquire resource asynchronously"""
        timeout = timeout or self.config.acquire_timeout

        async with self._condition:
            deadline = time.time() + timeout

            while True:
                # Try to get available resource
                resource = await self._get_available_resource()
                if resource:
                    return resource

                # Check if we can create new resource
                total_resources = len(self._available) + len(self._in_use)
                if total_resources < self.config.max_size:
                    try:
                        new_resource = await self._create_resource()
                        if new_resource:
                            return new_resource
                    except Exception as e:
                        self.logger.error(f"Failed to create resource in pool {self.name}: {e}")

                # Wait for resource to become available
                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    return None

                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining_time)
                except asyncio.TimeoutError:
                    return None

    async def release(self, resource: T):
        """Release resource back to pool"""
        async with self._lock:
            resource_id = id(resource)

            if resource_id not in self._in_use:
                return

            pooled_resource = self._in_use.pop(resource_id)

            try:
                if self.validator(resource) and pooled_resource.is_valid:
                    pooled_resource.mark_used()
                    self._available.append(pooled_resource)

                    # Notify waiting tasks
                    async with self._condition:
                        self._condition.notify()
                else:
                    await self._destroy_resource(pooled_resource)

            except Exception as e:
                self.logger.error(f"Error validating resource in pool {self.name}: {e}")
                await self._destroy_resource(pooled_resource)

    async def _get_available_resource(self) -> Optional[T]:
        """Get available resource from pool"""
        while self._available:
            pooled_resource = self._available.popleft()

            if pooled_resource.is_expired(self.config.max_lifetime, self.config.idle_timeout):
                await self._destroy_resource(pooled_resource)
                continue

            try:
                if not self.validator(pooled_resource.resource):
                    await self._destroy_resource(pooled_resource)
                    continue

                pooled_resource.mark_used()
                self._in_use[id(pooled_resource.resource)] = pooled_resource
                return pooled_resource.resource

            except Exception as e:
                self.logger.error(f"Error validating resource in pool {self.name}: {e}")
                await self._destroy_resource(pooled_resource)
                continue

        return None

    async def _create_resource(self) -> Optional[T]:
        """Create new resource"""
        try:
            # Run factory in thread pool if it's blocking
            if asyncio.iscoroutinefunction(self.factory):
                resource = await self.factory()
            else:
                loop = asyncio.get_event_loop()
                resource = await loop.run_in_executor(None, self.factory)

            pooled_resource = PooledResource(resource=resource, created_at=time.time(), last_used=0)

            pooled_resource.mark_used()
            self._in_use[id(resource)] = pooled_resource
            self._total_created += 1

            return resource

        except Exception as e:
            self.logger.error(f"Failed to create resource in pool {self.name}: {e}")
            return None

    async def _destroy_resource(self, pooled_resource: PooledResource[T]):
        """Destroy resource"""
        try:
            if asyncio.iscoroutinefunction(self.destroyer):
                await self.destroyer(pooled_resource.resource)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.destroyer, pooled_resource.resource)

            self._total_destroyed += 1
        except Exception as e:
            self.logger.error(f"Error destroying resource in pool {self.name}: {e}")

    async def _ensure_min_resources(self):
        """Ensure minimum number of resources"""
        async with self._lock:
            total_resources = len(self._available) + len(self._in_use)
            needed = self.config.min_size - total_resources

            for _ in range(needed):
                try:
                    if asyncio.iscoroutinefunction(self.factory):
                        resource = await self.factory()
                    else:
                        loop = asyncio.get_event_loop()
                        resource = await loop.run_in_executor(None, self.factory)

                    pooled_resource = PooledResource(
                        resource=resource, created_at=time.time(), last_used=time.time()
                    )
                    self._available.append(pooled_resource)
                    self._total_created += 1
                except Exception as e:
                    self.logger.error(f"Failed to create minimum resource in pool {self.name}: {e}")
                    break

    async def _maintenance_loop(self):
        """Background maintenance loop"""
        while self._running:
            try:
                await asyncio.sleep(self.config.validation_interval)
                await self._cleanup_expired_resources()
                await self._ensure_min_resources()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Maintenance error in pool {self.name}: {e}")

    async def _cleanup_expired_resources(self):
        """Clean up expired resources"""
        async with self._lock:
            expired_resources = []
            remaining_resources = deque()

            while self._available:
                pooled_resource = self._available.popleft()

                if pooled_resource.is_expired(self.config.max_lifetime, self.config.idle_timeout):
                    expired_resources.append(pooled_resource)
                else:
                    remaining_resources.append(pooled_resource)

            self._available = remaining_resources

            for pooled_resource in expired_resources:
                await self._destroy_resource(pooled_resource)

    async def _close_all_resources(self):
        """Close all resources"""
        async with self._lock:
            while self._available:
                pooled_resource = self._available.popleft()
                await self._destroy_resource(pooled_resource)

    @asynccontextmanager
    async def get_resource(self, timeout: Optional[float] = None):
        """Async context manager for resource acquisition"""
        resource = await self.acquire(timeout)
        if resource is None:
            raise TimeoutError(f"Failed to acquire resource from pool {self.name}")

        try:
            yield resource
        finally:
            await self.release(resource)


class ResourceManager:
    """
    Central resource manager for all pools.
    """

    def __init__(self):
        self.pools: Dict[str, ResourcePool] = {}
        self.async_pools: Dict[str, AsyncResourcePool] = {}
        self.logger = get_global_logger()
        self._lock = threading.RLock()

    def create_model_pool(
        self,
        model_name: str,
        model_factory: Callable[[], Any],
        device: str = "cpu",
        config: Optional[PoolConfig] = None,
    ) -> ModelPool:
        """Create and register model pool"""
        with self._lock:
            pool = ModelPool(model_name, model_factory, device, config)
            self.pools[f"model_{model_name}"] = pool
            return pool

    def create_connection_pool(
        self, name: str, connection_factory: Callable[[], Any], config: Optional[PoolConfig] = None
    ) -> ConnectionPool:
        """Create and register connection pool"""
        with self._lock:
            pool = ConnectionPool(name, connection_factory, config)
            self.pools[f"connection_{name}"] = pool
            return pool

    def get_pool(self, name: str) -> Optional[ResourcePool]:
        """Get pool by name"""
        with self._lock:
            return self.pools.get(name)

    def get_model(self, model_name: str, timeout: Optional[float] = None):
        """Get model from pool (context manager)"""
        pool_name = f"model_{model_name}"
        pool = self.pools.get(pool_name)

        if pool is None:
            raise ValueError(f"No model pool found for {model_name}")

        return pool.get_resource(timeout)

    def get_connection(self, name: str, timeout: Optional[float] = None):
        """Get connection from pool (context manager)"""
        pool_name = f"connection_{name}"
        pool = self.pools.get(pool_name)

        if pool is None:
            raise ValueError(f"No connection pool found for {name}")

        return pool.get_resource(timeout)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools"""
        with self._lock:
            stats = {}
            for name, pool in self.pools.items():
                stats[name] = pool.get_stats()
            return stats

    def close_all_pools(self):
        """Close all pools"""
        with self._lock:
            for pool in self.pools.values():
                try:
                    pool.close()
                except Exception as e:
                    self.logger.error(f"Error closing pool: {e}")

            self.pools.clear()


# Global resource manager
_global_resource_manager = None


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance"""
    global _global_resource_manager

    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()

    return _global_resource_manager
