"""
Generation 3: Scaling Features Implementation

This module implements comprehensive performance optimization, caching,
auto-scaling, load balancing, and resource pooling for production scale.
"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import weakref

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class HighPerformanceCache:
    """Advanced caching system with TTL, LRU, and memory management"""
    
    def __init__(
        self, 
        max_size: int = 10000,
        default_ttl: float = 3600.0,
        cleanup_interval: float = 300.0,
        max_memory_mb: float = 500.0
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        self._cache = {}
        self._access_times = {}
        self._expiry_times = {}
        self._memory_usage = 0
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_evictions': 0
        }
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with access time tracking"""
        with self._lock:
            current_time = time.time()
            
            # Check if key exists and not expired
            if key in self._cache:
                if current_time <= self._expiry_times.get(key, float('inf')):
                    self._access_times[key] = current_time
                    self._stats['hits'] += 1
                    return self._cache[key]
                else:
                    # Expired - remove
                    self._remove_key(key)
            
            self._stats['misses'] += 1
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache with optional TTL"""
        with self._lock:
            current_time = time.time()
            ttl = ttl or self.default_ttl
            
            # Calculate memory usage estimate
            value_size = self._estimate_size(value)
            
            # Remove old value if exists
            if key in self._cache:
                self._remove_key(key)
            
            # Check memory constraints
            if self._memory_usage + value_size > self.max_memory_bytes:
                if not self._make_room(value_size):
                    return False  # Couldn't make room
            
            # Check size constraints
            if len(self._cache) >= self.max_size:
                if not self._evict_lru():
                    return False
            
            # Store value
            self._cache[key] = value
            self._access_times[key] = current_time
            self._expiry_times[key] = current_time + ttl
            self._memory_usage += value_size
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                self._remove_key(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._expiry_times.clear()
            self._memory_usage = 0
    
    def _remove_key(self, key: str):
        """Remove key and update memory usage"""
        if key in self._cache:
            value_size = self._estimate_size(self._cache[key])
            del self._cache[key]
            self._access_times.pop(key, None)
            self._expiry_times.pop(key, None)
            self._memory_usage -= value_size
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item"""
        if not self._access_times:
            return False
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._remove_key(lru_key)
        self._stats['evictions'] += 1
        return True
    
    def _make_room(self, needed_size: int) -> bool:
        """Make room for new item by evicting old ones"""
        evicted = 0
        while (self._memory_usage + needed_size > self.max_memory_bytes and 
               self._cache and evicted < self.max_size // 2):
            if not self._evict_lru():
                break
            evicted += 1
            self._stats['memory_evictions'] += 1
        
        return self._memory_usage + needed_size <= self.max_memory_bytes
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object"""
        try:
            import sys
            return sys.getsizeof(obj, 0)
        except:
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj) * 2  # Unicode characters
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj) + 64
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items()) + 64
            else:
                return 64  # Default estimate
    
    def _cleanup_loop(self):
        """Background cleanup of expired entries"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
            except Exception:
                pass  # Continue running
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, expiry in self._expiry_times.items()
                if current_time > expiry
            ]
            
            for key in expired_keys:
                self._remove_key(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate': hit_rate,
                **self._stats
            }


class AdaptiveResourcePool:
    """Dynamic resource pool with auto-scaling based on load"""
    
    def __init__(
        self,
        resource_factory: Callable,
        min_size: int = 2,
        max_size: int = 20,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        check_interval: float = 30.0
    ):
        self.resource_factory = resource_factory
        self.min_size = min_size
        self.max_size = max_size
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.check_interval = check_interval
        
        self._pool = deque()
        self._in_use = set()
        self._lock = threading.RLock()
        self._metrics = {
            'total_requests': 0,
            'peak_usage': 0,
            'scale_up_events': 0,
            'scale_down_events': 0
        }
        
        # Initialize minimum resources
        for _ in range(min_size):
            self._pool.append(self.resource_factory())
        
        # Start scaling thread
        self._scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._scaling_thread.start()
    
    @contextmanager
    def get_resource(self):
        """Get resource from pool (context manager)"""
        resource = self._acquire_resource()
        try:
            yield resource
        finally:
            self._release_resource(resource)
    
    def _acquire_resource(self):
        """Acquire resource from pool"""
        with self._lock:
            self._metrics['total_requests'] += 1
            
            # Try to get from pool
            if self._pool:
                resource = self._pool.popleft()
            else:
                # Create new resource if under max
                if len(self._in_use) < self.max_size:
                    resource = self.resource_factory()
                else:
                    # Wait for resource to become available
                    resource = self._wait_for_resource()
            
            self._in_use.add(resource)
            self._metrics['peak_usage'] = max(self._metrics['peak_usage'], len(self._in_use))
            
            return resource
    
    def _release_resource(self, resource):
        """Release resource back to pool"""
        with self._lock:
            if resource in self._in_use:
                self._in_use.remove(resource)
                self._pool.append(resource)
    
    def _wait_for_resource(self):
        """Wait for resource to become available"""
        # Simple blocking wait - in production, might use condition variables
        while True:
            time.sleep(0.1)
            with self._lock:
                if self._pool:
                    return self._pool.popleft()
    
    def _scaling_loop(self):
        """Background scaling based on usage metrics"""
        while True:
            try:
                time.sleep(self.check_interval)
                self._check_scaling()
            except Exception:
                pass  # Continue running
    
    def _check_scaling(self):
        """Check if scaling is needed"""
        with self._lock:
            total_resources = len(self._pool) + len(self._in_use)
            if total_resources == 0:
                return
            
            utilization = len(self._in_use) / total_resources
            
            # Scale up if high utilization
            if (utilization > self.scale_up_threshold and 
                total_resources < self.max_size):
                new_count = min(2, self.max_size - total_resources)
                for _ in range(new_count):
                    self._pool.append(self.resource_factory())
                self._metrics['scale_up_events'] += 1
            
            # Scale down if low utilization
            elif (utilization < self.scale_down_threshold and 
                  total_resources > self.min_size):
                remove_count = min(
                    len(self._pool),
                    total_resources - self.min_size,
                    total_resources // 4  # Don't remove more than 25% at once
                )
                for _ in range(remove_count):
                    if self._pool:
                        self._pool.pop()
                self._metrics['scale_down_events'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'in_use': len(self._in_use),
                'total_resources': len(self._pool) + len(self._in_use),
                'utilization': len(self._in_use) / max(1, len(self._pool) + len(self._in_use)),
                **self._metrics
            }


class LoadBalancer:
    """Intelligent load balancer with health checking and failover"""
    
    def __init__(self, health_check_interval: float = 30.0):
        self.health_check_interval = health_check_interval
        self._backends = []
        self._backend_stats = defaultdict(lambda: {
            'requests': 0,
            'failures': 0,
            'response_times': deque(maxlen=100),
            'healthy': True,
            'last_check': 0
        })
        self._lock = threading.RLock()
        self._current_backend = 0
        
        # Start health check thread
        self._health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_thread.start()
    
    def add_backend(self, backend: Any, weight: int = 1):
        """Add backend with optional weight"""
        with self._lock:
            self._backends.append({'backend': backend, 'weight': weight})
    
    def remove_backend(self, backend: Any):
        """Remove backend"""
        with self._lock:
            self._backends = [b for b in self._backends if b['backend'] != backend]
    
    def get_backend(self) -> Any:
        """Get best available backend using weighted round-robin"""
        with self._lock:
            healthy_backends = [
                b for b in self._backends
                if self._backend_stats[id(b['backend'])]['healthy']
            ]
            
            if not healthy_backends:
                # Fallback to any backend if none are healthy
                healthy_backends = self._backends
            
            if not healthy_backends:
                raise RuntimeError("No backends available")
            
            # Weighted round-robin selection
            total_weight = sum(b['weight'] for b in healthy_backends)
            if total_weight == 0:
                return healthy_backends[0]['backend']
            
            # Simple round-robin for now (can be enhanced with true weighted selection)
            backend_info = healthy_backends[self._current_backend % len(healthy_backends)]
            self._current_backend += 1
            
            return backend_info['backend']
    
    def record_request(self, backend: Any, duration: float, success: bool):
        """Record request metrics for backend"""
        with self._lock:
            backend_id = id(backend)
            stats = self._backend_stats[backend_id]
            stats['requests'] += 1
            stats['response_times'].append(duration)
            
            if not success:
                stats['failures'] += 1
                
                # Mark as unhealthy if failure rate too high
                if stats['requests'] > 10:
                    failure_rate = stats['failures'] / stats['requests']
                    if failure_rate > 0.5:  # 50% failure rate
                        stats['healthy'] = False
    
    def _health_check_loop(self):
        """Background health checking"""
        while True:
            try:
                time.sleep(self.health_check_interval)
                self._check_backend_health()
            except Exception:
                pass
    
    def _check_backend_health(self):
        """Check health of all backends"""
        with self._lock:
            current_time = time.time()
            
            for backend_info in self._backends:
                backend = backend_info['backend']
                backend_id = id(backend)
                stats = self._backend_stats[backend_id]
                
                # Simple health check - reset failure count periodically
                if current_time - stats['last_check'] > self.health_check_interval:
                    stats['last_check'] = current_time
                    
                    # Reset failure rate gradually
                    if stats['requests'] > 0:
                        stats['failures'] = max(0, stats['failures'] - 1)
                        
                        # Mark as healthy if failure rate is low
                        failure_rate = stats['failures'] / stats['requests']
                        if failure_rate < 0.2:  # 20% failure rate
                            stats['healthy'] = True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        with self._lock:
            backend_stats = {}
            
            for backend_info in self._backends:
                backend_id = id(backend_info['backend'])
                stats = self._backend_stats[backend_id]
                
                response_times = list(stats['response_times'])
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                
                backend_stats[f"backend_{backend_id}"] = {
                    'requests': stats['requests'],
                    'failures': stats['failures'],
                    'failure_rate': stats['failures'] / max(1, stats['requests']),
                    'avg_response_time': avg_response_time,
                    'healthy': stats['healthy'],
                    'weight': backend_info['weight']
                }
            
            return {
                'total_backends': len(self._backends),
                'healthy_backends': sum(1 for b in self._backends 
                                      if self._backend_stats[id(b['backend'])]['healthy']),
                'backend_stats': backend_stats
            }


class AsyncBatchProcessor:
    """Asynchronous batch processing with intelligent batching"""
    
    def __init__(
        self,
        processor_func: Callable,
        batch_size: int = 32,
        max_wait_time: float = 0.1,
        max_concurrent_batches: int = 4
    ):
        self.processor_func = processor_func
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_concurrent_batches = max_concurrent_batches
        
        self._pending_items = []
        self._pending_futures = []
        self._batch_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)
        self._last_batch_time = time.time()
        
        # Start batch processing task
        self._batch_task = None
    
    async def process_item(self, item: Any) -> Any:
        """Process single item through batching"""
        # Create future for result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        async with self._batch_lock:
            self._pending_items.append(item)
            self._pending_futures.append(future)
            
            # Check if batch is ready
            current_time = time.time()
            should_flush = (
                len(self._pending_items) >= self.batch_size or
                current_time - self._last_batch_time >= self.max_wait_time
            )
            
            if should_flush:
                await self._flush_batch()
        
        # Start batch processing if not already running
        if self._batch_task is None or self._batch_task.done():
            self._batch_task = asyncio.create_task(self._batch_processing_loop())
        
        return await future
    
    async def _flush_batch(self):
        """Flush current batch for processing"""
        if not self._pending_items:
            return
        
        # Get current batch
        batch_items = self._pending_items[:]
        batch_futures = self._pending_futures[:]
        
        # Clear pending
        self._pending_items.clear()
        self._pending_futures.clear()
        self._last_batch_time = time.time()
        
        # Process batch asynchronously
        asyncio.create_task(self._process_batch(batch_items, batch_futures))
    
    async def _process_batch(self, items: List[Any], futures: List[asyncio.Future]):
        """Process a batch of items"""
        async with self._semaphore:
            try:
                # Process batch
                results = await asyncio.get_event_loop().run_in_executor(
                    None, self.processor_func, items
                )
                
                # Set results
                for future, result in zip(futures, results):
                    if not future.done():
                        future.set_result(result)
                        
            except Exception as e:
                # Set exception for all futures
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
    
    async def _batch_processing_loop(self):
        """Background loop to flush batches based on time"""
        while True:
            await asyncio.sleep(self.max_wait_time / 2)
            
            async with self._batch_lock:
                if self._pending_items:
                    current_time = time.time()
                    if current_time - self._last_batch_time >= self.max_wait_time:
                        await self._flush_batch()


class PerformanceOptimizer:
    """System-wide performance optimization and monitoring"""
    
    def __init__(self):
        self.cache = HighPerformanceCache()
        self.metrics = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'operation_times': defaultdict(lambda: deque(maxlen=1000))
        }
        self._optimization_rules = []
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
    
    def cached_operation(self, cache_key: str, ttl: Optional[float] = None):
        """Decorator for caching operation results"""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Check cache first
                result = self.cache.get(cache_key)
                if result is not None:
                    return result
                
                # Execute and cache
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics
                self.metrics['operation_times'][func.__name__].append(duration)
                
                # Cache result
                self.cache.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def add_optimization_rule(self, condition: Callable, action: Callable):
        """Add performance optimization rule"""
        self._optimization_rules.append((condition, action))
    
    def _monitoring_loop(self):
        """Background system monitoring"""
        while True:
            try:
                time.sleep(10)  # Monitor every 10 seconds
                self._collect_system_metrics()
                self._apply_optimization_rules()
            except Exception:
                pass
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        if _PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                self.metrics['cpu_usage'].append(cpu_percent)
                self.metrics['memory_usage'].append(memory_percent)
            except:
                pass
    
    def _apply_optimization_rules(self):
        """Apply optimization rules based on current metrics"""
        for condition, action in self._optimization_rules:
            try:
                if condition(self.metrics):
                    action(self.metrics)
            except Exception:
                pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        cpu_usage = list(self.metrics['cpu_usage'])
        memory_usage = list(self.metrics['memory_usage'])
        
        summary = {
            'cache_stats': self.cache.get_stats(),
            'system_metrics': {
                'avg_cpu_usage': sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                'avg_memory_usage': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                'peak_cpu_usage': max(cpu_usage) if cpu_usage else 0,
                'peak_memory_usage': max(memory_usage) if memory_usage else 0
            },
            'operation_performance': {}
        }
        
        # Add operation performance metrics
        for op_name, times in self.metrics['operation_times'].items():
            times_list = list(times)
            if times_list:
                summary['operation_performance'][op_name] = {
                    'avg_time': sum(times_list) / len(times_list),
                    'min_time': min(times_list),
                    'max_time': max(times_list),
                    'total_calls': len(times_list)
                }
        
        return summary


# Global performance optimizer
_global_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


def performance_optimized(cache_ttl: Optional[float] = None):
    """Decorator for performance-optimized operations"""
    def decorator(func: Callable) -> Callable:
        optimizer = get_performance_optimizer()
        
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            @optimizer.cached_operation(cache_key, cache_ttl)
            def cached_func():
                return func(*args, **kwargs)
            
            return cached_func()
        
        return wrapper
    return decorator


# Export all scaling features
__all__ = [
    'HighPerformanceCache',
    'AdaptiveResourcePool', 
    'LoadBalancer',
    'AsyncBatchProcessor',
    'PerformanceOptimizer',
    'get_performance_optimizer',
    'performance_optimized'
]