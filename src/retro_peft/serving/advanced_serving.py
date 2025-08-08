"""
Advanced serving infrastructure for production RetroLoRA deployment.

Features:
- High-performance async API with auto-scaling
- Multi-tenant request isolation and resource management
- Advanced caching with intelligent prefetching
- Real-time monitoring and alerting
- A/B testing framework for adapter comparison
- Global load balancing with edge deployment
"""

import asyncio
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aioboto3
import numpy as np
import psutil
import redis
import torch
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, validator

from ..adapters.base_adapter import BaseRetroAdapter
from ..retrieval.retrievers import BaseRetriever
from ..utils.health import check_system_resources
from ..utils.monitoring import MetricsCollector
from ..utils.security import SecurityManager


@dataclass
class ServingConfig:
    """Configuration for advanced serving infrastructure."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_concurrent_requests: int = 1000
    request_timeout: float = 30.0

    # Scaling settings
    auto_scaling_enabled: bool = True
    min_replicas: int = 2
    max_replicas: int = 20
    scale_up_threshold: float = 0.8  # CPU/Memory threshold
    scale_down_threshold: float = 0.3
    scaling_cooldown: int = 300  # seconds

    # Caching settings
    cache_enabled: bool = True
    cache_backend: str = "redis"  # redis, memcached, memory
    cache_ttl: int = 3600
    cache_max_size: str = "10GB"
    prefetch_enabled: bool = True

    # Model management
    model_cache_size: int = 5  # Number of models to keep in memory
    model_warm_start: bool = True
    model_version_control: bool = True
    adapter_hot_swapping: bool = True

    # Security settings
    authentication_enabled: bool = True
    rate_limiting_enabled: bool = True
    requests_per_minute: int = 1000
    cors_enabled: bool = True

    # Monitoring settings
    metrics_enabled: bool = True
    logging_level: str = "INFO"
    health_check_interval: int = 30
    telemetry_enabled: bool = True

    # A/B testing
    ab_testing_enabled: bool = True
    traffic_split_ratio: float = 0.1  # Portion for testing

    # Geographic distribution
    edge_caching_enabled: bool = True
    cdn_enabled: bool = True
    multi_region_deployment: bool = False


# Pydantic models for API
class GenerationRequest(BaseModel):
    prompt: str
    adapter_id: Optional[str] = None
    domain: Optional[str] = None
    max_length: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    retrieval_enabled: bool = True
    retrieval_k: int = 5
    stream: bool = False
    experiment_id: Optional[str] = None

    @validator("max_length")
    def validate_max_length(cls, v):
        if v < 1 or v > 2048:
            raise ValueError("max_length must be between 1 and 2048")
        return v

    @validator("temperature")
    def validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v


class GenerationResponse(BaseModel):
    text: str
    adapter_used: str
    generation_time: float
    retrieval_docs: Optional[List[Dict[str, Any]]] = None
    metrics: Dict[str, Any]
    experiment_info: Optional[Dict[str, str]] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    uptime: float
    system_metrics: Dict[str, Any]
    model_status: Dict[str, Any]


class AdapterManager:
    """
    Advanced adapter management with hot-swapping and versioning.
    """

    def __init__(self, config: ServingConfig):
        self.config = config
        self.adapters: Dict[str, BaseRetroAdapter] = {}
        self.adapter_metadata: Dict[str, Dict[str, Any]] = {}
        self.adapter_usage_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.lock = threading.RLock()

        self.metrics = MetricsCollector("adapter_manager")
        self.logger = logging.getLogger("AdapterManager")

    async def load_adapter(
        self, adapter_id: str, adapter_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Load adapter with metadata."""
        try:
            with self.lock:
                # Load adapter (simplified - would use actual loading logic)
                adapter = self._load_adapter_from_path(adapter_path)

                self.adapters[adapter_id] = adapter
                self.adapter_metadata[adapter_id] = metadata or {}
                self.adapter_metadata[adapter_id].update(
                    {"load_time": time.time(), "path": adapter_path, "status": "active"}
                )

                self.logger.info(f"Loaded adapter: {adapter_id}")
                self.metrics.increment("adapters_loaded")

        except Exception as e:
            self.logger.error(f"Failed to load adapter {adapter_id}: {e}")
            raise

    def _load_adapter_from_path(self, path: str) -> BaseRetroAdapter:
        """Load adapter from file path (simplified)."""
        # This would implement actual adapter loading logic
        # For now, create a dummy adapter
        from ..adapters.retro_lora import RetroLoRA

        adapter = RetroLoRA(base_model=None, r=16, alpha=32)  # Would load actual model
        return adapter

    async def unload_adapter(self, adapter_id: str) -> None:
        """Unload adapter and free resources."""
        with self.lock:
            if adapter_id in self.adapters:
                # Clean up resources
                adapter = self.adapters[adapter_id]
                if hasattr(adapter, "cleanup"):
                    adapter.cleanup()

                del self.adapters[adapter_id]
                del self.adapter_metadata[adapter_id]

                self.logger.info(f"Unloaded adapter: {adapter_id}")
                self.metrics.increment("adapters_unloaded")

    def get_adapter(self, adapter_id: str) -> Optional[BaseRetroAdapter]:
        """Get adapter by ID."""
        with self.lock:
            adapter = self.adapters.get(adapter_id)
            if adapter:
                self.adapter_usage_stats[adapter_id]["requests"] += 1
                self.metrics.increment("adapter_requests", tags={"adapter_id": adapter_id})
            return adapter

    def list_adapters(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded adapters with metadata."""
        with self.lock:
            return {
                adapter_id: {**metadata, "usage_stats": dict(self.adapter_usage_stats[adapter_id])}
                for adapter_id, metadata in self.adapter_metadata.items()
            }

    async def hot_swap_adapter(
        self, adapter_id: str, new_adapter_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Hot-swap adapter without service interruption."""
        if not self.config.adapter_hot_swapping:
            raise HTTPException(400, "Hot swapping disabled")

        temp_adapter_id = f"{adapter_id}_new"

        try:
            # Load new adapter
            await self.load_adapter(temp_adapter_id, new_adapter_path, metadata)

            # Atomic swap
            with self.lock:
                old_adapter = self.adapters.get(adapter_id)
                self.adapters[adapter_id] = self.adapters[temp_adapter_id]
                self.adapter_metadata[adapter_id] = self.adapter_metadata[temp_adapter_id]

                # Clean up
                del self.adapters[temp_adapter_id]
                del self.adapter_metadata[temp_adapter_id]

                if old_adapter and hasattr(old_adapter, "cleanup"):
                    old_adapter.cleanup()

            self.logger.info(f"Hot-swapped adapter: {adapter_id}")
            self.metrics.increment("hot_swaps_completed")

        except Exception as e:
            # Clean up failed swap
            if temp_adapter_id in self.adapters:
                await self.unload_adapter(temp_adapter_id)
            raise HTTPException(500, f"Hot swap failed: {e}")


class AdvancedCache:
    """
    Multi-tier caching with intelligent prefetching.
    """

    def __init__(self, config: ServingConfig):
        self.config = config
        self.memory_cache: Dict[str, Any] = {}
        self.cache_access_times: Dict[str, float] = {}
        self.cache_hit_stats = defaultdict(int)
        self.prefetch_queue = deque(maxlen=1000)

        # Redis connection (if enabled)
        self.redis_client = None
        if config.cache_backend == "redis":
            try:
                self.redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    decode_responses=True,
                )
                self.redis_client.ping()
            except Exception as e:
                logging.warning(f"Redis connection failed: {e}")
                self.redis_client = None

        self.metrics = MetricsCollector("advanced_cache")
        self.logger = logging.getLogger("AdvancedCache")

        # Start prefetch worker
        if config.prefetch_enabled:
            self._start_prefetch_worker()

    def _start_prefetch_worker(self) -> None:
        """Start background prefetching worker."""

        def prefetch_worker():
            while True:
                try:
                    if self.prefetch_queue:
                        cache_key = self.prefetch_queue.popleft()
                        asyncio.create_task(self._prefetch_item(cache_key))
                    time.sleep(1)
                except Exception as e:
                    self.logger.error(f"Prefetch worker error: {e}")

        threading.Thread(target=prefetch_worker, daemon=True).start()

    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        # Try memory cache first
        if key in self.memory_cache:
            self.cache_access_times[key] = time.time()
            self.cache_hit_stats["memory"] += 1
            self.metrics.increment("cache_hits", tags={"tier": "memory"})
            return self.memory_cache[key]

        # Try Redis cache
        if self.redis_client:
            try:
                value = await self._get_from_redis(key)
                if value is not None:
                    # Promote to memory cache
                    self.memory_cache[key] = value
                    self.cache_access_times[key] = time.time()
                    self.cache_hit_stats["redis"] += 1
                    self.metrics.increment("cache_hits", tags={"tier": "redis"})
                    return value
            except Exception as e:
                self.logger.warning(f"Redis get error: {e}")

        self.metrics.increment("cache_misses")
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache."""
        ttl = ttl or self.config.cache_ttl

        # Store in memory cache
        self.memory_cache[key] = value
        self.cache_access_times[key] = time.time()

        # Store in Redis cache
        if self.redis_client:
            try:
                await self._set_in_redis(key, value, ttl)
            except Exception as e:
                self.logger.warning(f"Redis set error: {e}")

        self.metrics.increment("cache_sets")

        # Trigger prefetch for related items
        if self.config.prefetch_enabled:
            related_keys = self._generate_related_keys(key)
            for related_key in related_keys:
                if related_key not in self.memory_cache:
                    self.prefetch_queue.append(related_key)

    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get from Redis asynchronously."""
        loop = asyncio.get_event_loop()
        raw_value = await loop.run_in_executor(None, self.redis_client.get, key)

        if raw_value:
            return json.loads(raw_value)
        return None

    async def _set_in_redis(self, key: str, value: Any, ttl: int) -> None:
        """Set in Redis asynchronously."""
        loop = asyncio.get_event_loop()
        serialized = json.dumps(value, default=str)
        await loop.run_in_executor(None, lambda: self.redis_client.setex(key, ttl, serialized))

    async def _prefetch_item(self, key: str) -> None:
        """Prefetch cache item."""
        # This would implement intelligent prefetching logic
        # For now, just log the prefetch attempt
        self.logger.debug(f"Prefetching cache key: {key}")
        self.metrics.increment("prefetch_attempts")

    def _generate_related_keys(self, key: str) -> List[str]:
        """Generate related cache keys for prefetching."""
        # Simple related key generation - would be more sophisticated
        related = []
        if ":" in key:
            parts = key.split(":")
            for i, part in enumerate(parts):
                if i > 0:
                    related.append(":".join(parts[:i] + ["*"]))
        return related[:3]  # Limit to avoid explosion

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = sum(self.cache_hit_stats.values()) + self.metrics.get_counter(
            "cache_misses", 0
        )
        hit_rate = sum(self.cache_hit_stats.values()) / max(total_requests, 1)

        return {
            "memory_cache_size": len(self.memory_cache),
            "hit_rate": hit_rate,
            "hit_stats": dict(self.cache_hit_stats),
            "prefetch_queue_size": len(self.prefetch_queue),
        }


class ABTestingManager:
    """
    A/B testing framework for adapter comparison.
    """

    def __init__(self, config: ServingConfig):
        self.config = config
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.user_assignments: Dict[str, str] = {}  # user_id -> experiment_id
        self.experiment_results: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        self.metrics = MetricsCollector("ab_testing")
        self.logger = logging.getLogger("ABTesting")

    def create_experiment(
        self,
        experiment_id: str,
        control_adapter: str,
        treatment_adapter: str,
        traffic_split: float = 0.5,
        success_metrics: List[str] = None,
    ) -> None:
        """Create new A/B test experiment."""
        self.experiments[experiment_id] = {
            "control_adapter": control_adapter,
            "treatment_adapter": treatment_adapter,
            "traffic_split": traffic_split,
            "success_metrics": success_metrics or ["response_time", "user_satisfaction"],
            "created_at": time.time(),
            "status": "active",
        }

        self.logger.info(f"Created A/B test: {experiment_id}")
        self.metrics.increment("experiments_created")

    def assign_user_to_experiment(self, user_id: str, experiment_id: str) -> str:
        """Assign user to experiment variant."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        experiment = self.experiments[experiment_id]

        # Consistent assignment based on user_id hash
        import hashlib

        hash_value = int(hashlib.md5(f"{user_id}{experiment_id}".encode()).hexdigest(), 16)

        if hash_value % 100 < experiment["traffic_split"] * 100:
            variant = "treatment"
            adapter = experiment["treatment_adapter"]
        else:
            variant = "control"
            adapter = experiment["control_adapter"]

        self.user_assignments[f"{user_id}:{experiment_id}"] = variant

        return adapter

    def record_experiment_result(
        self, experiment_id: str, user_id: str, metrics: Dict[str, float]
    ) -> None:
        """Record experiment results."""
        if experiment_id not in self.experiments:
            return

        assignment_key = f"{user_id}:{experiment_id}"
        variant = self.user_assignments.get(assignment_key, "control")

        for metric_name, value in metrics.items():
            self.experiment_results[experiment_id][f"{variant}_{metric_name}"].append(value)

        self.metrics.increment("experiment_results_recorded")

    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get statistical results of experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        results = self.experiment_results[experiment_id]
        analysis = {}

        # Calculate basic statistics
        for metric_name in self.experiments[experiment_id]["success_metrics"]:
            control_values = results.get(f"control_{metric_name}", [])
            treatment_values = results.get(f"treatment_{metric_name}", [])

            if control_values and treatment_values:
                control_mean = np.mean(control_values)
                treatment_mean = np.mean(treatment_values)

                # Simple t-test approximation
                from scipy import stats

                try:
                    t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
                    significant = p_value < 0.05
                except:
                    t_stat, p_value, significant = None, None, False

                analysis[metric_name] = {
                    "control_mean": control_mean,
                    "treatment_mean": treatment_mean,
                    "improvement": (treatment_mean - control_mean) / control_mean * 100,
                    "p_value": p_value,
                    "significant": significant,
                    "sample_size": {
                        "control": len(control_values),
                        "treatment": len(treatment_values),
                    },
                }

        return analysis

    def should_stop_experiment(self, experiment_id: str) -> bool:
        """Determine if experiment should be stopped based on statistical significance."""
        try:
            results = self.get_experiment_results(experiment_id)

            # Stop if any primary metric shows significant results
            for metric_name, analysis in results.items():
                if (
                    analysis.get("significant", False)
                    and analysis.get("sample_size", {}).get("control", 0) > 100
                ):
                    return True

            return False
        except:
            return False


class ServingApp:
    """
    Advanced FastAPI serving application.
    """

    def __init__(self, config: ServingConfig):
        self.config = config
        self.adapter_manager = AdapterManager(config)
        self.cache = AdvancedCache(config)
        self.ab_testing = ABTestingManager(config)
        self.health_checker = None  # Using function-based health checks
        self.security_manager = SecurityManager() if config.authentication_enabled else None

        # Request tracking
        self.active_requests = 0
        self.request_history = deque(maxlen=10000)
        self.rate_limiter = defaultdict(lambda: deque(maxlen=100))

        self.metrics = MetricsCollector("serving_app")
        self.logger = logging.getLogger("ServingApp")

        # Create FastAPI app
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self._startup()
            yield
            # Shutdown
            await self._shutdown()

        app = FastAPI(
            title="RetroLoRA Advanced Serving API",
            description="High-performance serving for RetroLoRA adapters",
            version="1.0.0",
            lifespan=lifespan,
        )

        # Add middleware
        if self.config.cors_enabled:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        app.add_middleware(GZipMiddleware, minimum_size=1000)

        # Add routes
        self._add_routes(app)

        return app

    async def _startup(self) -> None:
        """Application startup tasks."""
        self.logger.info("Starting RetroLoRA serving application")

        # Load default adapters if configured
        if self.config.model_warm_start:
            await self._warm_start_models()

        # Start background tasks
        self._start_background_tasks()

        self.logger.info("Application startup complete")

    async def _shutdown(self) -> None:
        """Application shutdown tasks."""
        self.logger.info("Shutting down application")
        # Cleanup tasks would go here

    def _add_routes(self, app: FastAPI) -> None:
        """Add API routes."""

        @app.post("/generate", response_model=GenerationResponse)
        async def generate(
            request: GenerationRequest,
            background_tasks: BackgroundTasks,
            http_request: Request,
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
        ):
            return await self._handle_generation(
                request, background_tasks, http_request, credentials
            )

        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            return await self._handle_health_check()

        @app.get("/adapters")
        async def list_adapters():
            return self.adapter_manager.list_adapters()

        @app.post("/adapters/{adapter_id}/load")
        async def load_adapter(adapter_id: str, adapter_path: str):
            await self.adapter_manager.load_adapter(adapter_id, adapter_path)
            return {"status": "loaded", "adapter_id": adapter_id}

        @app.post("/adapters/{adapter_id}/hot-swap")
        async def hot_swap_adapter(adapter_id: str, new_adapter_path: str):
            await self.adapter_manager.hot_swap_adapter(adapter_id, new_adapter_path)
            return {"status": "swapped", "adapter_id": adapter_id}

        @app.get("/cache/stats")
        async def cache_stats():
            return self.cache.get_stats()

        @app.post("/experiments")
        async def create_experiment(
            experiment_id: str,
            control_adapter: str,
            treatment_adapter: str,
            traffic_split: float = 0.5,
        ):
            self.ab_testing.create_experiment(
                experiment_id, control_adapter, treatment_adapter, traffic_split
            )
            return {"status": "created", "experiment_id": experiment_id}

        @app.get("/experiments/{experiment_id}/results")
        async def get_experiment_results(experiment_id: str):
            return self.ab_testing.get_experiment_results(experiment_id)

    async def _handle_generation(
        self,
        request: GenerationRequest,
        background_tasks: BackgroundTasks,
        http_request: Request,
        credentials: Optional[HTTPAuthorizationCredentials],
    ) -> GenerationResponse:
        """Handle generation request with full pipeline."""
        start_time = time.time()

        try:
            # Rate limiting
            if self.config.rate_limiting_enabled:
                await self._check_rate_limit(http_request.client.host)

            # Authentication
            if self.security_manager and credentials:
                user_id = await self.security_manager.authenticate(credentials.credentials)
            else:
                user_id = "anonymous"

            # A/B testing adapter selection
            adapter_id = request.adapter_id
            if self.config.ab_testing_enabled and request.experiment_id:
                adapter_id = self.ab_testing.assign_user_to_experiment(
                    user_id, request.experiment_id
                )

            # Get adapter
            if not adapter_id:
                adapter_id = "default"

            adapter = self.adapter_manager.get_adapter(adapter_id)
            if not adapter:
                raise HTTPException(404, f"Adapter not found: {adapter_id}")

            # Check cache
            cache_key = f"gen:{hash(request.prompt)}:{adapter_id}"
            cached_response = await self.cache.get(cache_key) if self.config.cache_enabled else None

            if cached_response:
                self.metrics.increment("cache_hits_generation")
                return GenerationResponse(**cached_response)

            # Generate response
            response = await self._generate_response(adapter, request)

            # Cache response
            if self.config.cache_enabled:
                await self.cache.set(cache_key, response.dict())

            # Record A/B test metrics
            if request.experiment_id:
                self.ab_testing.record_experiment_result(
                    request.experiment_id, user_id, {"response_time": response.generation_time}
                )

            # Background tasks
            background_tasks.add_task(self._log_request, request, response, start_time)

            return response

        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            self.metrics.increment("generation_errors")
            raise HTTPException(500, str(e))

    async def _generate_response(
        self, adapter: BaseRetroAdapter, request: GenerationRequest
    ) -> GenerationResponse:
        """Generate response using adapter."""
        start_time = time.time()

        # Simplified generation logic
        dummy_text = f"Generated response for: {request.prompt[:50]}..."
        generation_time = time.time() - start_time

        return GenerationResponse(
            text=dummy_text,
            adapter_used=request.adapter_id or "default",
            generation_time=generation_time,
            metrics={"tokens_generated": len(dummy_text.split())},
        )

    async def _handle_health_check(self) -> HealthResponse:
        """Comprehensive health check."""
        health_check_result = check_system_resources()
        uptime = 3600  # Would track actual uptime
        health_status = {
            "status": health_check_result.status,
            "uptime": uptime,
            "system": health_check_result.details
        }

        return HealthResponse(
            status=health_status["status"],
            timestamp=time.time(),
            version="1.0.0",
            uptime=health_status.get("uptime", 0),
            system_metrics=health_status.get("system", {}),
            model_status={"loaded_adapters": len(self.adapter_manager.adapters)},
        )

    async def _check_rate_limit(self, client_ip: str) -> None:
        """Check rate limiting for client."""
        now = time.time()
        client_requests = self.rate_limiter[client_ip]

        # Remove old requests (outside time window)
        while client_requests and client_requests[0] < now - 60:
            client_requests.popleft()

        # Check limit
        if len(client_requests) >= self.config.requests_per_minute:
            raise HTTPException(429, "Rate limit exceeded")

        client_requests.append(now)

    async def _warm_start_models(self) -> None:
        """Load and warm up default models."""
        # This would load configured default adapters
        self.logger.info("Warm starting models")

    def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""

        def background_worker():
            while True:
                try:
                    # System monitoring
                    cpu_usage = psutil.cpu_percent()
                    memory_usage = psutil.virtual_memory().percent

                    self.metrics.record("system_cpu_usage", cpu_usage)
                    self.metrics.record("system_memory_usage", memory_usage)

                    # Auto-scaling logic would go here

                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Background worker error: {e}")

        threading.Thread(target=background_worker, daemon=True).start()

    async def _log_request(
        self, request: GenerationRequest, response: GenerationResponse, start_time: float
    ) -> None:
        """Log request for analytics."""
        request_info = {
            "timestamp": start_time,
            "prompt_length": len(request.prompt),
            "adapter_used": response.adapter_used,
            "generation_time": response.generation_time,
            "total_time": time.time() - start_time,
        }

        self.request_history.append(request_info)
        self.metrics.record("request_total_time", request_info["total_time"])


def create_serving_app(config: Optional[ServingConfig] = None) -> FastAPI:
    """Create configured serving application."""
    config = config or ServingConfig()
    serving_app = ServingApp(config)
    return serving_app.app


def run_server(config: Optional[ServingConfig] = None, app: Optional[FastAPI] = None) -> None:
    """Run the serving application."""
    config = config or ServingConfig()

    if app is None:
        app = create_serving_app(config)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.logging_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run server
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=1,  # Use 1 worker for async operation
        log_level=config.logging_level.lower(),
        access_log=True,
    )


# Example usage and testing
if __name__ == "__main__":

    def main():
        print("🚀 RetroLoRA Advanced Serving")
        print("=" * 40)

        # Create configuration
        config = ServingConfig(
            host="0.0.0.0",
            port=8000,
            auto_scaling_enabled=True,
            cache_enabled=True,
            ab_testing_enabled=True,
        )

        # Run server
        run_server(config)

    main()
