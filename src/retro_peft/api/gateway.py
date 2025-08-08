"""
Production API Gateway for retro-peft-adapters.

High-performance gateway with authentication, rate limiting,
load balancing, and comprehensive monitoring.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import prometheus_client
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from ..scaling.async_pipeline import AsyncRetrievalPipeline
from ..scaling.load_balancer import Backend, LoadBalancer
from ..scaling.metrics import ScalingMetrics
from ..utils.logging import get_global_logger
from .auth import AuthenticationManager, AuthResult
from .endpoints import HealthAPI, InferenceAPI, MetricsAPI
from .rate_limiter import RateLimiter


@dataclass
class GatewayConfig:
    """API Gateway configuration"""

    host: str = "0.0.0.0"
    port: int = 8000
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: float = 30.0
    enable_cors: bool = True
    enable_gzip: bool = True
    enable_metrics: bool = True
    enable_auth: bool = True
    enable_rate_limiting: bool = True
    workers: int = 1
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])


class APIGateway:
    """
    Production-ready API Gateway for retrieval-augmented inference.

    Features:
    - High-performance async FastAPI server
    - JWT and API key authentication
    - Rate limiting and request throttling
    - Load balancing across model instances
    - Comprehensive monitoring and metrics
    - Circuit breakers and fault tolerance
    - Request/response logging and tracing
    """

    def __init__(
        self,
        config: GatewayConfig,
        pipeline: AsyncRetrievalPipeline,
        load_balancer: Optional[LoadBalancer] = None,
        auth_manager: Optional[AuthenticationManager] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """
        Initialize API Gateway.

        Args:
            config: Gateway configuration
            pipeline: Async inference pipeline
            load_balancer: Optional load balancer
            auth_manager: Optional authentication manager
            rate_limiter: Optional rate limiter
        """
        self.config = config
        self.pipeline = pipeline
        self.load_balancer = load_balancer
        self.auth_manager = auth_manager
        self.rate_limiter = rate_limiter

        # Metrics
        self.metrics = ScalingMetrics()

        # FastAPI app
        self.app = FastAPI(
            title="Retro-PEFT API Gateway",
            description="Production API for retrieval-augmented parameter-efficient fine-tuning",
            version="1.0.0",
            docs_url="/docs" if not config.enable_auth else None,  # Disable in production
            redoc_url="/redoc" if not config.enable_auth else None,
        )

        # API endpoints
        self.inference_api = InferenceAPI(self)
        self.health_api = HealthAPI(self)
        self.metrics_api = MetricsAPI(self)

        self.logger = get_global_logger()

        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_error_handlers()

    def _setup_middleware(self):
        """Setup FastAPI middleware"""

        # CORS middleware
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=self.config.cors_methods,
                allow_headers=self.config.cors_headers,
            )

        # Gzip compression
        if self.config.enable_gzip:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)

        # Request logging and metrics middleware
        @self.app.middleware("http")
        async def logging_middleware(request: Request, call_next):
            start_time = time.time()
            request_id = f"req_{int(start_time * 1000000)}"

            # Log request
            self.logger.info(
                f"Request {request_id}: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else "unknown",
                },
            )

            # Process request
            try:
                response = await call_next(request)

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # ms

                # Record metrics
                if self.config.enable_metrics:
                    self.metrics.record_request(
                        service="api_gateway",
                        latency_ms=processing_time,
                        success=response.status_code < 400,
                        labels={
                            "method": request.method,
                            "path": request.url.path,
                            "status_code": str(response.status_code),
                        },
                    )

                # Log response
                self.logger.info(
                    f"Response {request_id}: {response.status_code} in {processing_time:.2f}ms",
                    extra={
                        "request_id": request_id,
                        "status_code": response.status_code,
                        "processing_time_ms": processing_time,
                    },
                )

                return response

            except Exception as e:
                processing_time = (time.time() - start_time) * 1000

                # Record error metrics
                if self.config.enable_metrics:
                    self.metrics.record_request(
                        service="api_gateway",
                        latency_ms=processing_time,
                        success=False,
                        labels={
                            "method": request.method,
                            "path": request.url.path,
                            "error_type": type(e).__name__,
                        },
                    )

                # Log error
                self.logger.error(
                    f"Request {request_id} failed: {e}",
                    extra={
                        "request_id": request_id,
                        "error": str(e),
                        "processing_time_ms": processing_time,
                    },
                )

                raise

        # Request size limitation middleware
        @self.app.middleware("http")
        async def request_size_middleware(request: Request, call_next):
            if request.headers.get("content-length"):
                content_length = int(request.headers["content-length"])
                if content_length > self.config.max_request_size:
                    return JSONResponse(status_code=413, content={"error": "Request too large"})

            return await call_next(request)

    def _setup_routes(self):
        """Setup API routes"""

        # Include API routers
        self.app.include_router(
            self.inference_api.router, prefix="/v1/inference", tags=["inference"]
        )

        self.app.include_router(self.health_api.router, prefix="/health", tags=["health"])

        if self.config.enable_metrics:
            self.app.include_router(self.metrics_api.router, prefix="/metrics", tags=["metrics"])

        # Root endpoint
        @self.app.get("/")
        async def root():
            return {
                "service": "Retro-PEFT API Gateway",
                "version": "1.0.0",
                "status": "healthy",
                "timestamp": time.time(),
            }

        # Prometheus metrics endpoint
        if self.config.enable_metrics:

            @self.app.get("/metrics/prometheus")
            async def prometheus_metrics():
                return Response(prometheus_client.generate_latest(), media_type="text/plain")

    def _setup_error_handlers(self):
        """Setup custom error handlers"""

        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": exc.detail,
                    "status_code": exc.status_code,
                    "timestamp": time.time(),
                },
            )

        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            self.logger.error(f"Unhandled exception: {exc}", exc_info=True)

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "status_code": 500,
                    "timestamp": time.time(),
                },
            )

    async def authenticate_request(self, request: Request) -> AuthResult:
        """
        Authenticate incoming request.

        Args:
            request: FastAPI request object

        Returns:
            Authentication result
        """
        if not self.config.enable_auth or not self.auth_manager:
            # Auth disabled, allow all requests
            return AuthResult(success=True, user_id="anonymous")

        # Extract auth header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing Authorization header")

        # Authenticate
        auth_result = await self.auth_manager.authenticate(auth_header)

        if not auth_result.success:
            raise HTTPException(
                status_code=401, detail=auth_result.error or "Authentication failed"
            )

        return auth_result

    async def check_rate_limit(self, request: Request, user_id: str) -> bool:
        """
        Check rate limit for request.

        Args:
            request: FastAPI request object
            user_id: User identifier

        Returns:
            True if request is allowed, False otherwise
        """
        if not self.config.enable_rate_limiting or not self.rate_limiter:
            return True

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit
        allowed = await self.rate_limiter.check_rate_limit(
            key=f"{user_id}:{client_ip}", endpoint=request.url.path
        )

        if not allowed:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        return True

    def get_auth_dependency(self) -> Callable:
        """Get FastAPI dependency for authentication"""

        async def auth_dependency(request: Request) -> AuthResult:
            return await self.authenticate_request(request)

        return auth_dependency

    def get_rate_limit_dependency(self) -> Callable:
        """Get FastAPI dependency for rate limiting"""

        async def rate_limit_dependency(
            request: Request, auth_result: AuthResult = Depends(self.get_auth_dependency())
        ) -> bool:
            return await self.check_rate_limit(request, auth_result.user_id)

        return rate_limit_dependency

    async def start(self):
        """Start the API gateway"""
        # Start pipeline
        if not self.pipeline._running:
            await self.pipeline.start()

        self.logger.info(f"Starting API Gateway on {self.config.host}:{self.config.port}")

        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app=self.app,
            host=self.config.host,
            port=self.config.port,
            workers=1,  # Use 1 worker for async app
            loop="asyncio",
            access_log=False,  # We handle logging in middleware
            log_config=None,  # Disable uvicorn logging config
        )

        # Start server
        server = uvicorn.Server(uvicorn_config)
        await server.serve()

    async def stop(self):
        """Stop the API gateway"""
        # Stop pipeline
        if self.pipeline._running:
            await self.pipeline.stop()

        # Stop metrics
        self.metrics.stop()

        self.logger.info("API Gateway stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get gateway statistics"""
        stats = {
            "gateway": {
                "config": {
                    "host": self.config.host,
                    "port": self.config.port,
                    "enable_auth": self.config.enable_auth,
                    "enable_rate_limiting": self.config.enable_rate_limiting,
                    "enable_metrics": self.config.enable_metrics,
                },
                "uptime": time.time(),  # Would track actual uptime
            },
            "pipeline": self.pipeline.get_stats() if self.pipeline else None,
            "load_balancer": self.load_balancer.get_stats() if self.load_balancer else None,
            "metrics": (
                self.metrics.get_performance_summary() if self.config.enable_metrics else None
            ),
        }

        return stats


# Context manager for gateway lifecycle
@asynccontextmanager
async def api_gateway(
    config: GatewayConfig, pipeline: AsyncRetrievalPipeline, **kwargs
) -> APIGateway:
    """Context manager for API gateway lifecycle"""
    gateway = APIGateway(config, pipeline, **kwargs)

    try:
        yield gateway
    finally:
        await gateway.stop()
