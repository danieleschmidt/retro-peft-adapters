"""
API endpoint implementations for the gateway.

Provides RESTful endpoints for inference, health checks, and metrics.
"""

import json
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..utils.logging import get_global_logger
from .auth import AuthResult


# Request/Response Models
class InferenceRequest(BaseModel):
    """Inference request model"""

    prompt: str = Field(..., description="Input prompt for generation")
    retrieval_k: int = Field(5, description="Number of documents to retrieve")
    retrieval_enabled: bool = Field(True, description="Enable retrieval augmentation")
    max_length: Optional[int] = Field(None, description="Maximum generation length")
    temperature: Optional[float] = Field(None, description="Generation temperature")
    stream: bool = Field(False, description="Enable streaming response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Request metadata")


class BatchInferenceRequest(BaseModel):
    """Batch inference request model"""

    prompts: List[str] = Field(..., description="List of input prompts")
    retrieval_k: int = Field(5, description="Number of documents to retrieve")
    retrieval_enabled: bool = Field(True, description="Enable retrieval augmentation")
    max_length: Optional[int] = Field(None, description="Maximum generation length")
    temperature: Optional[float] = Field(None, description="Generation temperature")
    return_retrieval_info: bool = Field(False, description="Include retrieval information")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Request metadata")


class InferenceResponse(BaseModel):
    """Inference response model"""

    id: str = Field(..., description="Request ID")
    generated_text: str = Field(..., description="Generated text")
    success: bool = Field(..., description="Whether generation was successful")
    processing_time: float = Field(..., description="Processing time in milliseconds")
    retrieval_info: Optional[Dict[str, Any]] = Field(None, description="Retrieval information")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Response metadata")


class BatchInferenceResponse(BaseModel):
    """Batch inference response model"""

    results: List[InferenceResponse] = Field(..., description="List of inference results")
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    processing_time: float = Field(..., description="Total processing time")


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str = Field(..., description="Health status")
    timestamp: float = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds")
    checks: Dict[str, Dict[str, Any]] = Field(..., description="Individual health checks")


class MetricsResponse(BaseModel):
    """Metrics response model"""

    timestamp: float = Field(..., description="Metrics timestamp")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    resources: Dict[str, Any] = Field(..., description="Resource metrics")
    scaling: Dict[str, Any] = Field(..., description="Scaling metrics")


class InferenceAPI:
    """
    Inference API endpoints.
    """

    def __init__(self, gateway):
        """
        Initialize inference API.

        Args:
            gateway: API gateway instance
        """
        self.gateway = gateway
        self.router = APIRouter()
        self.logger = get_global_logger()

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup inference API routes"""

        @self.router.post("/generate", response_model=InferenceResponse)
        async def generate(
            request: InferenceRequest,
            auth_result: AuthResult = Depends(self.gateway.get_auth_dependency()),
            rate_limit_ok: bool = Depends(self.gateway.get_rate_limit_dependency()),
        ):
            """Generate text using retrieval-augmented inference"""
            start_time = time.time()
            request_id = f"gen_{int(start_time * 1000000)}"

            try:
                # Validate permissions
                if self.gateway.config.enable_auth:
                    has_permission = await self.gateway.auth_manager.validate_permissions(
                        auth_result.user_id, ["read"]
                    )
                    if not has_permission:
                        raise HTTPException(status_code=403, detail="Insufficient permissions")

                # Generate response
                if request.stream:
                    # Streaming not implemented in this example
                    raise HTTPException(status_code=501, detail="Streaming not implemented")

                response = await self.gateway.pipeline.generate(
                    prompt=request.prompt,
                    retrieval_k=request.retrieval_k,
                    retrieval_enabled=request.retrieval_enabled,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    metadata=request.metadata,
                )

                processing_time = (time.time() - start_time) * 1000

                return InferenceResponse(
                    id=request_id,
                    generated_text=response["generated_text"],
                    success=response["success"],
                    processing_time=processing_time,
                    retrieval_info=response.get("retrieval_info"),
                    metadata=response.get("metadata"),
                )

            except Exception as e:
                self.logger.error(f"Inference error for request {request_id}: {e}")
                processing_time = (time.time() - start_time) * 1000

                if isinstance(e, HTTPException):
                    raise e

                return InferenceResponse(
                    id=request_id,
                    generated_text="",
                    success=False,
                    processing_time=processing_time,
                    metadata={"error": str(e)},
                )

        @self.router.post("/batch", response_model=BatchInferenceResponse)
        async def batch_generate(
            request: BatchInferenceRequest,
            auth_result: AuthResult = Depends(self.gateway.get_auth_dependency()),
            rate_limit_ok: bool = Depends(self.gateway.get_rate_limit_dependency()),
        ):
            """Generate text for multiple prompts"""
            start_time = time.time()

            try:
                # Validate permissions
                if self.gateway.config.enable_auth:
                    has_permission = await self.gateway.auth_manager.validate_permissions(
                        auth_result.user_id, ["read"]
                    )
                    if not has_permission:
                        raise HTTPException(status_code=403, detail="Insufficient permissions")

                # Validate batch size
                if len(request.prompts) > 100:  # Configurable limit
                    raise HTTPException(status_code=400, detail="Batch size too large")

                # Generate responses
                responses = await self.gateway.pipeline.generate(
                    prompt=request.prompts,
                    retrieval_k=request.retrieval_k,
                    retrieval_enabled=request.retrieval_enabled,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    metadata=request.metadata,
                )

                processing_time = (time.time() - start_time) * 1000

                # Convert to response format
                results = []
                successful_count = 0

                for i, response in enumerate(responses):
                    if response["success"]:
                        successful_count += 1

                    result = InferenceResponse(
                        id=f"batch_{int(start_time * 1000000)}_{i}",
                        generated_text=response["generated_text"],
                        success=response["success"],
                        processing_time=response.get("processing_time", 0),
                        retrieval_info=(
                            response.get("retrieval_info")
                            if request.return_retrieval_info
                            else None
                        ),
                        metadata=response.get("metadata"),
                    )
                    results.append(result)

                return BatchInferenceResponse(
                    results=results,
                    total_requests=len(request.prompts),
                    successful_requests=successful_count,
                    processing_time=processing_time,
                )

            except Exception as e:
                self.logger.error(f"Batch inference error: {e}")

                if isinstance(e, HTTPException):
                    raise e

                processing_time = (time.time() - start_time) * 1000

                # Return error results for all prompts
                results = []
                for i, prompt in enumerate(request.prompts):
                    result = InferenceResponse(
                        id=f"batch_error_{int(start_time * 1000000)}_{i}",
                        generated_text="",
                        success=False,
                        processing_time=0,
                        metadata={"error": str(e)},
                    )
                    results.append(result)

                return BatchInferenceResponse(
                    results=results,
                    total_requests=len(request.prompts),
                    successful_requests=0,
                    processing_time=processing_time,
                )

        @self.router.get("/models")
        async def list_models(
            auth_result: AuthResult = Depends(self.gateway.get_auth_dependency()),
        ):
            """List available models"""
            return {
                "models": [
                    {
                        "id": "retro-peft-default",
                        "name": "Retro-PEFT Default Model",
                        "description": "Default retrieval-augmented model",
                        "capabilities": ["text-generation", "retrieval-augmentation"],
                    }
                ]
            }


class HealthAPI:
    """
    Health check API endpoints.
    """

    def __init__(self, gateway):
        """
        Initialize health API.

        Args:
            gateway: API gateway instance
        """
        self.gateway = gateway
        self.router = APIRouter()
        self.logger = get_global_logger()
        self.start_time = time.time()

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup health API routes"""

        @self.router.get("/", response_model=HealthResponse)
        async def health_check():
            """Comprehensive health check"""
            try:
                checks = {}
                overall_status = "healthy"

                # Pipeline health
                if self.gateway.pipeline:
                    pipeline_stats = self.gateway.pipeline.get_stats()
                    checks["pipeline"] = {
                        "status": "healthy" if pipeline_stats else "unhealthy",
                        "details": pipeline_stats,
                    }
                    if not pipeline_stats:
                        overall_status = "unhealthy"

                # Load balancer health
                if self.gateway.load_balancer:
                    lb_stats = self.gateway.load_balancer.get_stats()
                    healthy_backends = lb_stats.get("healthy_backends", 0)
                    checks["load_balancer"] = {
                        "status": "healthy" if healthy_backends > 0 else "unhealthy",
                        "details": lb_stats,
                    }
                    if healthy_backends == 0:
                        overall_status = "degraded"

                # Auth system health
                if self.gateway.auth_manager:
                    auth_stats = self.gateway.auth_manager.get_stats()
                    checks["auth"] = {"status": "healthy", "details": auth_stats}

                # Rate limiter health
                if self.gateway.rate_limiter:
                    rl_stats = self.gateway.rate_limiter.get_stats()
                    checks["rate_limiter"] = {"status": "healthy", "details": rl_stats}

                return HealthResponse(
                    status=overall_status,
                    timestamp=time.time(),
                    version="1.0.0",
                    uptime=time.time() - self.start_time,
                    checks=checks,
                )

            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                return HealthResponse(
                    status="unhealthy",
                    timestamp=time.time(),
                    version="1.0.0",
                    uptime=time.time() - self.start_time,
                    checks={"error": {"status": "unhealthy", "details": str(e)}},
                )

        @self.router.get("/ready")
        async def readiness_check():
            """Readiness check for Kubernetes"""
            try:
                # Check if pipeline is ready
                if not self.gateway.pipeline or not self.gateway.pipeline._running:
                    raise HTTPException(status_code=503, detail="Pipeline not ready")

                return {"status": "ready", "timestamp": time.time()}

            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Service not ready: {e}")

        @self.router.get("/live")
        async def liveness_check():
            """Liveness check for Kubernetes"""
            return {"status": "alive", "timestamp": time.time()}


class MetricsAPI:
    """
    Metrics API endpoints.
    """

    def __init__(self, gateway):
        """
        Initialize metrics API.

        Args:
            gateway: API gateway instance
        """
        self.gateway = gateway
        self.router = APIRouter()
        self.logger = get_global_logger()

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup metrics API routes"""

        @self.router.get("/", response_model=MetricsResponse)
        async def get_metrics(
            auth_result: AuthResult = Depends(self.gateway.get_auth_dependency()),
        ):
            """Get comprehensive metrics"""
            try:
                # Validate admin permissions
                if self.gateway.config.enable_auth:
                    has_permission = await self.gateway.auth_manager.validate_permissions(
                        auth_result.user_id, ["admin"]
                    )
                    if not has_permission:
                        raise HTTPException(status_code=403, detail="Admin permissions required")

                # Collect metrics
                performance_metrics = self.gateway.metrics.get_performance_summary()

                # Resource metrics (would integrate with system monitoring)
                resource_metrics = {"cpu_usage": 0.0, "memory_usage": 0.0, "gpu_usage": 0.0}

                # Scaling metrics
                scaling_metrics = {
                    "recommendations": self.gateway.metrics.get_scaling_recommendations(),
                    "load_balancer": (
                        self.gateway.load_balancer.get_stats()
                        if self.gateway.load_balancer
                        else None
                    ),
                }

                return MetricsResponse(
                    timestamp=time.time(),
                    performance=performance_metrics,
                    resources=resource_metrics,
                    scaling=scaling_metrics,
                )

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Metrics error: {e}")
                raise HTTPException(status_code=500, detail="Failed to collect metrics")

        @self.router.get("/performance")
        async def get_performance_metrics(
            auth_result: AuthResult = Depends(self.gateway.get_auth_dependency()),
        ):
            """Get performance metrics only"""
            try:
                # Basic read permissions for performance metrics
                if self.gateway.config.enable_auth:
                    has_permission = await self.gateway.auth_manager.validate_permissions(
                        auth_result.user_id, ["read"]
                    )
                    if not has_permission:
                        raise HTTPException(status_code=403, detail="Insufficient permissions")

                return self.gateway.metrics.get_performance_summary()

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Performance metrics error: {e}")
                raise HTTPException(status_code=500, detail="Failed to collect performance metrics")

        @self.router.get("/export")
        async def export_metrics(
            format: str = "prometheus",
            auth_result: AuthResult = Depends(self.gateway.get_auth_dependency()),
        ):
            """Export metrics in various formats"""
            try:
                # Admin permissions for metric export
                if self.gateway.config.enable_auth:
                    has_permission = await self.gateway.auth_manager.validate_permissions(
                        auth_result.user_id, ["admin"]
                    )
                    if not has_permission:
                        raise HTTPException(status_code=403, detail="Admin permissions required")

                if format not in ["prometheus", "json"]:
                    raise HTTPException(status_code=400, detail="Unsupported format")

                metrics_data = self.gateway.metrics.export_metrics(format)

                if format == "prometheus":
                    return Response(content=metrics_data, media_type="text/plain")
                else:
                    return JSONResponse(content=json.loads(metrics_data))

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Metrics export error: {e}")
                raise HTTPException(status_code=500, detail="Failed to export metrics")

        @self.router.get("/scaling/recommendations")
        async def get_scaling_recommendations(
            auth_result: AuthResult = Depends(self.gateway.get_auth_dependency()),
        ):
            """Get scaling recommendations"""
            try:
                # Admin permissions for scaling recommendations
                if self.gateway.config.enable_auth:
                    has_permission = await self.gateway.auth_manager.validate_permissions(
                        auth_result.user_id, ["admin"]
                    )
                    if not has_permission:
                        raise HTTPException(status_code=403, detail="Admin permissions required")

                recommendations = self.gateway.metrics.get_scaling_recommendations()

                return {
                    "timestamp": time.time(),
                    "recommendations": recommendations,
                    "total_recommendations": len(recommendations),
                    "high_priority": len(
                        [r for r in recommendations if r.get("priority") == "high"]
                    ),
                    "critical": len(
                        [r for r in recommendations if r.get("priority") == "critical"]
                    ),
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Scaling recommendations error: {e}")
                raise HTTPException(status_code=500, detail="Failed to get scaling recommendations")
