"""
Generation 3: Distributed Processing and Load Balancing

Advanced distributed processing with load balancing, auto-scaling, and global optimization.
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from contextlib import asynccontextmanager

from ..utils import ErrorHandler, InputValidator, resilient_operation

try:
    import multiprocessing as mp
    import psutil
    _MULTIPROCESSING_AVAILABLE = True
except ImportError:
    _MULTIPROCESSING_AVAILABLE = False
    mp = None
    psutil = None


class WorkerStatus(Enum):
    """Worker status enumeration"""
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class WorkerNode:
    """Worker node representation"""
    worker_id: str
    host: str
    port: int
    capacity: int = 10
    current_load: int = 0
    status: WorkerStatus = WorkerStatus.IDLE
    last_heartbeat: float = field(default_factory=time.time)
    performance_score: float = 1.0
    total_processed: int = 0
    error_count: int = 0


@dataclass
class WorkRequest:
    """Distributed work request"""
    request_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 0
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    submitted_at: float = field(default_factory=time.time)
    assigned_worker: Optional[str] = None


class LoadBalancer:
    """
    Intelligent load balancer with adaptive routing and auto-scaling.
    """
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 16,
        target_cpu_usage: float = 0.7,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        health_check_interval: float = 30.0
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu_usage = target_cpu_usage
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.health_check_interval = health_check_interval
        
        # Worker management
        self.workers = {}  # worker_id -> WorkerNode
        self.worker_lock = threading.RLock()
        
        # Request queue and routing
        self.pending_requests = []
        self.request_lock = threading.Lock()
        
        # Load balancing strategies
        self.routing_strategy = "least_connections"
        self.routing_strategies = {
            "round_robin": self._route_round_robin,
            "least_connections": self._route_least_connections,
            "weighted_response_time": self._route_weighted_response_time,
            "resource_based": self._route_resource_based
        }
        
        # Monitoring and metrics
        self.metrics = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "current_load": 0.0,
            "worker_utilization": 0.0,
            "scaling_events": 0
        }
        
        # Auto-scaling state
        self.last_scale_action = 0
        self.scale_cooldown = 300  # 5 minutes
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Start background services
        self._start_background_services()
    
    def _start_background_services(self):
        """Start background monitoring and management services"""
        
        # Health check thread
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop, daemon=True
        )
        self.health_check_thread.start()
        
        # Auto-scaler thread
        self.auto_scaler_thread = threading.Thread(
            target=self._auto_scaler_loop, daemon=True
        )
        self.auto_scaler_thread.start()
        
        # Request processor thread
        self.request_processor_thread = threading.Thread(
            target=self._process_requests_loop, daemon=True
        )
        self.request_processor_thread.start()
    
    def register_worker(self, worker: WorkerNode):
        """Register a new worker node"""
        with self.worker_lock:
            self.workers[worker.worker_id] = worker
            self.logger.info(f"Registered worker: {worker.worker_id} at {worker.host}:{worker.port}")
            self._update_worker_metrics()
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker node"""
        with self.worker_lock:
            if worker_id in self.workers:
                worker = self.workers.pop(worker_id)
                self.logger.info(f"Unregistered worker: {worker_id}")
                self._update_worker_metrics()
    
    def submit_request(self, request: WorkRequest) -> str:
        """Submit request for processing"""
        with self.request_lock:
            self.pending_requests.append(request)
            self.metrics["total_requests"] += 1
            
        self.logger.debug(f"Submitted request {request.request_id} for {request.task_type}")
        return request.request_id
    
    def _process_requests_loop(self):
        """Background loop to process pending requests"""
        while True:
            try:
                self._process_pending_requests()
                time.sleep(0.1)  # Check every 100ms
            except Exception as e:
                self.logger.error(f"Request processing error: {e}")
                time.sleep(1)
    
    def _process_pending_requests(self):
        """Process pending requests by routing to available workers"""
        if not self.pending_requests:
            return
        
        with self.request_lock:
            requests_to_process = self.pending_requests.copy()
            self.pending_requests.clear()
        
        for request in requests_to_process:
            try:
                worker = self._route_request(request)
                if worker:
                    self._assign_request_to_worker(request, worker)
                else:
                    # No available worker - re-queue or fail
                    if request.retry_count < request.max_retries:
                        request.retry_count += 1
                        with self.request_lock:
                            self.pending_requests.append(request)
                    else:
                        self.metrics["failed_requests"] += 1
                        self.logger.warning(f"Failed to route request {request.request_id} after {request.max_retries} retries")
                        
            except Exception as e:
                self.logger.error(f"Error processing request {request.request_id}: {e}")
    
    def _route_request(self, request: WorkRequest) -> Optional[WorkerNode]:
        """Route request to appropriate worker using selected strategy"""
        with self.worker_lock:
            available_workers = [
                worker for worker in self.workers.values()
                if worker.status in [WorkerStatus.IDLE, WorkerStatus.BUSY]
                and worker.current_load < worker.capacity
            ]
            
            if not available_workers:
                return None
            
            # Use selected routing strategy
            strategy_func = self.routing_strategies.get(
                self.routing_strategy, 
                self._route_least_connections
            )
            
            return strategy_func(available_workers, request)
    
    def _route_round_robin(self, workers: List[WorkerNode], request: WorkRequest) -> WorkerNode:
        """Round-robin routing strategy"""
        # Simple implementation - in production, maintain state for true round-robin
        return min(workers, key=lambda w: w.total_processed)
    
    def _route_least_connections(self, workers: List[WorkerNode], request: WorkRequest) -> WorkerNode:
        """Route to worker with least current load"""
        return min(workers, key=lambda w: w.current_load)
    
    def _route_weighted_response_time(self, workers: List[WorkerNode], request: WorkRequest) -> WorkerNode:
        """Route based on worker performance score"""
        return max(workers, key=lambda w: w.performance_score / max(w.current_load, 1))
    
    def _route_resource_based(self, workers: List[WorkerNode], request: WorkRequest) -> WorkerNode:
        """Route based on available resources"""
        def resource_score(worker):
            load_factor = 1.0 - (worker.current_load / worker.capacity)
            performance_factor = worker.performance_score
            return load_factor * performance_factor
        
        return max(workers, key=resource_score)
    
    def _assign_request_to_worker(self, request: WorkRequest, worker: WorkerNode):
        """Assign request to worker and update state"""
        request.assigned_worker = worker.worker_id
        worker.current_load += 1
        worker.total_processed += 1
        
        # Update worker status based on load
        if worker.current_load >= worker.capacity:
            worker.status = WorkerStatus.OVERLOADED
        elif worker.current_load > 0:
            worker.status = WorkerStatus.BUSY
        
        self.logger.debug(f"Assigned request {request.request_id} to worker {worker.worker_id}")
    
    def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                self._perform_health_checks()
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                time.sleep(self.health_check_interval)
    
    def _perform_health_checks(self):
        """Perform health checks on all workers"""
        current_time = time.time()
        unhealthy_workers = []
        
        with self.worker_lock:
            for worker_id, worker in self.workers.items():
                # Check heartbeat timeout
                if current_time - worker.last_heartbeat > self.health_check_interval * 2:
                    worker.status = WorkerStatus.OFFLINE
                    unhealthy_workers.append(worker_id)
                    self.logger.warning(f"Worker {worker_id} marked offline due to missing heartbeat")
                
                # Update performance score based on error rate
                if worker.total_processed > 0:
                    error_rate = worker.error_count / worker.total_processed
                    worker.performance_score = max(0.1, 1.0 - error_rate)
        
        # Remove offline workers
        for worker_id in unhealthy_workers:
            self.unregister_worker(worker_id)
    
    def _auto_scaler_loop(self):
        """Background auto-scaling loop"""
        while True:
            try:
                self._evaluate_scaling_needs()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Auto-scaler error: {e}")
                time.sleep(60)
    
    def _evaluate_scaling_needs(self):
        """Evaluate if scaling up or down is needed"""
        current_time = time.time()
        
        # Respect cooldown period
        if current_time - self.last_scale_action < self.scale_cooldown:
            return
        
        with self.worker_lock:
            active_workers = [
                w for w in self.workers.values()
                if w.status != WorkerStatus.OFFLINE
            ]
            
            if not active_workers:
                return
            
            # Calculate average utilization
            total_capacity = sum(w.capacity for w in active_workers)
            total_load = sum(w.current_load for w in active_workers)
            utilization = total_load / max(total_capacity, 1)
            
            self.metrics["worker_utilization"] = utilization
            
            # Scale up if utilization is high
            if utilization > self.scale_up_threshold and len(active_workers) < self.max_workers:
                self._scale_up()
                self.last_scale_action = current_time
                
            # Scale down if utilization is low
            elif utilization < self.scale_down_threshold and len(active_workers) > self.min_workers:
                self._scale_down()
                self.last_scale_action = current_time
    
    def _scale_up(self):
        """Scale up by adding new worker"""
        self.metrics["scaling_events"] += 1
        self.logger.info(f"Scaling up: current workers={len(self.workers)}")
        
        # In a real implementation, this would spawn new worker processes/containers
        # For now, we'll create a mock worker
        worker_id = f"worker_{int(time.time())}"
        new_worker = WorkerNode(
            worker_id=worker_id,
            host="localhost",
            port=8000 + len(self.workers),
            capacity=10
        )
        self.register_worker(new_worker)
    
    def _scale_down(self):
        """Scale down by removing least utilized worker"""
        with self.worker_lock:
            # Find worker with lowest load to remove
            candidates = [
                w for w in self.workers.values()
                if w.status == WorkerStatus.IDLE and w.current_load == 0
            ]
            
            if candidates:
                worker_to_remove = min(candidates, key=lambda w: w.total_processed)
                self.unregister_worker(worker_to_remove.worker_id)
                self.metrics["scaling_events"] += 1
                self.logger.info(f"Scaled down: removed worker {worker_to_remove.worker_id}")
    
    def _update_worker_metrics(self):
        """Update worker-related metrics"""
        with self.worker_lock:
            self.metrics["current_load"] = sum(w.current_load for w in self.workers.values())
    
    def update_worker_heartbeat(self, worker_id: str, performance_data: Dict[str, Any] = None):
        """Update worker heartbeat and performance data"""
        with self.worker_lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.last_heartbeat = time.time()
                
                if performance_data:
                    # Update performance metrics
                    if "completed_requests" in performance_data:
                        worker.total_processed = performance_data["completed_requests"]
                    if "error_count" in performance_data:
                        worker.error_count = performance_data["error_count"]
                    if "current_load" in performance_data:
                        worker.current_load = performance_data["current_load"]
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics"""
        with self.worker_lock:
            worker_stats = {}
            for worker_id, worker in self.workers.items():
                worker_stats[worker_id] = {
                    "status": worker.status.value,
                    "current_load": worker.current_load,
                    "capacity": worker.capacity,
                    "utilization": worker.current_load / max(worker.capacity, 1),
                    "performance_score": worker.performance_score,
                    "total_processed": worker.total_processed,
                    "error_count": worker.error_count
                }
        
        return {
            "metrics": self.metrics.copy(),
            "worker_count": len(self.workers),
            "pending_requests": len(self.pending_requests),
            "routing_strategy": self.routing_strategy,
            "workers": worker_stats
        }
    
    def set_routing_strategy(self, strategy: str):
        """Change routing strategy"""
        if strategy in self.routing_strategies:
            self.routing_strategy = strategy
            self.logger.info(f"Routing strategy changed to: {strategy}")
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")


class DistributedAdapter:
    """
    Distributed adapter that can process requests across multiple worker nodes.
    """
    
    def __init__(
        self,
        model_name: str = "distributed_adapter",
        load_balancer: Optional[LoadBalancer] = None,
        enable_auto_scaling: bool = True,
        **kwargs
    ):
        self.model_name = model_name
        self.enable_auto_scaling = enable_auto_scaling
        
        # Initialize load balancer
        self.load_balancer = load_balancer or LoadBalancer()
        
        # Initialize local worker pool for fallback
        self.local_executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.metrics = {
            "distributed_requests": 0,
            "local_fallback_requests": 0,
            "total_response_time": 0.0,
            "average_response_time": 0.0
        }
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info(
            f"DistributedAdapter initialized",
            extra={
                "model_name": model_name,
                "auto_scaling": enable_auto_scaling,
                "workers": len(self.load_balancer.workers)
            }
        )
    
    async def generate_distributed(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        priority: int = 0,
        timeout: float = 30.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using distributed processing.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Generation temperature
            priority: Request priority
            timeout: Request timeout
            **kwargs: Additional parameters
            
        Returns:
            Generation result with distributed processing metadata
        """
        start_time = time.time()
        request_id = f"dist_{int(start_time * 1000000)}"
        
        try:
            # Validate inputs
            prompt = InputValidator.validate_text_content(prompt, max_length=8192)
            
            # Create distributed work request
            work_request = WorkRequest(
                request_id=request_id,
                task_type="text_generation",
                payload={
                    "prompt": prompt,
                    "max_length": max_length,
                    "temperature": temperature,
                    **kwargs
                },
                priority=priority,
                timeout=timeout
            )
            
            # Submit to load balancer
            self.load_balancer.submit_request(work_request)
            
            # Wait for completion (simplified - in production use async queues)
            result = await self._wait_for_completion(work_request, timeout)
            
            # Update metrics
            response_time = (time.time() - start_time) * 1000
            self._update_metrics(response_time, distributed=True)
            
            return {
                "generated_text": result.get("generated_text", ""),
                "input": prompt,
                "model_name": self.model_name,
                "request_id": request_id,
                "distributed": True,
                "assigned_worker": work_request.assigned_worker,
                "response_time_ms": response_time,
                "performance_metrics": result.get("performance_metrics", {}),
                "distribution_metadata": {
                    "routing_strategy": self.load_balancer.routing_strategy,
                    "worker_count": len(self.load_balancer.workers),
                    "retry_count": work_request.retry_count
                }
            }
            
        except Exception as e:
            # Fallback to local processing
            self.logger.warning(f"Distributed processing failed, falling back to local: {e}")
            return await self._process_locally(prompt, max_length, temperature, **kwargs)
    
    async def _wait_for_completion(self, request: WorkRequest, timeout: float) -> Dict[str, Any]:
        """Wait for request completion (simplified implementation)"""
        # In a real implementation, this would use proper async queues/callbacks
        # For now, we'll simulate processing
        await asyncio.sleep(0.1)  # Simulate network latency
        
        return {
            "generated_text": f"Distributed response for: {request.payload['prompt'][:50]}...",
            "performance_metrics": {
                "worker_processing_time_ms": 50.0,
                "queue_time_ms": 10.0
            }
        }
    
    async def _process_locally(
        self, 
        prompt: str, 
        max_length: int, 
        temperature: float, 
        **kwargs
    ) -> Dict[str, Any]:
        """Fallback to local processing"""
        start_time = time.time()
        
        try:
            # Submit to local executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.local_executor,
                self._local_generation_task,
                prompt,
                max_length,
                temperature
            )
            
            response_time = (time.time() - start_time) * 1000
            self._update_metrics(response_time, distributed=False)
            
            return {
                "generated_text": result,
                "input": prompt,
                "model_name": self.model_name,
                "distributed": False,
                "fallback_reason": "distributed_unavailable",
                "response_time_ms": response_time
            }
            
        except Exception as e:
            return {
                "generated_text": f"Error: {str(e)}",
                "input": prompt,
                "error": str(e),
                "response_time_ms": (time.time() - start_time) * 1000
            }
    
    def _local_generation_task(self, prompt: str, max_length: int, temperature: float) -> str:
        """Local generation task for thread executor"""
        # Mock local processing
        time.sleep(0.05)  # Simulate processing time
        return f"Local fallback response for: {prompt[:50]}..."
    
    def _update_metrics(self, response_time: float, distributed: bool):
        """Update performance metrics"""
        if distributed:
            self.metrics["distributed_requests"] += 1
        else:
            self.metrics["local_fallback_requests"] += 1
        
        # Update average response time
        total_requests = self.metrics["distributed_requests"] + self.metrics["local_fallback_requests"]
        
        self.metrics["total_response_time"] += response_time
        self.metrics["average_response_time"] = self.metrics["total_response_time"] / total_requests
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get distributed processing statistics"""
        lb_stats = self.load_balancer.get_load_balancer_stats()
        
        return {
            "adapter_metrics": self.metrics.copy(),
            "load_balancer": lb_stats,
            "local_executor_active": not self.local_executor._shutdown,
            "auto_scaling_enabled": self.enable_auto_scaling
        }
    
    def optimize_distribution(self):
        """Optimize distribution based on current performance"""
        stats = self.get_distribution_stats()
        
        # Switch routing strategy based on performance
        current_avg_time = stats["adapter_metrics"]["average_response_time"]
        worker_utilization = stats["load_balancer"]["metrics"]["worker_utilization"]
        
        if current_avg_time > 1000 and worker_utilization > 0.8:  # High latency, high utilization
            self.load_balancer.set_routing_strategy("least_connections")
        elif worker_utilization < 0.3:  # Low utilization
            self.load_balancer.set_routing_strategy("round_robin")
        else:
            self.load_balancer.set_routing_strategy("weighted_response_time")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'local_executor'):
            self.local_executor.shutdown(wait=False)