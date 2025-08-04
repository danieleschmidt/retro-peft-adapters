"""
Asynchronous pipeline for high-throughput retrieval-augmented inference.

Provides async/await support for I/O operations, concurrent processing,
and efficient resource utilization for production deployments.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Union, AsyncIterator, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiofiles
from contextlib import asynccontextmanager
import numpy as np

from ..utils.logging import get_global_logger
from .cache import get_cache_manager, MultiLevelCache
from .resource_pool import get_resource_manager


@dataclass
class AsyncRequest:
    """Async request item"""
    id: str
    prompt: str
    retrieval_k: int = 5
    retrieval_enabled: bool = True
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    priority: int = 0  # Higher number = higher priority
    timeout: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)


@dataclass  
class AsyncResult:
    """Async processing result"""
    id: str
    prompt: str
    generated_text: str
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0
    queue_time: float = 0.0
    retrieval_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    completed_at: float = field(default_factory=time.time)


class AsyncRetrievalPipeline:
    """
    High-performance async retrieval-augmented inference pipeline.
    
    Features:
    - Full async/await support
    - Request prioritization and queuing
    - Intelligent caching and batching
    - Resource pooling and connection management
    - Graceful error handling and timeouts
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-small",
        max_concurrent_requests: int = 100,
        request_timeout: float = 30.0,
        batch_size: int = 16,
        enable_caching: bool = True,
        cache_manager: Optional[MultiLevelCache] = None
    ):
        """
        Initialize async pipeline.
        
        Args:
            model_name: Name of the language model to use
            max_concurrent_requests: Maximum concurrent requests
            request_timeout: Default request timeout in seconds
            batch_size: Batch size for processing
            enable_caching: Whether to enable caching
            cache_manager: Optional custom cache manager
        """
        self.model_name = model_name
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout
        self.batch_size = batch_size
        self.enable_caching = enable_caching
        
        # Components
        self.cache_manager = cache_manager or (get_cache_manager() if enable_caching else None)
        self.resource_manager = get_resource_manager()
        self.logger = get_global_logger()
        
        # Async components
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Processing state
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_processing_time": 0.0,
            "avg_queue_time": 0.0
        }
        
        # Background processing
        self._processor_task = None
        self._running = False
    
    async def start(self):
        """Start the async pipeline"""
        if self._running:
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_requests())
        self.logger.info("Async pipeline started")
    
    async def stop(self):
        """Stop the async pipeline"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel processor task
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel remaining processing tasks
        for task in list(self._processing_tasks.values()):
            task.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Async pipeline stopped")
    
    async def generate(
        self,
        prompt: Union[str, List[str]],
        retrieval_k: int = 5,
        retrieval_enabled: bool = True,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate response(s) asynchronously.
        
        Args:
            prompt: Input prompt(s)
            retrieval_k: Number of documents to retrieve
            retrieval_enabled: Whether to enable retrieval
            max_length: Maximum generation length
            temperature: Generation temperature
            timeout: Request timeout
            priority: Request priority (higher is more important)
            metadata: Optional metadata
            
        Returns:
            Generated response(s)
        """
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]
        
        # Create requests
        requests = []
        for i, p in enumerate(prompts):
            request = AsyncRequest(
                id=f"req_{int(time.time() * 1000000)}_{i}",
                prompt=p,
                retrieval_k=retrieval_k,
                retrieval_enabled=retrieval_enabled,
                max_length=max_length,
                temperature=temperature,
                priority=priority,
                timeout=timeout or self.request_timeout,
                metadata=metadata
            )
            requests.append(request)
        
        # Process requests
        results = await self._process_requests_batch(requests)
        
        return results if is_batch else results[0]
    
    async def _process_requests_batch(
        self, 
        requests: List[AsyncRequest]
    ) -> List[Dict[str, Any]]:
        """Process a batch of requests"""
        # Submit to queue
        futures = []
        for request in requests:
            future = asyncio.Future()
            await self.request_queue.put((
                -request.priority,  # Negative for max-heap behavior
                request.created_at,
                request,
                future
            ))
            futures.append(future)
        
        # Wait for results
        results = []
        for future in futures:
            try:
                result = await future
                results.append(self._result_to_dict(result))
            except Exception as e:
                self.logger.error(f"Request failed: {e}")
                results.append({
                    "generated_text": "",
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    async def _process_requests(self):
        """Background task to process requests from queue"""
        batch_requests = []
        batch_futures = []
        
        while self._running:
            try:
                # Collect batch
                batch_timeout = 0.1  # 100ms batch collection timeout
                batch_start = time.time()
                
                while (
                    len(batch_requests) < self.batch_size and
                    (time.time() - batch_start) < batch_timeout
                ):
                    try:
                        priority, created_at, request, future = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=batch_timeout - (time.time() - batch_start)
                        )
                        
                        batch_requests.append(request)
                        batch_futures.append(future)
                        
                    except asyncio.TimeoutError:
                        break
                
                # Process batch if we have requests
                if batch_requests:
                    await self._process_batch(batch_requests, batch_futures)
                    batch_requests.clear()
                    batch_futures.clear()
                
                # Small delay to prevent busy waiting
                if not batch_requests:
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"Error in request processor: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(
        self,
        requests: List[AsyncRequest],
        futures: List[asyncio.Future]
    ):
        """Process a batch of requests"""
        batch_start = time.time()
        
        try:
            # Check cache for each request
            cached_results = {}
            uncached_requests = []
            uncached_futures = []
            
            for request, future in zip(requests, futures):
                if self.enable_caching:
                    cache_key = self._get_cache_key(request)
                    cached_result = self.cache_manager.get(cache_key)
                    
                    if cached_result is not None:
                        result = AsyncResult(
                            id=request.id,
                            prompt=request.prompt,
                            generated_text=cached_result["generated_text"],
                            success=True,
                            processing_time=0.0,
                            queue_time=batch_start - request.created_at,
                            retrieval_info=cached_result.get("retrieval_info"),
                            metadata=request.metadata
                        )
                        
                        future.set_result(result)
                        self._stats["cache_hits"] += 1
                        continue
                
                uncached_requests.append(request)
                uncached_futures.append(future)
                self._stats["cache_misses"] += 1
            
            # Process uncached requests
            if uncached_requests:
                await self._process_uncached_batch(
                    uncached_requests, 
                    uncached_futures,
                    batch_start
                )
            
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            
            # Set error for all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)
    
    async def _process_uncached_batch(
        self,
        requests: List[AsyncRequest],
        futures: List[asyncio.Future],
        batch_start: float
    ):
        """Process uncached requests"""
        async with self.semaphore:
            try:
                # Get model instance from pool
                async with self.resource_manager.get_model(self.model_name) as model:
                    # Process requests
                    for request, future in zip(requests, futures):
                        try:
                            # Check timeout
                            elapsed = time.time() - request.created_at
                            if elapsed > request.timeout:
                                raise asyncio.TimeoutError("Request timeout")
                            
                            # Generate response
                            start_time = time.time()
                            response = await self._generate_single(model, request)
                            processing_time = time.time() - start_time
                            
                            # Create result
                            result = AsyncResult(
                                id=request.id,
                                prompt=request.prompt,
                                generated_text=response["generated_text"],
                                success=True,
                                processing_time=processing_time,
                                queue_time=batch_start - request.created_at,
                                retrieval_info=response.get("retrieval_info"),
                                metadata=request.metadata
                            )
                            
                            # Cache result
                            if self.enable_caching:
                                cache_key = self._get_cache_key(request)
                                self.cache_manager.put(cache_key, {
                                    "generated_text": response["generated_text"],
                                    "retrieval_info": response.get("retrieval_info")
                                }, ttl=3600)  # 1 hour TTL
                            
                            future.set_result(result)
                            self._stats["successful_requests"] += 1
                            
                        except Exception as e:
                            self.logger.error(f"Request {request.id} failed: {e}")
                            
                            result = AsyncResult(
                                id=request.id,
                                prompt=request.prompt,
                                generated_text="",
                                success=False,
                                error=str(e),
                                processing_time=0.0,
                                queue_time=batch_start - request.created_at,
                                metadata=request.metadata
                            )
                            
                            future.set_result(result)
                            self._stats["failed_requests"] += 1
                            
            except Exception as e:
                self.logger.error(f"Model processing error: {e}")
                
                # Set error for all futures
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
    
    async def _generate_single(
        self, 
        model: Any, 
        request: AsyncRequest
    ) -> Dict[str, Any]:
        """Generate response for single request"""
        # This would be implemented based on the actual model interface
        # For now, return a mock response
        
        # Simulate retrieval if enabled
        retrieval_info = None
        if request.retrieval_enabled:
            retrieval_info = {
                "retrieved_documents": f"Retrieved {request.retrieval_k} documents",
                "retrieval_time": 0.05
            }
        
        # Simulate generation
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "generated_text": f"Generated response for: {request.prompt[:50]}...",
            "retrieval_info": retrieval_info
        }
    
    def _get_cache_key(self, request: AsyncRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            "prompt": request.prompt,
            "retrieval_k": request.retrieval_k if request.retrieval_enabled else 0,
            "retrieval_enabled": request.retrieval_enabled,
            "max_length": request.max_length,
            "temperature": request.temperature,
            "model": self.model_name
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        import hashlib
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _result_to_dict(self, result: AsyncResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "id": result.id,
            "generated_text": result.generated_text,
            "success": result.success,
            "error": result.error,
            "processing_time": result.processing_time,
            "queue_time": result.queue_time,
            "retrieval_info": result.retrieval_info,
            "metadata": result.metadata
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            **self._stats,
            "queue_size": self.request_queue.qsize(),
            "active_tasks": len(self._processing_tasks),
            "cache_stats": self.cache_manager.get_stats() if self.cache_manager else None
        }


class AsyncBatchProcessor:
    """
    Async batch processor for large-scale inference.
    """
    
    def __init__(
        self,
        pipeline: AsyncRetrievalPipeline,
        max_concurrent_batches: int = 10,
        progress_callback: Optional[Callable] = None
    ):
        """
        Initialize async batch processor.
        
        Args:
            pipeline: Async pipeline to use
            max_concurrent_batches: Maximum concurrent batches
            progress_callback: Optional progress callback
        """
        self.pipeline = pipeline
        self.max_concurrent_batches = max_concurrent_batches
        self.progress_callback = progress_callback
        self.logger = get_global_logger()
    
    async def process_batch(
        self,
        requests: List[Union[AsyncRequest, Dict[str, Any]]],
        batch_size: int = 16
    ) -> List[AsyncResult]:
        """
        Process batch of requests asynchronously.
        
        Args:
            requests: List of requests to process
            batch_size: Size of processing batches
            
        Returns:
            List of results
        """
        # Convert dict requests to AsyncRequest objects
        async_requests = []
        for req in requests:
            if isinstance(req, dict):
                async_requests.append(AsyncRequest(**req))
            else:
                async_requests.append(req)
        
        # Start pipeline if not running
        if not self.pipeline._running:
            await self.pipeline.start()
        
        # Process in chunks
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        async def process_chunk(chunk):
            async with semaphore:
                chunk_results = await self.pipeline._process_requests_batch(chunk)
                
                # Convert to AsyncResult objects
                async_results = []
                for i, result_dict in enumerate(chunk_results):
                    request = chunk[i]
                    async_result = AsyncResult(
                        id=request.id,
                        prompt=request.prompt,
                        generated_text=result_dict["generated_text"],
                        success=result_dict["success"],
                        error=result_dict.get("error"),
                        metadata=request.metadata
                    )
                    async_results.append(async_result)
                
                return async_results
        
        # Create tasks for all chunks
        tasks = []
        for i in range(0, len(async_requests), batch_size):
            chunk = async_requests[i:i + batch_size]
            task = asyncio.create_task(process_chunk(chunk))
            tasks.append(task)
        
        # Process with progress tracking
        completed = 0
        total = len(tasks)
        
        for task in asyncio.as_completed(tasks):
            chunk_results = await task
            results.extend(chunk_results)
            
            completed += 1
            if self.progress_callback:
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.progress_callback, 
                    completed, 
                    total
                )
        
        return results
    
    async def process_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        input_format: str = "jsonl",
        batch_size: int = 16,
        **kwargs
    ) -> List[AsyncResult]:
        """
        Process requests from file asynchronously.
        
        Args:
            input_file: Input file path
            output_file: Optional output file path
            input_format: Input file format
            batch_size: Processing batch size
            **kwargs: Additional arguments
            
        Returns:
            List of results
        """
        # Load requests
        requests = await self._load_requests_async(input_file, input_format)
        
        self.logger.info(f"Loaded {len(requests)} requests from {input_file}")
        
        # Process requests
        results = await self.process_batch(requests, batch_size)
        
        # Save results if output file specified
        if output_file:
            await self._save_results_async(results, output_file)
            self.logger.info(f"Results saved to {output_file}")
        
        return results
    
    async def _load_requests_async(
        self,
        input_file: str,
        input_format: str
    ) -> List[AsyncRequest]:
        """Load requests from file asynchronously"""
        requests = []
        
        if input_format == "jsonl":
            async with aiofiles.open(input_file, 'r') as f:
                line_num = 0
                async for line in f:
                    try:
                        data = json.loads(line.strip())
                        request = AsyncRequest(
                            id=data.get("id", f"req_{line_num}"),
                            prompt=data.get("prompt", ""),
                            retrieval_k=data.get("retrieval_k", 5),
                            retrieval_enabled=data.get("retrieval_enabled", True),
                            max_length=data.get("max_length"),
                            temperature=data.get("temperature"),
                            metadata=data.get("metadata")
                        )
                        requests.append(request)
                        line_num += 1
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing line {line_num}: {e}")
        
        elif input_format == "json":
            async with aiofiles.open(input_file, 'r') as f:
                content = await f.read()
                data_list = json.loads(content)
                
                for i, data in enumerate(data_list):
                    request = AsyncRequest(
                        id=data.get("id", f"req_{i}"),
                        prompt=data.get("prompt", ""),
                        retrieval_k=data.get("retrieval_k", 5),
                        retrieval_enabled=data.get("retrieval_enabled", True),
                        max_length=data.get("max_length"),
                        temperature=data.get("temperature"),
                        metadata=data.get("metadata")
                    )
                    requests.append(request)
        
        return requests
    
    async def _save_results_async(
        self,
        results: List[AsyncResult],
        output_file: str
    ):
        """Save results to file asynchronously"""
        output_data = []
        for result in results:
            result_dict = {
                "id": result.id,
                "prompt": result.prompt,
                "generated_text": result.generated_text,
                "success": result.success,
                "error": result.error,
                "processing_time": result.processing_time,
                "queue_time": result.queue_time,
                "retrieval_info": result.retrieval_info,
                "metadata": result.metadata
            }
            output_data.append(result_dict)
        
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(output_data, indent=2))


# Context manager for pipeline lifecycle
@asynccontextmanager
async def async_pipeline(
    model_name: str = "microsoft/DialoGPT-small",
    **kwargs
) -> AsyncRetrievalPipeline:
    """Context manager for async pipeline lifecycle"""
    pipeline = AsyncRetrievalPipeline(model_name=model_name, **kwargs)
    
    try:
        await pipeline.start()
        yield pipeline
    finally:
        await pipeline.stop()