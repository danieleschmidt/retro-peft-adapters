"""
Asynchronous processing capabilities for retro-peft components.

Provides async adapters, batch processing, and concurrent retrieval.
"""

import asyncio
import concurrent.futures
import logging
import time
from typing import Any, Dict, List, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass


@dataclass
class BatchRequest:
    """Represents a batch processing request"""
    id: str
    inputs: List[Any]
    metadata: Dict[str, Any]
    timestamp: float


@dataclass
class BatchResponse:
    """Represents a batch processing response"""
    request_id: str
    outputs: List[Any]
    errors: List[Optional[Exception]]
    processing_time: float
    metadata: Dict[str, Any]


class AsyncBatchProcessor:
    """
    High-performance async batch processor for adapters.
    
    Supports concurrent processing, dynamic batching, and resource management.
    """
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_queue_size: int = 1000,
        batch_timeout: float = 0.1,
        max_concurrent_batches: int = 4,
        use_process_pool: bool = False
    ):
        """
        Initialize async batch processor.
        
        Args:
            max_batch_size: Maximum items per batch
            max_queue_size: Maximum queue size before blocking
            batch_timeout: Maximum time to wait for batch to fill
            max_concurrent_batches: Max concurrent batch processing
            use_process_pool: Use process pool instead of thread pool
        """
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.batch_timeout = batch_timeout
        self.max_concurrent_batches = max_concurrent_batches
        
        # Processing queue
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing_futures = set()
        
        # Executor for CPU-intensive work
        if use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=max_concurrent_batches)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_concurrent_batches)
        
        # Statistics
        self.processed_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        
        # Control flags
        self.running = False
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the batch processor"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting async batch processor")
        
        # Start batch collection and processing
        asyncio.create_task(self._batch_collector())
    
    async def stop(self):
        """Stop the batch processor and cleanup"""
        self.running = False
        
        # Wait for pending batches to complete
        if self.processing_futures:
            await asyncio.gather(*self.processing_futures, return_exceptions=True)
        
        # Cleanup executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Async batch processor stopped")
    
    async def submit(
        self,
        processor_func: Callable,
        inputs: List[Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Submit a batch processing request.
        
        Args:
            processor_func: Function to process the batch
            inputs: List of inputs to process
            metadata: Optional metadata for the batch
            
        Returns:
            List of processed outputs
        """
        if not self.running:
            await self.start()
        
        # Create batch request
        request_id = f"batch_{int(time.time() * 1000000)}"
        request = BatchRequest(
            id=request_id,
            inputs=inputs,
            metadata=metadata or {},
            timestamp=time.time()
        )
        
        # Create response future
        response_future = asyncio.Future()
        
        # Add to processing queue
        await self.queue.put((request, processor_func, response_future))
        
        # Wait for response
        response = await response_future
        return response.outputs
    
    async def _batch_collector(self):
        """Collect requests into batches and submit for processing"""
        while self.running:
            try:
                batch_items = []
                
                # Collect items for batch
                deadline = time.time() + self.batch_timeout
                
                while (len(batch_items) < self.max_batch_size and 
                       time.time() < deadline and 
                       self.running):
                    try:
                        timeout = deadline - time.time()
                        if timeout <= 0:
                            break
                        
                        item = await asyncio.wait_for(
                            self.queue.get(), timeout=timeout
                        )
                        batch_items.append(item)
                        
                    except asyncio.TimeoutError:
                        break
                
                # Process batch if we have items
                if batch_items:
                    # Check if we're at concurrent batch limit
                    if len(self.processing_futures) >= self.max_concurrent_batches:
                        # Wait for at least one batch to complete
                        done, pending = await asyncio.wait(
                            self.processing_futures,
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        self.processing_futures = pending
                    
                    # Start batch processing
                    future = asyncio.create_task(
                        self._process_batch(batch_items)
                    )
                    self.processing_futures.add(future)
            
            except Exception as e:
                self.logger.error(f"Error in batch collector: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, batch_items: List[tuple]):
        """Process a batch of items"""
        start_time = time.time()
        
        try:
            # Group items by processor function
            processor_groups = {}
            for request, processor_func, response_future in batch_items:
                func_key = id(processor_func)
                if func_key not in processor_groups:
                    processor_groups[func_key] = {
                        'func': processor_func,
                        'items': []
                    }
                processor_groups[func_key]['items'].append(
                    (request, response_future)
                )
            
            # Process each group
            for group_data in processor_groups.values():
                processor_func = group_data['func']
                items = group_data['items']
                
                # Extract inputs for batch processing
                requests = [item[0] for item in items]
                response_futures = [item[1] for item in items]
                
                # Combine all inputs
                all_inputs = []
                input_boundaries = []
                
                for request in requests:
                    input_boundaries.append(len(all_inputs))
                    all_inputs.extend(request.inputs)
                input_boundaries.append(len(all_inputs))
                
                # Process batch
                try:
                    batch_outputs = await self._execute_processor(
                        processor_func, all_inputs
                    )
                    
                    # Split outputs back to individual requests
                    for i, (request, response_future) in enumerate(items):
                        start_idx = input_boundaries[i]
                        end_idx = input_boundaries[i + 1]
                        
                        request_outputs = batch_outputs[start_idx:end_idx]
                        
                        response = BatchResponse(
                            request_id=request.id,
                            outputs=request_outputs,
                            errors=[],
                            processing_time=time.time() - start_time,
                            metadata=request.metadata
                        )
                        
                        response_future.set_result(response)
                        self.processed_count += 1
                
                except Exception as e:
                    # Handle batch processing error
                    self.error_count += 1
                    self.logger.error(f"Batch processing error: {e}")
                    
                    for _, response_future in items:
                        if not response_future.done():
                            response_future.set_exception(e)
        
        except Exception as e:
            self.logger.error(f"Fatal batch processing error: {e}")
            
            # Set exceptions for all pending futures
            for _, _, response_future in batch_items:
                if not response_future.done():
                    response_future.set_exception(e)
        
        finally:
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
    
    async def _execute_processor(self, processor_func: Callable, inputs: List[Any]) -> List[Any]:
        """Execute processor function with inputs"""
        loop = asyncio.get_event_loop()
        
        # Run in executor to avoid blocking
        outputs = await loop.run_in_executor(
            self.executor, processor_func, inputs
        )
        
        return outputs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": (
                self.total_processing_time / max(self.processed_count, 1)
            ),
            "queue_size": self.queue.qsize(),
            "active_batches": len(self.processing_futures),
            "success_rate": (
                (self.processed_count - self.error_count) / 
                max(self.processed_count, 1)
            )
        }


class AsyncRetriever:
    """
    Asynchronous retrieval component with concurrent search capabilities.
    """
    
    def __init__(
        self, 
        base_retriever, 
        max_concurrent_searches: int = 10,
        cache_size: int = 1000
    ):
        """
        Initialize async retriever.
        
        Args:
            base_retriever: Base retriever implementation
            max_concurrent_searches: Max concurrent search operations
            cache_size: Size of search result cache
        """
        self.base_retriever = base_retriever
        self.max_concurrent_searches = max_concurrent_searches
        self.semaphore = asyncio.Semaphore(max_concurrent_searches)
        
        # Simple LRU cache for search results
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size
        
        self.logger = logging.getLogger(__name__)
    
    async def search_async(
        self, 
        query: str, 
        k: int = 5,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Asynchronous search with concurrent execution support.
        
        Args:
            query: Search query
            k: Number of results to return
            use_cache: Whether to use cached results
            
        Returns:
            List of search results
        """
        cache_key = f"{query}:{k}" if use_cache else None
        
        # Check cache first
        if cache_key and cache_key in self.cache:
            self.logger.debug(f"Cache hit for query: {query[:50]}...")
            return self.cache[cache_key]
        
        # Acquire semaphore to limit concurrency
        async with self.semaphore:
            try:
                # Run search in thread executor
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None, self.base_retriever.search, query, k
                )
                
                # Update cache
                if cache_key:
                    self._update_cache(cache_key, results)
                
                return results
                
            except Exception as e:
                self.logger.error(f"Async search error: {e}")
                raise
    
    async def batch_search_async(
        self, 
        queries: List[str], 
        k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch search with concurrent execution.
        
        Args:
            queries: List of search queries
            k: Number of results per query
            
        Returns:
            List of search result lists
        """
        tasks = [
            self.search_async(query, k) for query in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        clean_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch search error: {result}")
                clean_results.append([])  # Empty results for failed queries
            else:
                clean_results.append(result)
        
        return clean_results
    
    def _update_cache(self, key: str, value: List[Dict[str, Any]]):
        """Update LRU cache"""
        if key in self.cache:
            # Move to end
            self.cache_order.remove(key)
        elif len(self.cache) >= self.cache_size:
            # Remove oldest item
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.cache_order.append(key)
    
    def clear_cache(self):
        """Clear search cache"""
        self.cache.clear()
        self.cache_order.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "cache_utilization": len(self.cache) / self.cache_size
        }


class ConcurrentAdapterPool:
    """
    Pool of adapters for concurrent processing with load balancing.
    """
    
    def __init__(
        self,
        adapter_factory: Callable,
        pool_size: int = 4,
        max_queue_size: int = 100
    ):
        """
        Initialize adapter pool.
        
        Args:
            adapter_factory: Function that creates new adapter instances
            pool_size: Number of adapters in pool
            max_queue_size: Maximum request queue size
        """
        self.adapter_factory = adapter_factory
        self.pool_size = pool_size
        self.max_queue_size = max_queue_size
        
        # Create adapter pool
        self.adapters = []
        self.adapter_locks = []
        
        for i in range(pool_size):
            adapter = adapter_factory()
            self.adapters.append(adapter)
            self.adapter_locks.append(asyncio.Lock())
        
        # Load balancing
        self.current_adapter = 0
        self.adapter_usage = [0] * pool_size
        
        # Request queue
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.running = False
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the adapter pool"""
        if self.running:
            return
        
        self.running = True
        self.logger.info(f"Starting adapter pool with {self.pool_size} adapters")
        
        # Start request processing
        for i in range(self.pool_size):
            asyncio.create_task(self._process_requests(i))
    
    async def stop(self):
        """Stop the adapter pool"""
        self.running = False
        self.logger.info("Stopping adapter pool")
    
    async def submit_request(
        self, 
        method_name: str, 
        *args, 
        **kwargs
    ) -> Any:
        """
        Submit a request to the adapter pool.
        
        Args:
            method_name: Name of adapter method to call
            *args: Method arguments
            **kwargs: Method keyword arguments
            
        Returns:
            Method result
        """
        if not self.running:
            await self.start()
        
        # Create request future
        request_future = asyncio.Future()
        
        # Add to queue
        await self.request_queue.put((method_name, args, kwargs, request_future))
        
        # Wait for result
        return await request_future
    
    async def _process_requests(self, adapter_index: int):
        """Process requests for a specific adapter"""
        adapter = self.adapters[adapter_index]
        adapter_lock = self.adapter_locks[adapter_index]
        
        while self.running:
            try:
                # Get request from queue
                method_name, args, kwargs, request_future = await self.request_queue.get()
                
                # Process request with adapter lock
                async with adapter_lock:
                    try:
                        # Get method from adapter
                        method = getattr(adapter, method_name)
                        
                        # Execute method
                        if asyncio.iscoroutinefunction(method):
                            result = await method(*args, **kwargs)
                        else:
                            # Run in executor for sync methods
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                None, lambda: method(*args, **kwargs)
                            )
                        
                        request_future.set_result(result)
                        self.adapter_usage[adapter_index] += 1
                        
                    except Exception as e:
                        request_future.set_exception(e)
                        self.logger.error(
                            f"Adapter {adapter_index} processing error: {e}"
                        )
            
            except Exception as e:
                self.logger.error(f"Request processing error: {e}")
                await asyncio.sleep(0.1)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get adapter pool statistics"""
        return {
            "pool_size": self.pool_size,
            "queue_size": self.request_queue.qsize(),
            "adapter_usage": self.adapter_usage,
            "total_requests": sum(self.adapter_usage),
            "avg_requests_per_adapter": sum(self.adapter_usage) / self.pool_size,
            "load_balance_ratio": (
                max(self.adapter_usage) / max(min(self.adapter_usage), 1)
            )
        }