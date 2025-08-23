#!/usr/bin/env python3
"""
Generation 3 Test Suite - Scalable and high-performance features.
"""

import sys
import asyncio
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.insert(0, 'src')

def test_high_performance_cache():
    """Test advanced multi-level caching system"""
    print("Testing high-performance cache...")
    
    try:
        from retro_peft.scaling.high_performance_adapters import HighPerformanceCache
        
        # Create cache with compression
        cache = HighPerformanceCache(
            max_size=100,
            ttl_seconds=60,
            compression_enabled=True
        )
        
        # Test basic cache operations
        cache.put("test_prompt", {"result": "test response"}, param1="value1")
        cached_result = cache.get("test_prompt", param1="value1")
        
        assert cached_result is not None
        assert cached_result["result"] == "test response"
        print("✓ Basic cache operations work")
        
        # Test cache miss
        miss_result = cache.get("nonexistent_prompt")
        assert miss_result is None
        print("✓ Cache miss handling works")
        
        # Test cache statistics
        stats = cache.get_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert "hit_rate" in stats
        print(f"✓ Cache stats: {stats['hit_rate']:.2f} hit rate")
        
        # Test L1/L2 cache levels
        # Fill cache to trigger L2 storage
        for i in range(150):
            cache.put(f"prompt_{i}", {"data": f"response_{i}"})
        
        stats_after_fill = cache.get_stats()
        print(f"✓ Multi-level cache: L1={stats_after_fill['l1_size']}, L2={stats_after_fill['l2_size']}")
        
        return True
    except Exception as e:
        print(f"✗ High-performance cache error: {e}")
        traceback.print_exc()
        return False

def test_batch_processor():
    """Test batch processing system"""
    print("\nTesting batch processor...")
    
    try:
        from retro_peft.scaling.high_performance_adapters import BatchProcessor, BatchRequest
        
        # Create batch processor
        processor = BatchProcessor(
            max_batch_size=8,
            max_wait_time=0.05,
            max_workers=2
        )
        
        # Test batch processing with async
        async def test_batch_processing():
            requests = []
            
            # Create multiple requests
            for i in range(10):
                request = BatchRequest(
                    request_id=f"batch_req_{i}",
                    prompt=f"Test prompt {i}",
                    priority=i % 3  # Mix priorities
                )
                requests.append(request)
            
            # Submit requests concurrently
            tasks = []
            for request in requests:
                task = asyncio.create_task(processor.submit_request(request))
                tasks.append(task)
            
            # Wait for all responses
            responses = await asyncio.gather(*tasks)
            
            assert len(responses) == 10
            for response in responses:
                assert response.generated_text is not None
                assert "Mock response" in response.generated_text
            
            return responses
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        responses = loop.run_until_complete(test_batch_processing())
        loop.close()
        
        print(f"✓ Batch processing: {len(responses)} requests processed")
        
        # Check metrics
        metrics = processor.get_metrics()
        assert metrics["batches_processed"] > 0
        assert metrics["requests_processed"] >= 10
        print(f"✓ Batch metrics: {metrics['requests_processed']} requests, {metrics['batches_processed']} batches")
        
        return True
    except Exception as e:
        print(f"✗ Batch processor error: {e}")
        traceback.print_exc()
        return False

def test_scalable_adapter():
    """Test scalable LoRA adapter with async processing"""
    print("\nTesting scalable adapter...")
    
    try:
        from retro_peft.scaling.high_performance_adapters import ScalableRetroLoRA
        from retro_peft.retrieval import MockRetriever
        
        # Create scalable adapter
        adapter = ScalableRetroLoRA(
            model_name="scalable_test",
            rank=8,
            cache_size=100,
            max_batch_size=4,
            enable_compression=True
        )
        
        # Connect retriever
        retriever = MockRetriever()
        adapter.set_retriever(retriever)
        print("✓ Scalable adapter created and retriever connected")
        
        # Test synchronous generation
        result_sync = adapter.generate(
            "What is machine learning?",
            max_length=150,
            temperature=0.7,
            retrieval_k=3
        )
        
        assert "generated_text" in result_sync
        assert len(result_sync["generated_text"]) > 0
        assert "response_time_ms" in result_sync
        print(f"✓ Sync generation: {result_sync['response_time_ms']:.1f}ms")
        
        # Test asynchronous generation
        async def test_async_generation():
            tasks = []
            prompts = [
                "Explain deep learning",
                "What is natural language processing?",
                "How do neural networks work?",
                "What are transformers in AI?"
            ]
            
            for i, prompt in enumerate(prompts):
                task = asyncio.create_task(
                    adapter.generate_async(
                        prompt,
                        max_length=100,
                        priority=i % 2,
                        retrieval_k=2
                    )
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async_results = loop.run_until_complete(test_async_generation())
        loop.close()
        
        assert len(async_results) == 4
        for result in async_results:
            assert "generated_text" in result
            assert result["response_time_ms"] > 0
        
        print(f"✓ Async generation: {len(async_results)} requests processed")
        
        # Test performance stats
        stats = adapter.get_performance_stats()
        assert "adapter_metrics" in stats
        assert "cache_performance" in stats
        assert "batch_performance" in stats
        
        cache_hit_rate = stats["cache_performance"]["hit_rate"]
        print(f"✓ Performance stats: {cache_hit_rate:.2f} cache hit rate")
        
        # Test performance optimization
        adapter.optimize_performance()
        print("✓ Performance optimization completed")
        
        return True
    except Exception as e:
        print(f"✗ Scalable adapter error: {e}")
        traceback.print_exc()
        return False

def test_load_balancer():
    """Test load balancing and worker management"""
    print("\nTesting load balancer...")
    
    try:
        from retro_peft.scaling.distributed_processing import (
            LoadBalancer, WorkerNode, WorkRequest, WorkerStatus
        )
        
        # Create load balancer
        lb = LoadBalancer(
            min_workers=2,
            max_workers=8,
            target_cpu_usage=0.7
        )
        
        # Register some workers
        workers = []
        for i in range(3):
            worker = WorkerNode(
                worker_id=f"worker_{i}",
                host=f"host_{i}",
                port=8000 + i,
                capacity=5
            )
            workers.append(worker)
            lb.register_worker(worker)
        
        print(f"✓ Load balancer created with {len(workers)} workers")
        
        # Test request submission
        requests = []
        for i in range(10):
            request = WorkRequest(
                request_id=f"req_{i}",
                task_type="test_task",
                payload={"data": f"test_{i}"},
                priority=i % 3
            )
            requests.append(request)
            lb.submit_request(request)
        
        print(f"✓ Submitted {len(requests)} requests")
        
        # Wait a bit for processing
        time.sleep(0.5)
        
        # Check load balancer stats
        stats = lb.get_load_balancer_stats()
        assert stats["worker_count"] == 3
        assert stats["metrics"]["total_requests"] >= 10
        
        print(f"✓ Load balancer stats: {stats['worker_count']} workers, {stats['metrics']['total_requests']} requests")
        
        # Test different routing strategies
        strategies = ["round_robin", "least_connections", "weighted_response_time", "resource_based"]
        
        for strategy in strategies:
            try:
                lb.set_routing_strategy(strategy)
                print(f"✓ Set routing strategy: {strategy}")
            except Exception as e:
                print(f"✗ Failed to set strategy {strategy}: {e}")
        
        # Test worker heartbeat updates
        for worker in workers:
            lb.update_worker_heartbeat(
                worker.worker_id,
                {"completed_requests": 5, "error_count": 0, "current_load": 2}
            )
        
        print("✓ Worker heartbeat updates completed")
        
        return True
    except Exception as e:
        print(f"✗ Load balancer error: {e}")
        traceback.print_exc()
        return False

def test_distributed_adapter():
    """Test distributed adapter with load balancing"""
    print("\nTesting distributed adapter...")
    
    try:
        from retro_peft.scaling.distributed_processing import (
            DistributedAdapter, LoadBalancer, WorkerNode
        )
        
        # Create load balancer with workers
        lb = LoadBalancer(min_workers=2, max_workers=4)
        
        # Add some mock workers
        for i in range(2):
            worker = WorkerNode(
                worker_id=f"dist_worker_{i}",
                host="localhost",
                port=9000 + i,
                capacity=10
            )
            lb.register_worker(worker)
        
        # Create distributed adapter
        adapter = DistributedAdapter(
            model_name="distributed_test",
            load_balancer=lb,
            enable_auto_scaling=True
        )
        
        print(f"✓ Distributed adapter created with {len(lb.workers)} workers")
        
        # Test distributed generation
        async def test_distributed_generation():
            tasks = []
            prompts = [
                "Generate text about AI",
                "Explain machine learning",
                "Describe neural networks"
            ]
            
            for prompt in prompts:
                task = asyncio.create_task(
                    adapter.generate_distributed(
                        prompt,
                        max_length=100,
                        priority=0,
                        timeout=5.0
                    )
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        # Run distributed test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        distributed_results = loop.run_until_complete(test_distributed_generation())
        loop.close()
        
        assert len(distributed_results) == 3
        for result in distributed_results:
            assert "generated_text" in result
            assert "distributed" in result
            
        print(f"✓ Distributed generation: {len(distributed_results)} requests processed")
        
        # Test distribution stats
        stats = adapter.get_distribution_stats()
        assert "adapter_metrics" in stats
        assert "load_balancer" in stats
        
        total_requests = (
            stats["adapter_metrics"]["distributed_requests"] + 
            stats["adapter_metrics"]["local_fallback_requests"]
        )
        print(f"✓ Distribution stats: {total_requests} total requests")
        
        # Test optimization
        adapter.optimize_distribution()
        print("✓ Distribution optimization completed")
        
        return True
    except Exception as e:
        print(f"✗ Distributed adapter error: {e}")
        traceback.print_exc()
        return False

def test_integration_scaling():
    """Test full integration with scaling features"""
    print("\nTesting integration scaling...")
    
    try:
        from retro_peft.scaling.high_performance_adapters import ScalableRetroLoRA
        from retro_peft.retrieval.robust_retrieval import RobustMockRetriever
        
        # Create high-performance components
        retriever = RobustMockRetriever(
            cache_size=50,
            enable_monitoring=True
        )
        
        adapter = ScalableRetroLoRA(
            model_name="integration_scaling_test",
            rank=16,
            cache_size=100,
            max_batch_size=8,
            enable_compression=True
        )
        
        # Connect components
        adapter.set_retriever(retriever)
        print("✓ High-performance components integrated")
        
        # Test concurrent processing
        def concurrent_test():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                # Submit multiple requests concurrently
                for i in range(20):
                    future = executor.submit(
                        adapter.generate,
                        f"Test concurrent request {i}",
                        max_length=50,
                        retrieval_k=2
                    )
                    futures.append(future)
                
                # Collect results
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=10.0)
                        results.append(result)
                    except Exception as e:
                        print(f"Concurrent request failed: {e}")
                
                return results
        
        # Run concurrent test
        start_time = time.time()
        concurrent_results = concurrent_test()
        total_time = time.time() - start_time
        
        successful_results = [r for r in concurrent_results if "error" not in r]
        print(f"✓ Concurrent processing: {len(successful_results)}/{len(concurrent_results)} successful in {total_time:.1f}s")
        
        # Calculate throughput
        throughput = len(successful_results) / total_time
        print(f"✓ Throughput: {throughput:.1f} requests/second")
        
        # Check final stats
        adapter_stats = adapter.get_performance_stats()
        retriever_health = retriever.get_health_status()
        
        print(f"✓ Final adapter cache hit rate: {adapter_stats['cache_performance']['hit_rate']:.2f}")
        print(f"✓ Final retriever health: {retriever_health['status']}")
        
        # Test performance optimization
        adapter.optimize_performance()
        print("✓ Final performance optimization completed")
        
        return True
    except Exception as e:
        print(f"✗ Integration scaling error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all Generation 3 tests"""
    print("=== Generation 3: MAKE IT SCALE - Test Suite ===\n")
    
    tests = [
        test_high_performance_cache,
        test_batch_processor,
        test_scalable_adapter,
        test_load_balancer,
        test_distributed_adapter,
        test_integration_scaling
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print(f"\n=== Test Results ===")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 Generation 3 - MAKE IT SCALE: ALL TESTS PASSED!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)