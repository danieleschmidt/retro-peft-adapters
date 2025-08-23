#!/usr/bin/env python3
"""
Generation 3 Simple Test Suite - Core scalable features without complex async.
"""

import sys
import time
import traceback

# Add src to path
sys.path.insert(0, 'src')

def test_high_performance_cache():
    """Test advanced caching system"""
    print("Testing high-performance cache...")
    
    try:
        from retro_peft.scaling.high_performance_adapters import HighPerformanceCache
        
        cache = HighPerformanceCache(max_size=50, ttl_seconds=60)
        
        # Test basic operations
        cache.put("test1", {"data": "response1"}, param="value1")
        result = cache.get("test1", param="value1")
        
        assert result is not None
        assert result["data"] == "response1"
        print("✓ Basic cache operations work")
        
        # Test cache miss
        miss = cache.get("nonexistent")
        assert miss is None
        print("✓ Cache miss handling works")
        
        # Test stats
        stats = cache.get_stats()
        print(f"✓ Cache stats: {stats['hit_rate']:.2f} hit rate, {stats['total_size']} items")
        
        return True
    except Exception as e:
        print(f"✗ Cache error: {e}")
        return False

def test_batch_request_objects():
    """Test batch processing data structures"""
    print("\nTesting batch processing structures...")
    
    try:
        from retro_peft.scaling.high_performance_adapters import BatchRequest, BatchResponse
        
        # Test BatchRequest
        request = BatchRequest(
            request_id="test_123",
            prompt="Test prompt",
            max_length=100,
            priority=1
        )
        
        assert request.request_id == "test_123"
        assert request.prompt == "Test prompt"
        assert request.priority == 1
        print("✓ BatchRequest creation works")
        
        # Test BatchResponse
        response = BatchResponse(
            request_id="test_123",
            generated_text="Test response",
            performance_metrics={"time": 50.0}
        )
        
        assert response.request_id == "test_123"
        assert response.generated_text == "Test response"
        print("✓ BatchResponse creation works")
        
        return True
    except Exception as e:
        print(f"✗ Batch structures error: {e}")
        return False

def test_scalable_adapter_basic():
    """Test basic scalable adapter functionality"""
    print("\nTesting scalable adapter basics...")
    
    try:
        from retro_peft.scaling.high_performance_adapters import ScalableRetroLoRA
        from retro_peft.retrieval import MockRetriever
        
        # Create adapter
        adapter = ScalableRetroLoRA(
            model_name="test_scalable",
            rank=8,
            cache_size=10,
            max_batch_size=2
        )
        print("✓ Scalable adapter created")
        
        # Connect retriever
        retriever = MockRetriever()
        adapter.set_retriever(retriever)
        print("✓ Retriever connected")
        
        # Test basic generation
        result = adapter.generate(
            "What is AI?",
            max_length=50,
            temperature=0.7
        )
        
        assert "generated_text" in result
        assert len(result["generated_text"]) > 0
        assert "response_time_ms" in result
        print(f"✓ Generation works: {result['response_time_ms']:.1f}ms")
        
        # Test caching with same prompt
        result2 = adapter.generate("What is AI?", max_length=50)
        assert result2.get("cache_hit") == True
        print("✓ Caching works")
        
        # Get stats
        stats = adapter.get_performance_stats()
        assert "adapter_metrics" in stats
        print(f"✓ Stats available: {stats['adapter_metrics']['total_requests']} requests")
        
        return True
    except Exception as e:
        print(f"✗ Scalable adapter error: {e}")
        traceback.print_exc()
        return False

def test_load_balancer_basic():
    """Test load balancer basic functionality"""
    print("\nTesting load balancer basics...")
    
    try:
        from retro_peft.scaling.distributed_processing import LoadBalancer, WorkerNode
        
        # Create load balancer
        lb = LoadBalancer(min_workers=1, max_workers=4)
        print("✓ Load balancer created")
        
        # Register a worker
        worker = WorkerNode(
            worker_id="test_worker",
            host="localhost",
            port=8000,
            capacity=5
        )
        lb.register_worker(worker)
        print("✓ Worker registered")
        
        # Check stats
        stats = lb.get_load_balancer_stats()
        assert stats["worker_count"] == 1
        print(f"✓ Load balancer stats: {stats['worker_count']} workers")
        
        # Test routing strategies
        strategies = ["round_robin", "least_connections"]
        for strategy in strategies:
            lb.set_routing_strategy(strategy)
            print(f"✓ Set strategy: {strategy}")
        
        return True
    except Exception as e:
        print(f"✗ Load balancer error: {e}")
        return False

def test_worker_node():
    """Test worker node functionality"""
    print("\nTesting worker node...")
    
    try:
        from retro_peft.scaling.distributed_processing import WorkerNode, WorkerStatus
        
        # Create worker
        worker = WorkerNode(
            worker_id="test_worker_123",
            host="test.example.com",
            port=9000,
            capacity=10
        )
        
        assert worker.worker_id == "test_worker_123"
        assert worker.host == "test.example.com" 
        assert worker.port == 9000
        assert worker.capacity == 10
        assert worker.status == WorkerStatus.IDLE
        print("✓ Worker node creation works")
        
        # Test status changes
        worker.current_load = 5
        worker.status = WorkerStatus.BUSY
        assert worker.current_load == 5
        assert worker.status == WorkerStatus.BUSY
        print("✓ Worker state changes work")
        
        return True
    except Exception as e:
        print(f"✗ Worker node error: {e}")
        return False

def test_performance_integration():
    """Test performance features integration"""
    print("\nTesting performance integration...")
    
    try:
        from retro_peft.scaling.high_performance_adapters import ScalableRetroLoRA
        from retro_peft.retrieval.robust_retrieval import RobustMockRetriever
        
        # Create components
        retriever = RobustMockRetriever(cache_size=20)
        adapter = ScalableRetroLoRA(
            model_name="perf_test",
            cache_size=20,
            max_batch_size=4
        )
        
        adapter.set_retriever(retriever)
        print("✓ Performance components integrated")
        
        # Test multiple requests to trigger caching
        prompts = [
            "Explain machine learning",
            "What is deep learning?", 
            "How do neural networks work?",
            "Explain machine learning"  # Repeat for cache hit
        ]
        
        results = []
        start_time = time.time()
        
        for prompt in prompts:
            result = adapter.generate(prompt, max_length=30, retrieval_k=1)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Check results
        assert len(results) == 4
        cache_hits = sum(1 for r in results if r.get("cache_hit", False))
        
        print(f"✓ Performance test: {len(results)} requests in {total_time:.1f}s")
        print(f"✓ Cache performance: {cache_hits} cache hits")
        
        # Check final stats
        final_stats = adapter.get_performance_stats()
        retriever_health = retriever.get_health_status()
        
        print(f"✓ Adapter requests: {final_stats['adapter_metrics']['total_requests']}")
        print(f"✓ Retriever health: {retriever_health['status']}")
        
        return True
    except Exception as e:
        print(f"✗ Performance integration error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all Generation 3 simple tests"""
    print("=== Generation 3: MAKE IT SCALE - Simple Test Suite ===\n")
    
    tests = [
        test_high_performance_cache,
        test_batch_request_objects,
        test_scalable_adapter_basic,
        test_load_balancer_basic,
        test_worker_node,
        test_performance_integration
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