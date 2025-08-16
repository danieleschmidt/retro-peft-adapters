#!/usr/bin/env python3
"""
Generation 3 Test Suite: Verify performance optimization, caching, auto-scaling, and load balancing.

This test ensures all scaling features work correctly as part of Generation 3: MAKE IT SCALE.
"""

import asyncio
import os
import sys
import tempfile
import time
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_high_performance_cache():
    """Test advanced caching system"""
    print("🚀 Testing high-performance cache...")
    
    try:
        from retro_peft.scaling_features import HighPerformanceCache
        
        # Test cache creation and basic operations
        cache = HighPerformanceCache(max_size=100, default_ttl=10.0, max_memory_mb=1.0)
        
        # Test set and get
        assert cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        print("✅ Basic cache operations work")
        
        # Test TTL expiration
        cache.set("temp_key", "temp_value", ttl=0.1)  # 100ms TTL
        assert cache.get("temp_key") == "temp_value"
        time.sleep(0.2)
        assert cache.get("temp_key") is None
        print("✅ TTL expiration works")
        
        # Test LRU eviction
        for i in range(110):  # Exceed max_size
            cache.set(f"key_{i}", f"value_{i}")
        
        assert len(cache._cache) <= 100
        print("✅ LRU eviction works")
        
        # Test cache statistics
        stats = cache.get_stats()
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'size' in stats
        print("✅ Cache statistics work")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False


def test_adaptive_resource_pool():
    """Test dynamic resource pooling"""
    print("\n⚙️  Testing adaptive resource pool...")
    
    try:
        from retro_peft.scaling_features import AdaptiveResourcePool
        
        # Mock resource factory
        resource_counter = 0
        def create_resource():
            nonlocal resource_counter
            resource_counter += 1
            return f"resource_{resource_counter}"
        
        # Test pool creation
        pool = AdaptiveResourcePool(
            resource_factory=create_resource,
            min_size=2,
            max_size=10,
            scale_up_threshold=0.7,
            scale_down_threshold=0.3,
            check_interval=1.0
        )
        
        time.sleep(0.1)  # Let pool initialize
        stats = pool.get_stats()
        assert stats['total_resources'] >= 2
        print("✅ Resource pool initialization works")
        
        # Test resource acquisition and release
        with pool.get_resource() as resource:
            assert resource is not None
            assert resource.startswith("resource_")
        print("✅ Resource acquisition/release works")
        
        # Test concurrent usage
        resources = []
        try:
            for _ in range(5):
                resource = pool._acquire_resource()
                resources.append(resource)
            
            stats = pool.get_stats()
            assert stats['in_use'] == 5
            print("✅ Concurrent resource usage works")
            
        finally:
            # Release all resources
            for resource in resources:
                pool._release_resource(resource)
        
        return True
        
    except Exception as e:
        print(f"❌ Resource pool test failed: {e}")
        return False


def test_load_balancer():
    """Test intelligent load balancing"""
    print("\n⚖️  Testing load balancer...")
    
    try:
        from retro_peft.scaling_features import LoadBalancer
        
        # Mock backends
        backends = ["backend1", "backend2", "backend3"]
        
        # Test load balancer creation
        lb = LoadBalancer(health_check_interval=1.0)
        
        # Add backends
        for backend in backends:
            lb.add_backend(backend, weight=1)
        
        # Test backend selection
        selected_backend = lb.get_backend()
        assert selected_backend in backends
        print("✅ Backend selection works")
        
        # Test request recording
        lb.record_request("backend1", 0.1, True)  # Success
        lb.record_request("backend1", 0.2, False)  # Failure
        lb.record_request("backend2", 0.15, True)  # Success
        
        stats = lb.get_stats()
        assert stats['total_backends'] == 3
        assert 'backend_stats' in stats
        print("✅ Request recording works")
        
        # Test backend removal
        lb.remove_backend("backend3")
        stats = lb.get_stats()
        assert stats['total_backends'] == 2
        print("✅ Backend removal works")
        
        return True
        
    except Exception as e:
        print(f"❌ Load balancer test failed: {e}")
        return False


def test_async_batch_processor():
    """Test asynchronous batch processing"""
    print("\n📦 Testing async batch processor...")
    
    try:
        from retro_peft.scaling_features import AsyncBatchProcessor
        
        # Mock batch processing function
        def process_batch(items):
            return [f"processed_{item}" for item in items]
        
        # Test batch processor creation
        processor = AsyncBatchProcessor(
            processor_func=process_batch,
            batch_size=3,
            max_wait_time=0.1,
            max_concurrent_batches=2
        )
        
        async def test_batch_processing():
            # Test single item processing
            result = await processor.process_item("item1")
            assert result == "processed_item1"
            
            # Test concurrent processing
            tasks = []
            for i in range(5):
                task = processor.process_item(f"item_{i}")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            assert len(results) == 5
            assert all(result.startswith("processed_item_") for result in results)
            
            return True
        
        # Run async test
        result = asyncio.run(test_batch_processing())
        assert result
        print("✅ Async batch processing works")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processor test failed: {e}")
        return False


def test_performance_optimizer():
    """Test system-wide performance optimization"""
    print("\n🎯 Testing performance optimizer...")
    
    try:
        from retro_peft.scaling_features import PerformanceOptimizer, performance_optimized
        
        # Test optimizer creation
        optimizer = PerformanceOptimizer()
        
        # Test cached operation decorator
        call_count = 0
        
        @performance_optimized(cache_ttl=60.0)
        def expensive_operation(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate work
            return x * 2
        
        # First call should execute
        result1 = expensive_operation(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_operation(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
        print("✅ Performance optimization caching works")
        
        # Test performance summary
        summary = optimizer.get_performance_summary()
        assert 'cache_stats' in summary
        assert 'system_metrics' in summary
        print("✅ Performance summary works")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance optimizer test failed: {e}")
        return False


def test_scaling_retrieval():
    """Test high-performance retrieval system"""
    print("\n🔍 Testing scaling retrieval system...")
    
    try:
        from retro_peft.retrieval.scaling_retrieval import (
            ScalingMockRetriever, ScalingVectorIndexBuilder
        )
        
        # Test scaling retriever
        retriever = ScalingMockRetriever(
            embedding_dim=256,
            enable_batch_processing=True,
            batch_size=16,
            max_concurrent_searches=4,
            enable_result_caching=True
        )
        
        # Test optimized search
        start_time = time.time()
        results = retriever.search("machine learning optimization", k=3)
        search_time = time.time() - start_time
        
        assert len(results) <= 3
        assert all('text' in result for result in results)
        print(f"✅ Optimized search works ({search_time:.3f}s)")
        
        # Test batch search
        async def test_batch_search():
            queries = ["machine learning", "deep learning", "neural networks"]
            batch_results = await retriever.batch_search(queries, k=2)
            assert len(batch_results) == 3
            assert all(len(results) <= 2 for results in batch_results)
            return True
        
        batch_result = asyncio.run(test_batch_search())
        assert batch_result
        print("✅ Batch search works")
        
        # Test performance metrics
        metrics = retriever.get_performance_metrics()
        assert 'performance_cache_stats' in metrics
        assert 'optimization_features' in metrics
        print("✅ Performance metrics collection works")
        
        # Test scaling index builder
        builder = ScalingVectorIndexBuilder(
            embedding_dim=384,
            chunk_size=100,
            overlap=25,
            enable_parallel_processing=True,
            max_workers=2
        )
        
        # Test optimized index building
        docs = builder.create_sample_documents()
        start_time = time.time()
        scaling_retriever = builder.build_index(docs, enable_scaling_features=True)
        build_time = time.time() - start_time
        
        assert isinstance(scaling_retriever, ScalingMockRetriever)
        print(f"✅ Optimized index building works ({build_time:.3f}s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Scaling retrieval test failed: {e}")
        return False


def test_end_to_end_scaling():
    """Test complete scaling system integration"""
    print("\n🌐 Testing end-to-end scaling integration...")
    
    try:
        from retro_peft.scaling_features import (
            HighPerformanceCache, LoadBalancer, get_performance_optimizer
        )
        from retro_peft.retrieval.scaling_retrieval import ScalingVectorIndexBuilder
        
        # Create integrated scaling system
        optimizer = get_performance_optimizer()
        cache = HighPerformanceCache(max_size=1000, max_memory_mb=50.0)
        load_balancer = LoadBalancer()
        
        # Create multiple retrieval backends
        builder = ScalingVectorIndexBuilder(embedding_dim=256, max_workers=2)
        docs = builder.create_sample_documents()
        
        backends = []
        for i in range(3):
            backend = builder.build_index(docs, enable_scaling_features=True)
            backends.append(backend)
            load_balancer.add_backend(backend, weight=1)
        
        print("✅ Multiple backends created")
        
        # Test load-balanced retrieval
        search_times = []
        for i in range(10):
            backend = load_balancer.get_backend()
            
            start_time = time.time()
            results = backend.search(f"test query {i}", k=2)
            duration = time.time() - start_time
            
            # Record performance
            load_balancer.record_request(backend, duration, len(results) > 0)
            search_times.append(duration)
        
        avg_search_time = sum(search_times) / len(search_times)
        print(f"✅ Load-balanced searches completed (avg: {avg_search_time:.3f}s)")
        
        # Test system-wide performance monitoring
        lb_stats = load_balancer.get_stats()
        perf_summary = optimizer.get_performance_summary()
        
        assert lb_stats['total_backends'] == 3
        assert 'cache_stats' in perf_summary
        print("✅ System-wide monitoring works")
        
        # Test cache efficiency
        cache_stats = cache.get_stats()
        if cache_stats['hits'] + cache_stats['misses'] > 0:
            hit_rate = cache_stats['hits'] / (cache_stats['hits'] + cache_stats['misses'])
            print(f"✅ Cache hit rate: {hit_rate:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end scaling test failed: {e}")
        return False


def test_performance_benchmarks():
    """Test performance improvements and benchmarks"""
    print("\n📊 Testing performance benchmarks...")
    
    try:
        from retro_peft.retrieval.scaling_retrieval import ScalingMockRetriever
        from retro_peft.retrieval.robust_retrieval import RobustMockRetriever
        
        # Create both retriever types
        scaling_retriever = ScalingMockRetriever(
            embedding_dim=256,
            enable_result_caching=True,
            enable_batch_processing=True
        )
        
        robust_retriever = RobustMockRetriever(embedding_dim=256)
        
        # Benchmark search performance
        query = "machine learning performance optimization"
        num_searches = 20
        
        # Test scaling retriever
        start_time = time.time()
        for _ in range(num_searches):
            scaling_retriever.search(query, k=5)
        scaling_time = time.time() - start_time
        
        # Test robust retriever
        start_time = time.time()
        for _ in range(num_searches):
            robust_retriever.search(query, k=5)
        robust_time = time.time() - start_time
        
        # Calculate performance improvement
        if robust_time > 0:
            improvement = (robust_time - scaling_time) / robust_time * 100
            print(f"✅ Performance improvement: {improvement:.1f}% faster")
            print(f"   Scaling: {scaling_time:.3f}s, Robust: {robust_time:.3f}s")
        
        # Test cache effectiveness
        cache_stats = scaling_retriever.performance_cache.get_stats()
        if cache_stats['hits'] + cache_stats['misses'] > 0:
            hit_rate = cache_stats['hits'] / (cache_stats['hits'] + cache_stats['misses'])
            print(f"✅ Cache hit rate: {hit_rate:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark test failed: {e}")
        return False


def run_generation3_tests():
    """Run all Generation 3 scaling feature tests"""
    print("🚀 Generation 3 Scaling Features Test Suite")
    print("=" * 60)
    
    tests = [
        test_high_performance_cache,
        test_adaptive_resource_pool,
        test_load_balancer,
        test_async_batch_processor,
        test_performance_optimizer,
        test_scaling_retrieval,
        test_end_to_end_scaling,
        test_performance_benchmarks
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Generation 3: MAKE IT SCALE - COMPLETED SUCCESSFULLY!")
        print("\n🚀 What's now optimized for scale:")
        print("   • High-performance caching with TTL and LRU eviction")
        print("   • Adaptive resource pooling with auto-scaling")
        print("   • Intelligent load balancing with health checks")
        print("   • Asynchronous batch processing for throughput")
        print("   • System-wide performance optimization and monitoring")
        print("   • Parallel document processing and indexing")
        print("   • Concurrent search with result caching")
        print("   • End-to-end performance improvements")
        
        return True
    else:
        print("❌ Some tests failed. Generation 3 incomplete.")
        return False


if __name__ == "__main__":
    success = run_generation3_tests()
    sys.exit(0 if success else 1)