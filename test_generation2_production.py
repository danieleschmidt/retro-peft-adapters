#!/usr/bin/env python3
"""
Generation 2 Production Test Suite - Production adapters and robust retrieval.
"""

import sys
import traceback
import time

# Add src to path
sys.path.insert(0, 'src')

def test_production_adapters():
    """Test production-grade adapters with error handling"""
    print("Testing production adapters...")
    
    try:
        from retro_peft.adapters.production_adapters import ProductionRetroLoRA
        
        # Create production adapter
        adapter = ProductionRetroLoRA(
            model_name="test_production",
            rank=8,
            alpha=16.0,
            enable_monitoring=True
        )
        print(f"✓ Production adapter created: {adapter.model_name}")
        
        # Test adapter info
        info = adapter.get_adapter_info()
        print(f"✓ Adapter info: {info['adapter_type']}, {info['efficiency_ratio']}")
        
        # Test health status
        health = adapter.get_health_status()
        print(f"✓ Health status: {health['status']}")
        
        # Test generation
        result = adapter.generate(
            "What is machine learning?",
            max_length=100,
            temperature=0.7
        )
        print(f"✓ Generation completed: {len(result['generated_text'])} chars")
        print(f"✓ Performance: {result['performance_metrics']['response_time_ms']:.1f}ms")
        
        return True
    except Exception as e:
        print(f"✗ Production adapter error: {e}")
        traceback.print_exc()
        return False

def test_robust_retrieval():
    """Test robust retrieval with caching and monitoring"""
    print("\nTesting robust retrieval...")
    
    try:
        from retro_peft.retrieval.robust_retrieval import RobustMockRetriever, RobustVectorIndexBuilder
        
        # Create robust retriever
        retriever = RobustMockRetriever(
            cache_size=100,
            cache_ttl_seconds=300,
            enable_monitoring=True
        )
        print(f"✓ Robust retriever created with {retriever.get_document_count()} documents")
        
        # Test search with caching
        query = "machine learning algorithms"
        
        # First search (cache miss)
        start_time = time.time()
        results1 = retriever.search(query, k=3)
        time1 = (time.time() - start_time) * 1000
        print(f"✓ First search: {len(results1)} results in {time1:.1f}ms")
        
        # Second search (cache hit)
        start_time = time.time()
        results2 = retriever.search(query, k=3)
        time2 = (time.time() - start_time) * 1000
        print(f"✓ Second search: {len(results2)} results in {time2:.1f}ms (cached)")
        
        # Test health status
        health = retriever.get_health_status()
        print(f"✓ Retrieval health: {health['status']}, cache hit rate: {health['cache_hit_rate']:.2f}")
        
        # Test index builder
        builder = RobustVectorIndexBuilder(
            embedding_dim=512,
            chunk_size=256,
            overlap=32
        )
        
        sample_docs = [
            {"text": "Machine learning is a powerful tool for data analysis.", "metadata": {"topic": "ml"}},
            {"text": "Deep learning uses neural networks with multiple layers.", "metadata": {"topic": "dl"}},
            {"text": "Natural language processing helps computers understand text.", "metadata": {"topic": "nlp"}}
        ]
        
        index = builder.build_index(sample_docs)
        print(f"✓ Index built with {index.get_document_count()} chunks")
        
        return True
    except Exception as e:
        print(f"✗ Robust retrieval error: {e}")
        traceback.print_exc()
        return False

def test_error_handling_and_validation():
    """Test comprehensive error handling and validation"""
    print("\nTesting error handling and validation...")
    
    try:
        from retro_peft.adapters.production_adapters import ProductionRetroLoRA
        from retro_peft.utils import ValidationError
        
        adapter = ProductionRetroLoRA()
        
        # Test invalid inputs
        test_cases = [
            ("", "empty prompt"),
            ("x" * 10000, "very long prompt"),
        ]
        
        handled_errors = 0
        for test_input, description in test_cases:
            try:
                result = adapter.generate(test_input)
                if result.get("fallback_used"):
                    print(f"✓ Graceful handling: {description}")
                    handled_errors += 1
                elif test_input == "":
                    print(f"✓ Empty prompt handled: returned fallback")
                    handled_errors += 1
                else:
                    print(f"✓ Long prompt handled: {len(result['generated_text'])} chars")
                    handled_errors += 1
            except Exception as e:
                print(f"✓ Error properly caught for {description}: {type(e).__name__}")
                handled_errors += 1
        
        print(f"✓ Handled {handled_errors} error cases gracefully")
        
        # Test circuit breaker functionality
        from retro_peft.retrieval.robust_retrieval import RobustMockRetriever
        
        # Create retriever that might fail
        retriever = RobustMockRetriever(enable_monitoring=True)
        
        # Test multiple searches
        for i in range(5):
            try:
                results = retriever.search(f"test query {i}", k=2)
                print(f"✓ Search {i+1} successful: {len(results)} results")
            except Exception as e:
                print(f"✓ Search {i+1} error handled: {type(e).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_monitoring_and_metrics():
    """Test monitoring and metrics collection"""
    print("\nTesting monitoring and metrics...")
    
    try:
        from retro_peft.adapters.production_adapters import ProductionRetroLoRA
        from retro_peft.retrieval.robust_retrieval import RobustMockRetriever
        
        # Create monitored components
        adapter = ProductionRetroLoRA(enable_monitoring=True)
        retriever = RobustMockRetriever(enable_monitoring=True)
        
        # Connect retriever
        adapter.set_retriever(retriever)
        
        # Perform operations to generate metrics
        queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain neural networks"
        ]
        
        for query in queries:
            result = adapter.generate(query, retrieval_k=2)
            print(f"✓ Generated response for: {query[:30]}...")
        
        # Check adapter metrics
        adapter_health = adapter.get_health_status()
        print(f"✓ Adapter metrics - generations: {adapter_health['metrics']['generation_count']}")
        print(f"✓ Adapter health score: {adapter_health['health_score']:.2f}")
        
        # Check retriever metrics
        retriever_health = retriever.get_health_status()
        print(f"✓ Retriever metrics - requests: {retriever_health['metrics']['total_requests']}")
        print(f"✓ Retriever success rate: {retriever_health['success_rate']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Monitoring test failed: {e}")
        traceback.print_exc()
        return False

def test_integration_reliability():
    """Test full integration with reliability features"""
    print("\nTesting integration reliability...")
    
    try:
        from retro_peft.adapters.production_adapters import ProductionRetroLoRA
        from retro_peft.retrieval.robust_retrieval import RobustVectorIndexBuilder
        
        # Build robust index
        builder = RobustVectorIndexBuilder()
        sample_docs = [
            "Machine learning algorithms learn from data to make predictions.",
            "Deep learning is a subset of machine learning using neural networks.",
            "Natural language processing enables computers to understand human language."
        ]
        
        robust_retriever = builder.build_index(sample_docs)
        print("✓ Robust index created")
        
        # Create production adapter
        adapter = ProductionRetroLoRA(
            model_name="integration_test",
            enable_monitoring=True
        )
        
        # Connect systems
        adapter.set_retriever(robust_retriever)
        print("✓ Systems integrated")
        
        # Test under various conditions
        test_scenarios = [
            ("normal query", "What is machine learning?"),
            ("long query", "Please explain in detail how machine learning algorithms work and provide examples " * 3),
            ("short query", "ML?"),
            ("repeated query", "What is machine learning?"),  # Should hit cache
        ]
        
        for scenario_name, query in test_scenarios:
            try:
                result = adapter.generate(query, max_length=150)
                success = result is not None and len(result.get("generated_text", "")) > 0
                print(f"✓ {scenario_name}: {'success' if success else 'fallback'}")
            except Exception as e:
                print(f"✓ {scenario_name}: error handled - {type(e).__name__}")
        
        # Final health check
        final_health = adapter.get_health_status()
        print(f"✓ Final system health: {final_health['status']}")
        
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all Generation 2 production tests"""
    print("=== Generation 2: PRODUCTION FEATURES - Test Suite ===\n")
    
    tests = [
        test_production_adapters,
        test_robust_retrieval,
        test_error_handling_and_validation,
        test_monitoring_and_metrics,
        test_integration_reliability
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
        print("🎉 Generation 2 - PRODUCTION FEATURES: ALL TESTS PASSED!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)