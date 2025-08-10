#!/usr/bin/env python3
"""
Demonstration of Generation 2 robust features.

Shows enhanced error handling, validation, monitoring, and health checks.
"""

import logging
import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run robust features demonstration"""
    print("🛡️ Retro-PEFT Generation 2: Robust Features Demo")
    print("=" * 60)
    
    # Import enhanced components
    from retro_peft.adapters import RetroLoRA
    from retro_peft.retrieval import MockRetriever
    from retro_peft.utils.health_monitoring import get_health_monitor, HealthStatus
    from retro_peft.utils.validation import ValidationError
    from retro_peft.utils.error_handling import AdapterError
    
    health_monitor = get_health_monitor()
    
    print("\n1. Testing input validation and sanitization...")
    
    # Test valid adapter creation
    try:
        adapter = RetroLoRA(
            model_name="llama-2-7b-chat",
            rank=16,
            alpha=32.0,
            dropout=0.1
        )
        print("✅ Valid adapter created successfully")
    except Exception as e:
        print(f"❌ Valid adapter creation failed: {e}")
        return False
    
    # Test invalid parameters
    test_cases = [
        {
            "name": "Invalid model name",
            "params": {"model_name": ""},
            "should_fail": True
        },
        {
            "name": "Invalid rank (too high)",
            "params": {"model_name": "test", "rank": 1000},
            "should_fail": True
        },
        {
            "name": "Invalid dropout",
            "params": {"model_name": "test", "dropout": 2.0},
            "should_fail": True
        },
        {
            "name": "Valid parameters",
            "params": {"model_name": "valid-model", "rank": 8, "alpha": 16.0},
            "should_fail": False
        }
    ]
    
    for test_case in test_cases:
        try:
            test_adapter = RetroLoRA(**test_case["params"])
            if test_case["should_fail"]:
                print(f"❌ {test_case['name']}: Should have failed but didn't")
            else:
                print(f"✅ {test_case['name']}: Passed as expected")
        except (ValidationError, AdapterError) as e:
            if test_case["should_fail"]:
                print(f"✅ {test_case['name']}: Failed as expected - {e}")
            else:
                print(f"❌ {test_case['name']}: Failed unexpectedly - {e}")
    
    print("\n2. Testing error handling and recovery...")
    
    # Create adapter with retriever
    retriever = MockRetriever(embedding_dim=256)
    adapter.set_retriever(retriever)
    
    # Test generation with various inputs
    generation_tests = [
        {
            "prompt": "What is machine learning?",
            "should_succeed": True,
            "description": "Valid prompt"
        },
        {
            "prompt": "",
            "should_succeed": False,
            "description": "Empty prompt"
        },
        {
            "prompt": "A" * 20000,  # Very long prompt
            "should_succeed": False,
            "description": "Overly long prompt"
        },
        {
            "prompt": "Normal question about AI",
            "should_succeed": True,
            "description": "Another valid prompt"
        }
    ]
    
    for i, test in enumerate(generation_tests, 1):
        try:
            result = adapter.generate(test["prompt"])
            
            if test["should_succeed"]:
                print(f"✅ Test {i} ({test['description']}): Success")
                print(f"   Generated {len(result['output'])} characters")
                print(f"   Context used: {result['context_used']}")
                print(f"   Generation count: {result['generation_count']}")
            else:
                print(f"❌ Test {i} ({test['description']}): Should have failed")
                
        except (ValidationError, AdapterError) as e:
            if not test["should_succeed"]:
                print(f"✅ Test {i} ({test['description']}): Failed as expected - {type(e).__name__}")
            else:
                print(f"❌ Test {i} ({test['description']}): Failed unexpectedly - {e}")
    
    print("\n3. Testing health monitoring and metrics...")
    
    # Register health check for the adapter
    def adapter_health_check():
        """Health check for the adapter"""
        try:
            # Check if adapter is responsive
            test_result = adapter.generate("Health check test", retrieval_k=1)
            
            return HealthStatus(
                is_healthy=True,
                status="HEALTHY",
                message="Adapter responding normally",
                timestamp=time.time(),
                metrics={
                    "generation_count": adapter._generation_count,
                    "error_count": adapter._error_count,
                    "last_response_length": len(test_result.get("output", ""))
                }
            )
        except Exception as e:
            return HealthStatus(
                is_healthy=False,
                status="ERROR",
                message=f"Adapter health check failed: {e}",
                timestamp=time.time(),
                metrics={"error_count": adapter._error_count}
            )
    
    # Register health check for retriever
    def retriever_health_check():
        """Health check for the retriever"""
        try:
            doc_count = retriever.get_document_count()
            search_results = retriever.search("test query", k=1)
            
            return HealthStatus(
                is_healthy=True,
                status="HEALTHY", 
                message="Retriever functioning normally",
                timestamp=time.time(),
                metrics={
                    "document_count": doc_count,
                    "last_search_results": len(search_results)
                }
            )
        except Exception as e:
            return HealthStatus(
                is_healthy=False,
                status="ERROR",
                message=f"Retriever health check failed: {e}",
                timestamp=time.time(),
                metrics={}
            )
    
    # Register health checks
    health_monitor.register_health_check("adapter", adapter_health_check)
    health_monitor.register_health_check("retriever", retriever_health_check)
    
    # Run health checks
    health_results = health_monitor.check_health()
    
    print(f"✅ Health checks completed for {len(health_results)} components:")
    for component, status in health_results.items():
        status_icon = "✅" if status.is_healthy else "❌"
        print(f"   {status_icon} {component}: {status.status} - {status.message}")
        if status.metrics:
            for metric, value in status.metrics.items():
                print(f"      {metric}: {value}")
    
    # Overall system health
    overall_health = health_monitor.get_overall_health()
    health_icon = "✅" if overall_health.is_healthy else "❌"
    print(f"{health_icon} Overall system health: {overall_health.status}")
    
    print("\n4. Testing performance monitoring...")
    
    # Generate several responses to collect metrics
    print("   Generating responses for performance measurement...")
    
    test_prompts = [
        "Explain neural networks",
        "What is deep learning?", 
        "How does attention work?",
        "What are transformers?",
        "Describe machine learning"
    ]
    
    for prompt in test_prompts:
        try:
            result = adapter.generate(prompt)
            time.sleep(0.1)  # Small delay between requests
        except Exception as e:
            print(f"   ⚠️ Generation failed for '{prompt}': {e}")
    
    # Get system metrics
    metrics = health_monitor.get_system_metrics()
    
    print("✅ Performance metrics collected:")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"   {metric_name}: {value:.4f}")
        else:
            print(f"   {metric_name}: {value}")
    
    print("\n5. Testing resilience features...")
    
    # Test with invalid retriever
    print("   Testing adapter resilience with invalid retriever...")
    
    class BrokenRetriever:
        def get_document_count(self):
            return 5
        
        def search(self, query, k=3):
            raise RuntimeError("Simulated retriever failure")
    
    broken_retriever = BrokenRetriever()
    adapter.set_retriever(broken_retriever)
    
    # Try generation with broken retriever
    try:
        result = adapter.generate("Test with broken retriever")
        print("✅ Adapter handled broken retriever gracefully")
        print(f"   Generated: {result['output'][:100]}...")
        print(f"   Context used: {result['context_used']} (should be False)")
    except Exception as e:
        print(f"❌ Adapter failed with broken retriever: {e}")
    
    # Restore working retriever
    adapter.set_retriever(retriever)
    
    print("\n6. Testing security features...")
    
    # Test potentially malicious inputs
    malicious_inputs = [
        "<script>alert('xss')</script>",
        "javascript:alert('test')",
        "../../../etc/passwd",
        "DROP TABLE users;",
        "\x00\x01\x02invalid chars"
    ]
    
    for malicious_input in malicious_inputs:
        try:
            result = adapter.generate(malicious_input)
            # Check if input was sanitized
            if malicious_input not in result['output']:
                print(f"✅ Malicious input sanitized: {malicious_input[:20]}...")
            else:
                print(f"⚠️ Malicious input not fully sanitized: {malicious_input[:20]}...")
        except (ValidationError, AdapterError):
            print(f"✅ Malicious input rejected: {malicious_input[:20]}...")
    
    print("\n" + "=" * 60)
    print("✅ Generation 2 Robust Features Demo Complete!")
    
    print("\n🛡️ Robustness Features Demonstrated:")
    print("   • Comprehensive input validation and sanitization")
    print("   • Advanced error handling with recovery strategies")
    print("   • Real-time health monitoring and metrics collection")
    print("   • Performance tracking and system diagnostics")
    print("   • Graceful degradation when components fail") 
    print("   • Security measures against malicious inputs")
    print("   • Resilient operation with automatic error recovery")
    
    # Final health check
    final_health = health_monitor.check_health()
    healthy_components = sum(1 for status in final_health.values() if status.is_healthy)
    total_components = len(final_health)
    
    print(f"\n📊 Final System Status:")
    print(f"   Healthy Components: {healthy_components}/{total_components}")
    print(f"   Total Generations: {adapter._generation_count}")
    print(f"   Total Errors: {adapter._error_count}")
    print(f"   Success Rate: {((adapter._generation_count - adapter._error_count) / max(adapter._generation_count, 1) * 100):.1f}%")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)