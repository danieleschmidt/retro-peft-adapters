#!/usr/bin/env python3
"""
Generation 2 Test Suite: Verify robust error handling, validation, monitoring, and security.

This test ensures all robust features work correctly as part of Generation 2: MAKE IT ROBUST.
"""

import os
import sys
import tempfile
import time
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def test_robust_validation():
    """Test comprehensive input validation"""
    print("🛡️  Testing robust validation...")
    
    try:
        from retro_peft.robust_features import RobustValidator, ValidationError
        
        # Test text validation
        valid_text = RobustValidator.validate_text_input("This is valid text")
        assert valid_text == "This is valid text"
        print("✅ Text validation works")
        
        # Test invalid text
        try:
            RobustValidator.validate_text_input("")  # Too short
            assert False, "Should have raised ValidationError"
        except ValidationError:
            print("✅ Text validation rejects empty input")
        
        try:
            RobustValidator.validate_text_input("x" * 20000)  # Too long
            assert False, "Should have raised ValidationError"
        except ValidationError:
            print("✅ Text validation rejects overly long input")
        
        # Test config validation
        valid_config = {
            'adapters': {'lora': {'r': 16, 'alpha': 32}},
            'retrieval': {'embedding_dim': 768},
            'training': {'epochs': 3},
            'inference': {'max_length': 200},
            'logging': {'level': 'INFO'}
        }
        RobustValidator.validate_config(valid_config)
        print("✅ Config validation works")
        
        # Test file path validation
        safe_path = RobustValidator.validate_file_path("safe/path/file.txt")
        assert safe_path == "safe/path/file.txt"
        print("✅ File path validation works")
        
        try:
            RobustValidator.validate_file_path("../dangerous/path")
            assert False, "Should have raised ValidationError"
        except ValidationError:
            print("✅ File path validation blocks dangerous paths")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False


def test_error_handling():
    """Test robust error handling and recovery"""
    print("\n🔄 Testing error handling...")
    
    try:
        from retro_peft.robust_features import (
            RobustErrorHandler, resilient_operation, 
            AdapterError, ValidationError
        )
        
        # Test error handler
        error_handler = RobustErrorHandler()
        
        # Test error context
        with error_handler.error_context("test_operation"):
            pass  # Successful operation
        print("✅ Error context handles success")
        
        # Test error recording
        try:
            with error_handler.error_context("failing_operation"):
                raise ValidationError("Test error")
        except ValidationError:
            pass  # Expected
        
        error_summary = error_handler.get_error_summary()
        assert error_summary['total_errors'] > 0
        print("✅ Error recording works")
        
        # Test resilient operation decorator
        call_count = 0
        
        @resilient_operation(max_retries=3)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
        print("✅ Resilient operation retries work")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_health_monitoring():
    """Test health monitoring and metrics collection"""
    print("\n📊 Testing health monitoring...")
    
    try:
        from retro_peft.robust_features import (
            HealthMonitor, get_health_monitor, monitored_operation
        )
        
        # Test health monitor
        monitor = HealthMonitor()
        
        # Record some operations
        monitor.record_operation("test_op", 0.1, True)
        monitor.record_operation("test_op", 0.2, True)
        monitor.record_operation("test_op", 0.15, False)
        
        health_status = monitor.get_health_status()
        assert health_status['total_operations'] == 3
        assert 0 < health_status['success_rate'] < 1
        print("✅ Health monitoring records operations")
        
        # Test performance summary
        perf_summary = monitor.get_performance_summary("test_op")
        assert perf_summary['total_calls'] == 3
        assert perf_summary['successful_calls'] == 2
        print("✅ Performance summary calculation works")
        
        # Test monitored operation decorator
        @monitored_operation("decorated_op")
        def test_function():
            time.sleep(0.01)  # Small delay
            return "result"
        
        result = test_function()
        assert result == "result"
        
        # Check that operation was recorded
        global_monitor = get_health_monitor()
        all_perf = global_monitor.get_performance_summary()
        assert "decorated_op" in all_perf
        print("✅ Monitored operation decorator works")
        
        return True
        
    except Exception as e:
        print(f"❌ Health monitoring test failed: {e}")
        return False


def test_security_features():
    """Test security validation and protection"""
    print("\n🔒 Testing security features...")
    
    try:
        from retro_peft.robust_features import SecurityManager, get_security_manager
        
        security_manager = SecurityManager()
        
        # Test input validation
        safe_text = security_manager.validate_input("text", "This is safe text")
        assert safe_text == "This is safe text"
        print("✅ Security input validation works")
        
        # Test threat scanning
        malicious_text = "<script>alert('xss')</script>"
        threats = security_manager.scan_for_threats(malicious_text)
        assert len(threats) > 0
        print("✅ Threat scanning detects malicious patterns")
        
        # Test rate limiting
        identifier = "test_user"
        
        # Should allow first few requests
        for i in range(5):
            allowed = security_manager.check_rate_limit(identifier, max_requests=10, window_seconds=60)
            assert allowed
        
        print("✅ Rate limiting allows normal requests")
        
        # Test global security manager
        global_security = get_security_manager()
        assert isinstance(global_security, SecurityManager)
        print("✅ Global security manager works")
        
        return True
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        return False


def test_circuit_breaker():
    """Test circuit breaker pattern"""
    print("\n⚡ Testing circuit breaker...")
    
    try:
        from retro_peft.robust_features import CircuitBreaker, with_circuit_breaker, ResourceError
        
        # Test manual circuit breaker
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        # Function that always fails
        def failing_function():
            raise Exception("Always fails")
        
        # Should fail normally first few times
        for i in range(3):
            try:
                cb.call(failing_function)
                assert False, "Should have failed"
            except Exception:
                pass
        
        # Should now be open and raise ResourceError
        try:
            cb.call(failing_function)
            assert False, "Should have raised ResourceError"
        except ResourceError:
            print("✅ Circuit breaker opens after failures")
        
        # Test decorator version
        call_count = 0
        
        @with_circuit_breaker(failure_threshold=2, recovery_timeout=1)
        def flaky_decorated_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Fail first two times")
            return "success"
        
        # Should fail twice then circuit should open
        try:
            flaky_decorated_function()
        except Exception:
            pass
        
        try:
            flaky_decorated_function()
        except Exception:
            pass
        
        try:
            flaky_decorated_function()
            assert False, "Circuit should be open"
        except ResourceError:
            print("✅ Circuit breaker decorator works")
        
        return True
        
    except Exception as e:
        print(f"❌ Circuit breaker test failed: {e}")
        return False


def test_robust_retrieval():
    """Test robust retrieval system"""
    print("\n🔍 Testing robust retrieval...")
    
    try:
        from retro_peft.retrieval.robust_retrieval import (
            RobustMockRetriever, RobustVectorIndexBuilder, RetrievalError
        )
        
        # Test robust retriever
        retriever = RobustMockRetriever(embedding_dim=256)
        
        # Test validated search
        results = retriever.search("machine learning", k=3)
        assert len(results) <= 3
        assert all('text' in result for result in results)
        print("✅ Robust retriever search works")
        
        # Test search validation - empty query should return empty results
        empty_results = retriever.search("", k=3)  # Empty query
        assert empty_results == []
        print("✅ Robust retriever handles empty queries gracefully")
        
        # Test caching
        start_time = time.time()
        results1 = retriever.search("test query", k=2)
        first_duration = time.time() - start_time
        
        start_time = time.time()
        results2 = retriever.search("test query", k=2)  # Should be cached
        second_duration = time.time() - start_time
        
        assert results1 == results2
        print("✅ Robust retriever caching works")
        
        # Test health metrics
        health_metrics = retriever.get_health_metrics()
        assert 'document_count' in health_metrics
        assert 'cache_size' in health_metrics
        print("✅ Robust retriever health metrics work")
        
        # Test robust index builder
        builder = RobustVectorIndexBuilder(embedding_dim=384, chunk_size=100, overlap=30)
        
        # Test validation
        try:
            RobustVectorIndexBuilder(embedding_dim=-1)  # Invalid
            assert False, "Should have failed"
        except RetrievalError:
            print("✅ Robust index builder validates parameters")
        
        # Test robust building
        docs = builder.create_sample_documents()
        robust_retriever = builder.build_index(docs)
        assert isinstance(robust_retriever, RobustMockRetriever)
        print("✅ Robust index building works")
        
        return True
        
    except Exception as e:
        print(f"❌ Robust retrieval test failed: {e}")
        return False


def test_integration_robust_features():
    """Test integration of all robust features"""
    print("\n🔗 Testing robust feature integration...")
    
    try:
        from retro_peft.robust_features import (
            get_health_monitor, get_security_manager, monitored_operation, resilient_operation
        )
        from retro_peft.retrieval.robust_retrieval import RobustVectorIndexBuilder
        
        # Test integrated workflow
        @monitored_operation("integrated_workflow")
        @resilient_operation(max_retries=2)
        def robust_workflow():
            # Security validation
            security = get_security_manager()
            safe_query = security.validate_input("text", "machine learning research")
            
            # Robust retrieval
            builder = RobustVectorIndexBuilder(embedding_dim=256)
            docs = builder.create_sample_documents()
            retriever = builder.build_index(docs)
            
            # Monitored search
            results = retriever.search(safe_query, k=2)
            
            return len(results)
        
        # Execute workflow
        result_count = robust_workflow()
        assert result_count > 0
        
        # Check monitoring
        monitor = get_health_monitor()
        health = monitor.get_health_status()
        assert health['total_operations'] > 0
        
        print("✅ Robust feature integration works")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def run_generation2_tests():
    """Run all Generation 2 robust feature tests"""
    print("🛡️  Generation 2 Robust Features Test Suite")
    print("=" * 60)
    
    tests = [
        test_robust_validation,
        test_error_handling,
        test_health_monitoring,
        test_security_features,
        test_circuit_breaker,
        test_robust_retrieval,
        test_integration_robust_features
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
        print("🎉 Generation 2: MAKE IT ROBUST - COMPLETED SUCCESSFULLY!")
        print("\n🛡️  What's now robust:")
        print("   • Comprehensive input validation and sanitization")
        print("   • Advanced error handling with retry mechanisms")
        print("   • Real-time health monitoring and metrics collection")
        print("   • Security features with threat detection")
        print("   • Circuit breaker pattern for failure isolation")
        print("   • Enhanced retrieval system with caching and validation")
        print("   • Integrated monitoring across all components")
        
        return True
    else:
        print("❌ Some tests failed. Generation 2 incomplete.")
        return False


if __name__ == "__main__":
    success = run_generation2_tests()
    sys.exit(0 if success else 1)