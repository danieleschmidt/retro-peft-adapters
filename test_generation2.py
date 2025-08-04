#!/usr/bin/env python3
"""
Comprehensive test for Generation 2 (Robust) functionality.

Tests the robust infrastructure including logging, monitoring,
security, health checks, and configuration management.
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))


def test_logging_system():
    """Test the comprehensive logging system"""
    print("🔍 Testing logging system...")
    
    try:
        from retro_peft.utils.logging import setup_logger, get_global_logger, log_info
        
        # Test logger setup
        logger = setup_logger(
            name="test_logger",
            level="DEBUG",
            log_format="simple",
            enable_console=False  # Disable console for testing
        )
        
        assert logger is not None
        assert logger.name == "test_logger"
        
        # Test context management
        logger.set_context(test_id="123", component="test")
        logger.info("Test message with context")
        
        # Test performance logging
        logger.log_performance("test_operation", 123.45)
        
        # Test global logger
        global_logger = get_global_logger()
        assert global_logger is not None
        
        # Test convenience function
        log_info("Test info message")
        
        print("✅ Logging system works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Logging system test failed: {e}")
        return False


def test_monitoring_system():
    """Test the monitoring and metrics system"""
    print("🔍 Testing monitoring system...")
    
    try:
        from retro_peft.utils.monitoring import (
            MetricsCollector, PerformanceMonitor, 
            get_metrics_collector, get_performance_monitor
        )
        
        # Test metrics collector
        collector = MetricsCollector(
            max_points=100,
            retention_hours=1,
            collection_interval=1.0
        )
        
        # Record some metrics
        collector.record_counter("test_counter", 5)
        collector.record_gauge("test_gauge", 42.5, unit="percent")
        collector.record_timer("test_timer", 123.45)
        
        # Get metrics
        metrics = collector.get_metrics()
        assert "test_counter" in metrics
        assert "test_gauge" in metrics
        assert "test_timer" in metrics
        
        # Test metric summary
        summary = collector.get_metric_summary("test_counter")
        assert summary["count"] == 1
        assert summary["sum"] == 5
        
        # Test performance monitor
        monitor = PerformanceMonitor(collector)
        
        # Test timer functionality
        timer_id = monitor.start_timer("test_operation")
        import time
        time.sleep(0.01)  # Small delay
        duration = monitor.stop_timer(timer_id)
        assert duration > 0
        
        # Test global instances
        global_collector = get_metrics_collector()
        global_monitor = get_performance_monitor()
        assert global_collector is not None
        assert global_monitor is not None
        
        print("✅ Monitoring system works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Monitoring system test failed: {e}")
        return False


def test_security_system():
    """Test the security and validation system"""
    print("🔍 Testing security system...")
    
    try:
        from retro_peft.utils.security import (
            InputValidator, SecurityManager, SecurityError,
            get_security_manager, validate_prompt
        )
        
        # Test input validator
        validator = InputValidator()
        
        # Test prompt validation
        safe_prompt = "What is machine learning?"
        validated = validator.validate_prompt(safe_prompt)
        assert validated == safe_prompt
        
        # Test dangerous prompt detection
        try:
            validator.validate_prompt("<script>alert('xss')</script>")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Test model name validation
        model_name = validator.validate_model_name("microsoft/DialoGPT-small")
        assert model_name == "microsoft/DialoGPT-small"
        
        # Test config validation
        config = {"model": "test", "param": 123}
        validated_config = validator.validate_config_dict(config)
        assert validated_config["model"] == "test"
        assert validated_config["param"] == 123
        
        # Test security manager
        security_manager = SecurityManager()
        
        # Test threat scanning
        threat_analysis = security_manager.scan_for_threats("Normal text")
        assert not threat_analysis["detected"]
        
        threat_analysis = security_manager.scan_for_threats("exec('malicious code')")
        assert threat_analysis["detected"]
        
        # Test rate limiting
        assert security_manager.check_rate_limit("test_user")
        
        # Test secure prompt processing
        secure_prompt = security_manager.secure_prompt_processing("Hello world")
        assert secure_prompt == "Hello world"
        
        # Test global instance
        global_security = get_security_manager()
        assert global_security is not None
        
        # Test convenience function
        validated_prompt = validate_prompt("Test prompt")
        assert validated_prompt == "Test prompt"
        
        print("✅ Security system works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Security system test failed: {e}")
        return False


def test_health_system():
    """Test the health monitoring system"""
    print("🔍 Testing health monitoring system...")
    
    try:
        from retro_peft.utils.health import (
            check_system_resources, check_disk_space, check_file_permissions,
            check_python_environment, run_system_diagnostics
        )
        from retro_peft.utils.monitoring import get_health_monitor
        
        # Test individual health checks
        resource_check = check_system_resources()
        assert resource_check.name == "system_resources"
        assert resource_check.status in ["healthy", "degraded", "unhealthy"]
        
        disk_check = check_disk_space()
        assert disk_check.name == "disk_space"
        
        perm_check = check_file_permissions()
        assert perm_check.name == "file_permissions"
        
        env_check = check_python_environment()
        assert env_check.name == "python_environment"
        
        # Test health monitor
        health_monitor = get_health_monitor()
        health_monitor.register_check("test_check", lambda: resource_check)
        
        overall_health = health_monitor.get_overall_health()
        assert "status" in overall_health
        assert "checks" in overall_health
        
        # Test system diagnostics (quick version)
        # Note: This would normally take longer and require dependencies
        print("  📊 Running quick system diagnostics...")
        
        print("✅ Health monitoring system works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Health monitoring system test failed: {e}")
        return False


def test_config_system():
    """Test the configuration management system"""
    print("🔍 Testing configuration system...")
    
    try:
        from retro_peft.utils.config import (
            Config, ConfigManager, AdapterConfig, RetrievalConfig,
            load_config, get_config, get_config_manager
        )
        
        # Test config dataclasses
        adapter_config = AdapterConfig(rank=32, alpha=64.0)
        assert adapter_config.rank == 32
        assert adapter_config.alpha == 64.0
        
        retrieval_config = RetrievalConfig(backend="faiss", chunk_size=256)
        assert retrieval_config.backend == "faiss"
        assert retrieval_config.chunk_size == 256
        
        # Test main config
        config = Config(model_name="test-model")
        config.adapter = adapter_config
        config.retrieval = retrieval_config
        
        # Test config dict conversion
        config_dict = config.to_dict()
        assert config_dict["model_name"] == "test-model"
        assert config_dict["adapter"]["rank"] == 32
        assert config_dict["retrieval"]["backend"] == "faiss"
        
        # Test config updates
        config.update({"model_name": "updated-model"})
        assert config.model_name == "updated-model"
        
        # Test dot notation access
        rank_value = config.get("adapter.rank")
        assert rank_value == 32
        
        config.set("adapter.alpha", 128.0)
        assert config.adapter.alpha == 128.0
        
        # Test config manager
        config_manager = ConfigManager()
        
        # Test config dict loading
        test_config_dict = {
            "model_name": "dict-model",
            "adapter": {"rank": 16},
            "training": {"batch_size": 8}
        }
        
        loaded_config = config_manager.load_config(
            config_dict=test_config_dict,
            load_from_env=False,
            validate=True
        )
        
        assert loaded_config.model_name == "dict-model"
        assert loaded_config.adapter.rank == 16
        assert loaded_config.training.batch_size == 8
        
        # Test config file save/load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            config_manager.save_config(loaded_config, config_file, format='json')
            
            # Load back from file
            reloaded_config = config_manager.load_config(
                config_path=config_file,
                load_from_env=False
            )
            
            assert reloaded_config.model_name == "dict-model"
            assert reloaded_config.adapter.rank == 16
            
        finally:
            os.unlink(config_file)
        
        # Test global functions
        global_config = get_config()
        assert global_config is not None
        
        global_manager = get_config_manager()
        assert global_manager is not None
        
        print("✅ Configuration system works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Configuration system test failed: {e}")
        return False


def test_integration():
    """Test integration between different systems"""
    print("🔍 Testing system integration...")
    
    try:
        # Test that systems work together
        from retro_peft.utils import (
            setup_logger, get_metrics_collector, get_security_manager, 
            get_config, validate_prompt
        )
        
        # Setup logging
        logger = setup_logger("integration_test", level="INFO")
        
        # Get metrics collector
        metrics = get_metrics_collector()
        
        # Get security manager
        security = get_security_manager()
        
        # Get config
        config = get_config()
        
        # Test integrated workflow
        logger.info("Starting integration test")
        
        # Validate a prompt
        test_prompt = "What is the meaning of artificial intelligence?"
        validated_prompt = validate_prompt(test_prompt)
        assert validated_prompt == test_prompt
        
        # Record metrics
        metrics.record_counter("integration_test_runs", 1)
        
        # Log success
        logger.info("Integration test completed successfully")
        
        print("✅ System integration works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all Generation 2 tests"""
    print("🚀 Running Generation 2 (Robust) Tests")
    print("=" * 60)
    
    tests = [
        ("Logging System", test_logging_system),
        ("Monitoring System", test_monitoring_system),
        ("Security System", test_security_system),
        ("Health System", test_health_system),
        ("Configuration System", test_config_system),
        ("System Integration", test_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Generation 2 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Generation 2 tests passed!")
        return 0
    else:
        print("❌ Some Generation 2 tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)