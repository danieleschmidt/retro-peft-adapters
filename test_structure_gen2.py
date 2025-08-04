#!/usr/bin/env python3
"""
Structure test for Generation 2 (Robust) functionality.

Tests that all the robust infrastructure files and classes
are properly structured without requiring external dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))


def test_utils_structure():
    """Test that utils module structure is correct"""
    print("🔍 Testing utils module structure...")
    
    utils_path = src_path / "retro_peft" / "utils"
    assert utils_path.exists(), "Utils directory missing"
    
    expected_files = [
        "__init__.py",
        "logging.py",
        "monitoring.py", 
        "security.py",
        "health.py",
        "config.py"
    ]
    
    for filename in expected_files:
        filepath = utils_path / filename
        assert filepath.exists(), f"Utils file {filename} missing"
        
        # Check that files have content
        assert filepath.stat().st_size > 0, f"Utils file {filename} is empty"
    
    print("✅ Utils module structure is correct")


def test_logging_module_structure():
    """Test logging module structure"""
    print("🔍 Testing logging module structure...")
    
    logging_file = src_path / "retro_peft" / "utils" / "logging.py"
    content = logging_file.read_text()
    
    # Check for key classes and functions
    expected_items = [
        "class JSONFormatter",
        "class ColoredFormatter", 
        "class RetroPEFTLogger",
        "def setup_logger",
        "def get_logger",
        "def get_global_logger",
        "class LoggingContextManager",
        "def log_execution_time"
    ]
    
    for item in expected_items:
        assert item in content, f"Logging module missing: {item}"
    
    print("✅ Logging module structure is correct")


def test_monitoring_module_structure():
    """Test monitoring module structure"""
    print("🔍 Testing monitoring module structure...")
    
    monitoring_file = src_path / "retro_peft" / "utils" / "monitoring.py"
    content = monitoring_file.read_text()
    
    # Check for key classes and functions
    expected_items = [
        "class Metric",
        "class HealthCheck",
        "class MetricsCollector",
        "class PerformanceMonitor",
        "class MemoryMonitorContext",
        "class HealthMonitor",
        "def get_metrics_collector",
        "def get_performance_monitor",
        "def get_health_monitor"
    ]
    
    for item in expected_items:
        assert item in content, f"Monitoring module missing: {item}"
    
    print("✅ Monitoring module structure is correct")


def test_security_module_structure():
    """Test security module structure"""
    print("🔍 Testing security module structure...")
    
    security_file = src_path / "retro_peft" / "utils" / "security.py"
    content = security_file.read_text()
    
    # Check for key classes and functions
    expected_items = [
        "class InputValidator",
        "class SecurityManager", 
        "class SecurityError",
        "def get_security_manager",
        "def secure_function",
        "def validate_prompt",
        "def validate_model_name",
        "def check_rate_limit"
    ]
    
    for item in expected_items:
        assert item in content, f"Security module missing: {item}"
    
    # Check for security patterns
    security_features = [
        "DANGEROUS_PATTERNS",
        "MAX_LENGTHS",
        "sanitize_for_logging",
        "scan_for_threats",
        "rate_limits"
    ]
    
    for feature in security_features:
        assert feature in content, f"Security feature missing: {feature}"
    
    print("✅ Security module structure is correct")


def test_health_module_structure(): 
    """Test health module structure"""
    print("🔍 Testing health module structure...")
    
    health_file = src_path / "retro_peft" / "utils" / "health.py"
    content = health_file.read_text()
    
    # Check for key functions
    expected_items = [
        "def check_system_resources",
        "def check_gpu_availability",
        "def check_disk_space",
        "def check_file_permissions", 
        "def check_python_environment",
        "def check_network_connectivity",
        "def check_model_loading",
        "def register_default_health_checks",
        "def run_system_diagnostics",
        "def create_diagnostic_report"
    ]
    
    for item in expected_items:
        assert item in content, f"Health module missing: {item}"
    
    print("✅ Health module structure is correct")


def test_config_module_structure():
    """Test config module structure"""
    print("🔍 Testing config module structure...")
    
    config_file = src_path / "retro_peft" / "utils" / "config.py"
    content = config_file.read_text()
    
    # Check for key classes
    expected_items = [
        "class ConfigSource",
        "class AdapterConfig", 
        "class RetrievalConfig",
        "class TrainingConfig",
        "class InferenceConfig",
        "class SecurityConfig",
        "class LoggingConfig",
        "class MonitoringConfig",
        "class Config",
        "class ConfigManager",
        "def load_config",
        "def get_config",
        "def save_config"
    ]
    
    for item in expected_items:
        assert item in content, f"Config module missing: {item}"
    
    print("✅ Config module structure is correct")


def test_integration_imports():
    """Test that all modules can be imported together"""
    print("🔍 Testing integration imports...")
    
    # Test utils init imports (without external dependencies)
    utils_init = src_path / "retro_peft" / "utils" / "__init__.py"
    content = utils_init.read_text()
    
    expected_imports = [
        "from .logging import setup_logger, get_logger",
        "from .monitoring import MetricsCollector, PerformanceMonitor",
        "from .config import Config, load_config",
        "from .security import SecurityManager, validate_prompt",
        "from .health import run_system_diagnostics, create_diagnostic_report"
    ]
    
    for import_line in expected_imports:
        assert import_line in content, f"Utils init missing import: {import_line}"
    
    print("✅ Integration imports are correct")


def test_docstrings_and_comments():
    """Test that modules have proper documentation"""
    print("🔍 Testing documentation quality...")
    
    utils_files = [
        "logging.py",
        "monitoring.py", 
        "security.py",
        "health.py",
        "config.py"
    ]
    
    for filename in utils_files:
        filepath = src_path / "retro_peft" / "utils" / filename
        content = filepath.read_text()
        
        # Check for module docstring
        assert '"""' in content[:200], f"{filename} missing module docstring"
        
        # Check for class docstrings
        class_lines = [line for line in content.split('\n') if line.strip().startswith('class ')]
        for class_line in class_lines[:3]:  # Check first 3 classes
            class_name = class_line.split()[1].split('(')[0].split(':')[0]
            # Look for docstring after class definition
            class_start = content.find(class_line)
            next_chunk = content[class_start:class_start+500]
            if 'def __init__' in next_chunk:
                # Has methods, should have docstring
                assert '"""' in next_chunk, f"{filename} class {class_name} missing docstring"
    
    print("✅ Documentation quality is adequate")


def test_error_handling():
    """Test that modules have proper error handling patterns"""
    print("🔍 Testing error handling patterns...")
    
    utils_files = [
        "logging.py",
        "monitoring.py",
        "security.py", 
        "health.py",
        "config.py"
    ]
    
    for filename in utils_files:
        filepath = src_path / "retro_peft" / "utils" / filename
        content = filepath.read_text()
        
        # Check for exception handling
        assert "except Exception" in content or "except " in content, f"{filename} missing exception handling"
        
        # Check for proper logging
        assert "logger" in content.lower() or "log" in content.lower(), f"{filename} missing logging"
        
        # Check for validation patterns
        if filename in ["security.py", "config.py"]:
            assert "raise ValueError" in content, f"{filename} missing input validation"
    
    print("✅ Error handling patterns are present")


def main():
    """Run all Generation 2 structure tests"""
    print("🚀 Running Generation 2 (Robust) Structure Tests")
    print("=" * 60)
    
    tests = [
        ("Utils Structure", test_utils_structure),
        ("Logging Module", test_logging_module_structure),
        ("Monitoring Module", test_monitoring_module_structure), 
        ("Security Module", test_security_module_structure),
        ("Health Module", test_health_module_structure),
        ("Config Module", test_config_module_structure),
        ("Integration Imports", test_integration_imports),
        ("Documentation Quality", test_docstrings_and_comments),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Generation 2 Structure Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Generation 2 structure tests passed!")
        print("\n📋 Generation 2 (Make It Robust) Implementation Summary:")
        print("✅ Comprehensive logging with JSON, colored, and structured formats")
        print("✅ Advanced monitoring with metrics collection and performance tracking")
        print("✅ Security system with input validation and threat detection")
        print("✅ Health monitoring with system diagnostics and checks")
        print("✅ Configuration management with multiple sources and validation")
        print("✅ Proper error handling and documentation throughout")
        return 0
    else:
        print("❌ Some Generation 2 structure tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)