#!/usr/bin/env python3
"""
Basic functionality test without external dependencies.

Tests package structure and import paths.
"""

import sys
import os
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

def test_package_structure():
    """Test that package structure is correct"""
    print("🔍 Testing package structure...")
    
    # Check that main package exists
    retro_peft_path = src_path / "retro_peft"
    assert retro_peft_path.exists(), "retro_peft package directory missing"
    
    # Check submodules exist
    expected_modules = [
        "adapters",
        "database", 
        "retrieval",
        "training",
        "inference"
    ]
    
    for module in expected_modules:
        module_path = retro_peft_path / module
        assert module_path.exists(), f"{module} submodule directory missing"
        
        init_file = module_path / "__init__.py"
        assert init_file.exists(), f"{module}/__init__.py missing"
    
    print("✅ Package structure is correct")

def test_version_info():
    """Test that version information is accessible"""
    print("🔍 Testing version information...")
    
    # Try to read version from __init__.py without importing
    init_file = src_path / "retro_peft" / "__init__.py"
    content = init_file.read_text()
    
    assert '__version__ = "0.1.0"' in content, "Version string not found"
    assert '__author__ = "Daniel Schmidt"' in content, "Author not found"
    assert '__email__ = "daniel@terragonlabs.com"' in content, "Email not found"
    
    print("✅ Version information is correct")

def test_file_contents():
    """Test that key files have expected content"""
    print("🔍 Testing file contents...")
    
    # Test that key adapter files exist and have classes
    adapter_files = [
        "base_adapter.py",
        "retro_lora.py", 
        "retro_adalora.py",
        "retro_ia3.py"
    ]
    
    for filename in adapter_files:
        filepath = src_path / "retro_peft" / "adapters" / filename
        assert filepath.exists(), f"Adapter file {filename} missing"
        
        content = filepath.read_text()
        assert "class " in content, f"No class definition found in {filename}"
        
        if filename != "base_adapter.py":
            assert "BaseRetroAdapter" in content, f"BaseRetroAdapter not referenced in {filename}"
    
    # Test retrieval files
    retrieval_files = [
        "index_builder.py",
        "retrievers.py",
        "contextual.py"
    ]
    
    for filename in retrieval_files:
        filepath = src_path / "retro_peft" / "retrieval" / filename
        assert filepath.exists(), f"Retrieval file {filename} missing"
        
        content = filepath.read_text()
        assert "class " in content, f"No class definition found in {filename}"
    
    print("✅ File contents are correct")

def test_examples():
    """Test that example files exist and are properly structured"""
    print("🔍 Testing examples...")
    
    examples_path = repo_root / "examples"
    assert examples_path.exists(), "Examples directory missing"
    
    example_files = [
        "basic_usage.py",
        "training_example.py"
    ]
    
    for filename in example_files:
        filepath = examples_path / filename
        assert filepath.exists(), f"Example file {filename} missing"
        
        content = filepath.read_text()
        assert "def main():" in content, f"main() function not found in {filename}"
        assert "if __name__ == \"__main__\":" in content, f"Main block not found in {filename}"
    
    print("✅ Examples are properly structured")

def test_project_files():
    """Test that important project files exist"""
    print("🔍 Testing project files...")
    
    project_files = [
        "README.md",
        "pyproject.toml", 
        "LICENSE",
        "ARCHITECTURE.md",
        "PROJECT_CHARTER.md"
    ]
    
    for filename in project_files:
        filepath = repo_root / filename
        assert filepath.exists(), f"Project file {filename} missing"
        
        # Check that files have content
        assert filepath.stat().st_size > 0, f"Project file {filename} is empty"
    
    print("✅ Project files are present")

def main():
    """Run all tests"""
    print("🚀 Running Basic Functionality Tests")
    print("=" * 50)
    
    tests = [
        test_package_structure,
        test_version_info,
        test_file_contents,
        test_examples,
        test_project_files
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)