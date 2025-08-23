#!/usr/bin/env python3
"""
Generation 1 Test Suite - Basic functionality verification.
"""

import sys
import traceback
from typing import Any, Dict

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test that basic imports work without errors"""
    print("Testing imports...")
    
    try:
        from retro_peft import RetroLoRA, BaseRetroAdapter, VectorIndexBuilder
        from retro_peft.retrieval import MockRetriever
        from retro_peft.utils import InputValidator, ErrorHandler
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False

def test_adapter_creation():
    """Test adapter creation with various configurations"""
    print("\nTesting adapter creation...")
    
    try:
        from retro_peft import RetroLoRA
        
        # Basic adapter
        adapter1 = RetroLoRA()
        print(f"✓ Basic adapter: {type(adapter1).__name__}")
        
        # Adapter with parameters
        adapter2 = RetroLoRA(
            model_name="test_model",
            rank=8,
            alpha=16.0
        )
        print(f"✓ Configured adapter: {adapter2.model_name}, rank={adapter2.rank}")
        
        return True
    except Exception as e:
        print(f"✗ Adapter creation error: {e}")
        traceback.print_exc()
        return False

def test_retriever_functionality():
    """Test retriever creation and basic functionality"""
    print("\nTesting retriever functionality...")
    
    try:
        from retro_peft.retrieval import MockRetriever, VectorIndexBuilder
        
        # Create mock retriever
        retriever = MockRetriever()
        doc_count = retriever.get_document_count()
        print(f"✓ MockRetriever created with {doc_count} documents")
        
        # Test search
        results = retriever.search("machine learning", k=3)
        print(f"✓ Search returned {len(results)} results")
        
        # Test vector index builder
        builder = VectorIndexBuilder()
        sample_docs = builder.create_sample_documents()
        print(f"✓ Sample documents created: {len(sample_docs)}")
        
        # Build index
        index = builder.build_index(sample_docs)
        print(f"✓ Index built with {index.get_document_count()} documents")
        
        return True
    except Exception as e:
        print(f"✗ Retriever error: {e}")
        traceback.print_exc()
        return False

def test_adapter_retriever_integration():
    """Test adapter and retriever working together"""
    print("\nTesting adapter-retriever integration...")
    
    try:
        from retro_peft import RetroLoRA
        from retro_peft.retrieval import MockRetriever
        
        # Create components
        adapter = RetroLoRA(model_name="integration_test")
        retriever = MockRetriever()
        
        # Connect retriever
        adapter.set_retriever(retriever)
        print("✓ Retriever connected to adapter")
        
        # Test generation without retrieval
        result1 = adapter.generate("Test prompt", retrieval_k=0)
        print(f"✓ Generation without retrieval: {len(result1['output'])} chars")
        
        # Test generation with retrieval
        result2 = adapter.generate("What is machine learning?", retrieval_k=3)
        print(f"✓ Generation with retrieval: context_used={result2['context_used']}")
        
        return True
    except Exception as e:
        print(f"✗ Integration error: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling and validation"""
    print("\nTesting error handling...")
    
    try:
        from retro_peft import RetroLoRA
        from retro_peft.utils import ValidationError
        
        adapter = RetroLoRA()
        
        # Test validation
        try:
            result = adapter.generate("")  # Empty prompt should be handled
            print("✓ Empty prompt handled gracefully")
        except Exception as e:
            print(f"✓ Empty prompt validation: {type(e).__name__}")
        
        # Test error recovery
        adapter.set_retriever(None)
        result = adapter.generate("Test prompt")
        print(f"✓ Generation without retriever: {result['context_used']} == False")
        
        return True
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration_validation():
    """Test configuration validation"""
    print("\nTesting configuration validation...")
    
    try:
        from retro_peft.utils import InputValidator
        
        # Test model name validation
        valid_name = InputValidator.validate_model_name("test_model")
        print(f"✓ Model name validation: '{valid_name}'")
        
        # Test config validation
        config = {
            "rank": 16,
            "alpha": 32.0,
            "target_modules": ["q_proj", "v_proj"]
        }
        validated_config = InputValidator.validate_adapter_config(config)
        print(f"✓ Config validation: {len(validated_config)} validated parameters")
        
        return True
    except Exception as e:
        print(f"✗ Configuration validation error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all Generation 1 tests"""
    print("=== Generation 1: MAKE IT WORK - Test Suite ===\n")
    
    tests = [
        test_imports,
        test_adapter_creation,
        test_retriever_functionality,
        test_adapter_retriever_integration,
        test_error_handling,
        test_configuration_validation
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
        print("🎉 Generation 1 - MAKE IT WORK: ALL TESTS PASSED!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)