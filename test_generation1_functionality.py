#!/usr/bin/env python3
"""
Generation 1 Test Suite: Verify basic functionality works across all components.

This test ensures all basic features work correctly as part of Generation 1: MAKE IT WORK.
"""

import os
import sys
import tempfile
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def test_package_imports():
    """Test that all package components can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        # Test main package
        import retro_peft
        assert hasattr(retro_peft, '__version__')
        print(f"✅ Main package: {retro_peft.__version__}")
        
        # Test adapters
        from retro_peft.adapters import BaseRetroAdapter
        print("✅ BaseRetroAdapter imported")
        
        # Test retrieval
        from retro_peft.retrieval import VectorIndexBuilder, MockRetriever
        print("✅ Retrieval components imported")
        
        # Test utilities
        from retro_peft.utils import config, logging
        print("✅ Utility components imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_mock_retrieval_functionality():
    """Test mock retrieval system works correctly"""
    print("\n🔍 Testing mock retrieval functionality...")
    
    try:
        from retro_peft.retrieval import MockRetriever
        
        # Test basic creation
        retriever = MockRetriever(embedding_dim=256)
        assert retriever.get_document_count() == 5  # Default documents
        print("✅ MockRetriever creation")
        
        # Test text search
        results = retriever.search("machine learning", k=3)
        assert len(results) > 0
        assert all('score' in result for result in results)
        print(f"✅ Text search: {len(results)} results")
        
        # Test custom documents
        custom_docs = [
            {
                "text": "Custom document about retrieval systems",
                "metadata": {"type": "custom"}
            }
        ]
        custom_retriever = MockRetriever(mock_documents=custom_docs)
        assert custom_retriever.get_document_count() == 1
        print("✅ Custom documents")
        
        # Test save/load
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            retriever.save_index(tmp.name)
            loaded = MockRetriever.load_index(tmp.name)
            assert loaded.get_document_count() == retriever.get_document_count()
            os.unlink(tmp.name)
        print("✅ Save/load functionality")
        
        # Test with PyTorch if available
        if _TORCH_AVAILABLE:
            query_tensor = torch.randn(1, 256)
            context, metadata = retriever.retrieve(query_embeddings=query_tensor, k=2)
            assert context.shape == (1, 2, 256)
            assert len(metadata) == 1
            print("✅ PyTorch tensor retrieval")
        
        return True
        
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        return False


def test_vector_index_builder():
    """Test vector index builder functionality"""
    print("\n🔍 Testing vector index builder...")
    
    try:
        from retro_peft.retrieval import VectorIndexBuilder
        
        # Test creation
        builder = VectorIndexBuilder(embedding_dim=384, chunk_size=50)
        print("✅ VectorIndexBuilder creation")
        
        # Test sample documents
        docs = builder.create_sample_documents()
        assert len(docs) == 5
        assert all('text' in doc and 'metadata' in doc for doc in docs)
        print("✅ Sample documents creation")
        
        # Test chunking
        test_text = "This is a test document. " * 20  # Create longer text
        chunks = builder.chunk_text(test_text, {"source": "test"})
        assert len(chunks) > 1
        assert all('text' in chunk for chunk in chunks)
        print(f"✅ Text chunking: {len(chunks)} chunks")
        
        # Test index building
        retriever = builder.build_index(docs)
        assert retriever.get_document_count() >= len(docs)
        print("✅ Index building")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector index builder test failed: {e}")
        return False


def test_adapter_imports():
    """Test that adapter classes can be imported and basic info accessed"""
    print("\n🔍 Testing adapter classes...")
    
    try:
        # Test lazy imports
        from retro_peft import RetroLoRA, RetroAdaLoRA, RetroIA3
        
        print("✅ RetroLoRA imported")
        print("✅ RetroAdaLoRA imported") 
        print("✅ RetroIA3 imported")
        
        # Test that they are classes
        assert callable(RetroLoRA)
        assert callable(RetroAdaLoRA)
        assert callable(RetroIA3)
        print("✅ Adapter classes are callable")
        
        return True
        
    except Exception as e:
        print(f"❌ Adapter import test failed: {e}")
        return False


def test_configuration_system():
    """Test configuration loading and validation"""
    print("\n🔍 Testing configuration system...")
    
    try:
        from retro_peft.utils import config
        
        # Test default configuration
        default_config = config.get_default_config()
        assert isinstance(default_config, dict)
        assert 'adapters' in default_config
        print("✅ Default configuration loaded")
        
        # Test configuration validation
        test_config = {
            'adapters': {
                'lora': {'r': 16, 'alpha': 32}
            },
            'retrieval': {
                'embedding_dim': 768,
                'chunk_size': 512
            },
            'training': {
                'epochs': 3
            },
            'inference': {
                'max_length': 200
            },
            'logging': {
                'level': 'INFO'
            }
        }
        is_valid = config.validate_config(test_config)
        assert is_valid
        print("✅ Configuration validation")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_logging_system():
    """Test logging functionality"""
    print("\n🔍 Testing logging system...")
    
    try:
        from retro_peft.utils import logging as retro_logging
        
        # Test logger creation
        logger = retro_logging.get_logger("test_logger")
        logger.info("Test log message")
        print("✅ Logger creation and usage")
        
        # Test performance logging
        with retro_logging.performance_timer("test_operation"):
            pass
        print("✅ Performance timer")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False


def test_basic_integration():
    """Test basic integration between components"""
    print("\n🔍 Testing component integration...")
    
    try:
        from retro_peft.retrieval import VectorIndexBuilder, MockRetriever
        from retro_peft.utils import config
        
        # Create retriever with configuration
        retriever_config = config.get_default_config()['retrieval']
        # Filter config to match VectorIndexBuilder parameters
        builder_config = {
            'embedding_dim': retriever_config['embedding_dim'],
            'chunk_size': retriever_config['chunk_size'],
            'overlap': retriever_config.get('overlap', 50)
        }
        builder = VectorIndexBuilder(**builder_config)
        
        # Build index and test retrieval
        docs = builder.create_sample_documents()
        retriever = builder.build_index(docs)
        
        # Test search
        results = retriever.search("machine learning adaptation", k=2)
        assert len(results) > 0
        print("✅ Configuration-driven retrieval")
        
        # Test metadata preservation
        for result in results:
            assert 'metadata' in result
            assert 'score' in result
        print("✅ Metadata preservation")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def run_generation1_tests():
    """Run all Generation 1 functionality tests"""
    print("🚀 Generation 1 Functionality Test Suite")
    print("=" * 60)
    
    tests = [
        test_package_imports,
        test_mock_retrieval_functionality, 
        test_vector_index_builder,
        test_adapter_imports,
        test_configuration_system,
        test_logging_system,
        test_basic_integration
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
        print("🎉 Generation 1: MAKE IT WORK - COMPLETED SUCCESSFULLY!")
        print("\n✨ What works now:")
        print("   • All package components can be imported")
        print("   • Mock retrieval system works with text search")
        print("   • Vector index builder creates and manages indices")
        print("   • Configuration system loads and validates settings")
        print("   • Logging system provides structured output")
        print("   • Components integrate properly")
        
        if _TORCH_AVAILABLE:
            print("   • PyTorch tensor operations work")
        else:
            print("   • Ready for PyTorch when available")
        
        return True
    else:
        print("❌ Some tests failed. Generation 1 incomplete.")
        return False


if __name__ == "__main__":
    success = run_generation1_tests()
    sys.exit(0 if success else 1)