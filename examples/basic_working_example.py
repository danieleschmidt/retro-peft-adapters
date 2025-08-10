#!/usr/bin/env python3
"""
Basic working example of Retro-PEFT-Adapters functionality.

This example demonstrates the core features without requiring heavy dependencies,
suitable for development and testing environments.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

def main():
    """Run basic working example"""
    print("🚀 Retro-PEFT-Adapters Basic Working Example")
    print("=" * 50)
    
    # Test package import
    print("\n1. Testing package import...")
    try:
        import retro_peft
        print(f"✅ Package imported successfully")
        print(f"   Version: {retro_peft.__version__}")
        print(f"   Author: {retro_peft.__author__}")
        print(f"   Available components: {retro_peft.__all__}")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test basic retrieval system
    print("\n2. Testing retrieval system...")
    try:
        from retro_peft.retrieval import VectorIndexBuilder, MockRetriever
        
        # Create sample documents
        builder = VectorIndexBuilder(embedding_dim=384, chunk_size=50)
        sample_docs = builder.create_sample_documents()
        
        print(f"✅ Created {len(sample_docs)} sample documents")
        
        # Build index
        retriever = builder.build_index(sample_docs)
        print(f"✅ Built retrieval index with {retriever.get_document_count()} documents")
        
        # Test retrieval
        results = retriever.search("machine learning algorithms", k=3)
        print(f"✅ Retrieved {len(results)} results for query 'machine learning algorithms'")
        
        for i, result in enumerate(results):
            print(f"   Result {i+1}: {result['text'][:100]}... (score: {result['score']:.3f})")
            
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        return False
    
    # Test adapter components (basic import)
    print("\n3. Testing adapter components...")
    try:
        # Test lazy imports
        base_adapter = retro_peft.BaseRetroAdapter
        retro_lora = retro_peft.RetroLoRA
        
        print(f"✅ BaseRetroAdapter available: {base_adapter.__name__}")
        print(f"✅ RetroLoRA available: {retro_lora.__name__}")
        
        if _TORCH_AVAILABLE:
            print("✅ PyTorch is available - adapters can be instantiated")
        else:
            print("⚠️  PyTorch not available - adapters will require dependencies")
            
    except Exception as e:
        print(f"❌ Adapter test failed: {e}")
        return False
    
    # Test mock retrieval integration
    print("\n4. Testing retrieval integration...")
    try:
        # Create retriever with custom documents
        custom_docs = [
            {
                "text": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that adapts large pre-trained models by learning pairs of rank-decomposition matrices.",
                "metadata": {"topic": "lora", "category": "adaptation"}
            },
            {
                "text": "Retrieval-augmented generation enhances language models by incorporating relevant information from external knowledge bases during generation.",
                "metadata": {"topic": "rag", "category": "enhancement"}
            }
        ]
        
        custom_retriever = MockRetriever(mock_documents=custom_docs, embedding_dim=768)
        
        if _TORCH_AVAILABLE:
            # Test retrieval with tensor queries
            query_tensor = torch.randn(1, 768)  # Mock query embedding
            context_embeddings, metadata = custom_retriever.retrieve(
                query_embeddings=query_tensor, k=2
            )
            print(f"✅ Mock retrieval with tensors: {context_embeddings.shape}")
            print(f"   Retrieved {len(metadata[0])} documents")
        else:
            # Test text-based search
            results = custom_retriever.search("adaptation techniques", k=2)
            print(f"✅ Text-based search: {len(results)} results")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ All basic functionality tests passed!")
    print("\n📚 What this demonstrates:")
    print("   • Package can be imported and basic metadata accessed")
    print("   • Retrieval system works with mock documents")
    print("   • Adapter classes are available for instantiation")
    print("   • Text-based search and embedding retrieval work")
    print("\n🎯 Next steps:")
    print("   • Install PyTorch for full adapter functionality")
    print("   • Add domain-specific documents for your use case")
    print("   • Experiment with different chunk sizes and retrieval parameters")
    
    return True

def test_advanced_features():
    """Test more advanced features if dependencies are available"""
    print("\n🔧 Testing advanced features...")
    
    if not _TORCH_AVAILABLE:
        print("⚠️  PyTorch not available - skipping advanced tests")
        return
    
    try:
        # Test retriever persistence
        print("   Testing retriever save/load...")
        from retro_peft.retrieval import MockRetriever
        
        retriever = MockRetriever(embedding_dim=256)
        temp_path = "/tmp/test_retriever.json"
        
        retriever.save_index(temp_path)
        loaded_retriever = MockRetriever.load_index(temp_path)
        
        print(f"   ✅ Saved and loaded retriever: {loaded_retriever.get_document_count()} docs")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        print(f"   ❌ Advanced test failed: {e}")

if __name__ == "__main__":
    success = main()
    
    if success:
        test_advanced_features()
    
    exit(0 if success else 1)