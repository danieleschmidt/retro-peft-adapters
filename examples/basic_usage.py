"""
Basic usage example for retro-peft-adapters.

This example demonstrates how to:
1. Set up a basic retrieval-augmented adapter
2. Build a simple vector index
3. Train the adapter
4. Use it for inference

Run with: python examples/basic_usage.py
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import retro-peft components
from retro_peft.adapters import RetroLoRA
from retro_peft.retrieval import VectorIndexBuilder, FAISSRetriever
from retro_peft.inference import RetrievalInferencePipeline


def main():
    """Main example function"""
    print("🚀 Retro-PEFT-Adapters Basic Usage Example")
    print("=" * 50)
    
    # Check if we can use GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Step 1: Load base model and tokenizer
        print("\n📥 Loading base model...")
        model_name = "microsoft/DialoGPT-small"  # Small model for demonstration
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Attach tokenizer to model for convenience
        base_model.tokenizer = tokenizer
        
        print(f"✅ Loaded model: {model_name}")
        
        # Step 2: Create sample documents for retrieval
        print("\n📚 Creating sample knowledge base...")
        sample_documents = [
            "Python is a high-level programming language known for its simplicity and readability.",
            "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            "Deep learning uses neural networks with many layers to model and understand complex patterns.",
            "Natural language processing (NLP) helps computers understand and generate human language.",
            "Transformers are a type of neural network architecture that has revolutionized NLP tasks.",
            "BERT is a bidirectional encoder representation from transformers used for language understanding.",
            "GPT models are generative pre-trained transformers designed for text generation tasks.",
            "Fine-tuning allows pre-trained models to be adapted for specific downstream tasks.",
            "Parameter-efficient fine-tuning methods like LoRA reduce the number of trainable parameters.",
            "Retrieval-augmented generation combines parametric and non-parametric knowledge."
        ]
        
        # Step 3: Build vector index
        print("\n🔍 Building vector index...")
        index_builder = VectorIndexBuilder(
            encoder="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=256,
            device=device
        )
        
        # Build index from documents
        index_data = index_builder.build_index(
            documents=sample_documents,
            batch_size=4
        )
        
        print(f"✅ Built index with {index_data['num_chunks']} chunks")
        
        # Step 4: Create FAISS retriever
        print("\n🔎 Setting up retriever...")
        retriever = FAISSRetriever(
            embedding_dim=index_data['embedding_dim'],
            encoder=index_builder.encoder
        )
        
        # Add documents to retriever
        retriever.add_documents(
            embeddings=index_data['embeddings'],
            metadata=index_data['chunks']
        )
        
        print("✅ Retriever ready")
        
        # Step 5: Create RetroLoRA adapter
        print("\n🔧 Creating RetroLoRA adapter...")
        retro_lora = RetroLoRA(
            base_model=base_model,
            r=8,  # Small rank for demonstration
            alpha=16,
            target_modules=["c_attn"],  # DialoGPT uses c_attn instead of q_proj/v_proj
            retrieval_dim=384,  # Match encoder dimension
            fusion_method="gated"
        )
        
        # Set retriever
        retro_lora.set_retriever(retriever)
        
        print("✅ RetroLoRA adapter created")
        
        # Step 6: Create inference pipeline
        print("\n⚡ Setting up inference pipeline...")
        pipeline = RetrievalInferencePipeline(
            model=retro_lora,
            retriever=retriever,
            tokenizer=tokenizer,
            device=device,
            max_length=100,  # Short responses for demo
            temperature=0.7
        )
        
        print("✅ Inference pipeline ready")
        
        # Step 7: Test inference
        print("\n🧪 Testing inference...")
        
        test_queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain transformers in NLP",
            "What is parameter-efficient fine-tuning?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            # Generate without retrieval
            result_no_retrieval = pipeline.generate(
                prompt=query,
                retrieval_enabled=False,
                max_length=80
            )
            
            print(f"Without retrieval: {result_no_retrieval['generated_text']}")
            
            # Generate with retrieval
            result_with_retrieval = pipeline.generate(
                prompt=query,
                retrieval_enabled=True,
                retrieval_k=3,
                return_retrieval_info=True,
                max_length=80
            )
            
            print(f"With retrieval: {result_with_retrieval['generated_text']}")
            
            # Show retrieved documents
            if 'retrieval_info' in result_with_retrieval:
                print("📄 Retrieved documents:")
                for j, doc in enumerate(result_with_retrieval['retrieval_info']['retrieved_docs'][:2]):
                    score = doc.get('score', 0.0)
                    text = doc.get('text', '')[:100] + "..."
                    print(f"  {j+1}. (score: {score:.3f}) {text}")
        
        # Step 8: Show performance metrics
        print("\n📊 Performance Metrics:")
        metrics = pipeline.get_metrics()
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n🎉 Basic usage example completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)