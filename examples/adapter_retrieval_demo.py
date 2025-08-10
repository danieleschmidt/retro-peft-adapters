#!/usr/bin/env python3
"""
Demonstration of adapter-retrieval integration.

Shows how RetroLoRA adapters can be enhanced with retrieval augmentation.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def main():
    """Run adapter-retrieval integration demo"""
    print("🔗 Retro-PEFT Adapter-Retrieval Integration Demo")
    print("=" * 55)
    
    # Import components
    from retro_peft.adapters import RetroLoRA
    from retro_peft.retrieval import VectorIndexBuilder, MockRetriever
    
    # Create a domain-specific knowledge base
    domain_documents = [
        {
            "text": "LoRA (Low-Rank Adaptation) reduces the number of trainable parameters by learning pairs of rank-decomposition matrices while keeping the original model frozen.",
            "metadata": {"topic": "lora", "source": "research_paper"}
        },
        {
            "text": "Parameter-efficient fine-tuning methods like LoRA, AdaLoRA, and IA³ enable adaptation of large language models with minimal computational overhead.",
            "metadata": {"topic": "peft", "source": "survey_paper"}
        },
        {
            "text": "Retrieval-augmented generation (RAG) improves language model outputs by incorporating relevant information from external knowledge bases.",
            "metadata": {"topic": "rag", "source": "technical_blog"}
        },
        {
            "text": "The scaling parameter alpha in LoRA controls the magnitude of the adaptation, typically set to 2-4 times the rank value.",
            "metadata": {"topic": "lora_hyperparameters", "source": "implementation_guide"}
        }
    ]
    
    print("\n1. Building knowledge base...")
    # Build retrieval index
    builder = VectorIndexBuilder(embedding_dim=512, chunk_size=30)
    retriever = builder.build_index(domain_documents)
    print(f"✅ Built knowledge base with {retriever.get_document_count()} documents")
    
    print("\n2. Creating RetroLoRA adapter...")
    # Create adapter
    adapter = RetroLoRA(
        model_name="llama-7b-chat",
        rank=16,
        alpha=32.0
    )
    
    # Connect retriever to adapter
    adapter.set_retriever(retriever)
    print(f"✅ Created RetroLoRA (rank={adapter.rank}, α={adapter.alpha})")
    print(f"✅ Connected retriever with {adapter.retriever.get_document_count()} documents")
    
    print("\n3. Testing retrieval-augmented generation...")
    
    # Test queries
    test_queries = [
        "What is LoRA and how does it work?",
        "How should I set the alpha parameter?",
        "What are the benefits of parameter-efficient fine-tuning?",
        "How does retrieval help language models?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        
        # Generate response with retrieval
        response = adapter.generate(query)
        
        print(f"   Response: {response['output']}")
        print(f"   Context used: {response['context_used']}")
        print(f"   Sources: {response['context_sources']}")
    
    print("\n4. Testing adapter training interface...")
    
    # Mock training step
    mock_batch = {
        "input": "Explain LoRA parameters",
        "target": "LoRA uses rank and alpha parameters..."
    }
    
    training_result = adapter.train_step(mock_batch)
    print(f"✅ Training step completed:")
    print(f"   Total loss: {training_result['loss']:.4f}")
    print(f"   Retrieval loss: {training_result['retrieval_loss']:.4f}")
    print(f"   Generation loss: {training_result['generation_loss']:.4f}")
    
    print("\n5. Adapter statistics...")
    trainable_params = adapter.get_trainable_parameters()
    print(f"✅ Trainable parameters: {trainable_params:,}")
    
    # Efficiency calculation
    base_model_params = 7_000_000_000  # 7B model
    efficiency = (trainable_params / base_model_params) * 100
    print(f"✅ Parameter efficiency: {efficiency:.4f}% of base model")
    
    print("\n" + "=" * 55)
    print("✅ Adapter-Retrieval Integration Demo Complete!")
    
    print("\n🎯 Key Features Demonstrated:")
    print("   • Knowledge base creation from domain documents")
    print("   • RetroLoRA adapter with configurable parameters")
    print("   • Retrieval-augmented text generation")
    print("   • Context-aware response generation")
    print("   • Training interface with retrieval supervision")
    print("   • Parameter efficiency reporting")
    
    print("\n🚀 Production Readiness:")
    print("   • Add PyTorch for full neural network functionality")
    print("   • Replace MockRetriever with FAISS/Qdrant for scale")
    print("   • Integrate with HuggingFace transformers for real models")
    print("   • Add more sophisticated retrieval ranking")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)