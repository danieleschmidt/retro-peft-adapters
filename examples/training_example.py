"""
Training example for retro-peft-adapters.

This example demonstrates how to:
1. Prepare training data
2. Set up contrastive retrieval training
3. Train a RetroLoRA adapter
4. Evaluate the trained model

Run with: python examples/training_example.py
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
from retro_peft.training import ContrastiveRetrievalTrainer, ContrastiveRetrievalDataset


def create_sample_training_data():
    """Create sample training data for demonstration"""
    
    # Sample queries and documents (in practice, these would be much larger)
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain neural networks",
        "What are transformers in AI?",
        "How does natural language processing work?",
        "What is computer vision?",
        "Explain reinforcement learning",
        "What is supervised learning?",
        "How do you train a neural network?",
        "What is the difference between AI and ML?"
    ]
    
    documents = [
        "Machine learning is a method of data analysis that automates analytical model building. It uses algorithms that iteratively learn from data to improve performance on a specific task.",
        "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
        "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information through weighted connections.",
        "Transformers are a neural network architecture that relies entirely on attention mechanisms to draw global dependencies between input and output sequences.",
        "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language.",
        "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world from digital images or videos.",
        "Reinforcement learning is an area of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward.",
        "Supervised learning is a machine learning approach where models are trained on labeled datasets to make predictions or classifications on new, unseen data.",
        "Training a neural network involves feeding it data, calculating the error in predictions, and adjusting weights through backpropagation to minimize this error.",
        "Artificial Intelligence (AI) is the broader concept of machines performing tasks intelligently, while Machine Learning (ML) is a specific approach to achieve AI through data-driven learning."
    ]
    
    # Create positive query-document pairs (simplified mapping)
    query_doc_pairs = [(i, i) for i in range(len(queries))]
    
    return queries, documents, query_doc_pairs


def main():
    """Main training example function"""
    print("🚀 Retro-PEFT-Adapters Training Example")
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
        
        # Step 2: Create training data
        print("\n📚 Preparing training data...")
        queries, documents, query_doc_pairs = create_sample_training_data()
        
        print(f"✅ Created {len(queries)} queries, {len(documents)} documents, {len(query_doc_pairs)} pairs")
        
        # Step 3: Build vector index for retrieval
        print("\n🔍 Building vector index...")
        index_builder = VectorIndexBuilder(
            encoder="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=256,
            device=device
        )
        
        # Build index from documents
        index_data = index_builder.build_index(
            documents=documents,
            batch_size=4
        )
        
        print(f"✅ Built index with {index_data['num_chunks']} chunks")
        
        # Step 4: Create retriever
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
            target_modules=["c_attn"],  # DialoGPT uses c_attn
            retrieval_dim=384,  # Match encoder dimension
            fusion_method="gated"
        )
        
        # Set retriever
        retro_lora.set_retriever(retriever)
        
        print("✅ RetroLoRA adapter created")
        print(f"Trainable parameters: {retro_lora.get_trainable_parameters():,}")
        
        # Step 6: Create contrastive training dataset
        print("\n📊 Creating training dataset...")
        train_dataset = ContrastiveRetrievalDataset(
            queries=queries,
            documents=documents,
            query_doc_pairs=query_doc_pairs,
            tokenizer=tokenizer,
            max_length=256,
            negative_sampling_ratio=2.0,
            hard_negative_ratio=0.5
        )
        
        print(f"✅ Created dataset with {len(train_dataset)} examples")
        
        # Step 7: Set up trainer
        print("\n🏋️ Setting up contrastive trainer...")
        trainer = ContrastiveRetrievalTrainer(
            model=retro_lora,
            temperature=0.07,
            in_batch_negatives=True,
            loss_type="contrastive"
        )
        
        print("✅ Trainer ready")
        
        # Step 8: Train the model (minimal training for demo)
        print("\n🚂 Starting training...")
        print("Note: This is a minimal training demo. Real training would use more data and epochs.")
        
        trainer.train(
            dataset=train_dataset,
            num_epochs=1,  # Very minimal for demo
            batch_size=2,  # Small batch size for demo
            learning_rate=5e-5,
            warmup_steps=10,
            hard_negative_mining=False,  # Disable for speed
            eval_steps=5,
            save_steps=10,
            output_dir="./demo_checkpoints"
        )
        
        print("✅ Training completed")
        
        # Step 9: Test the trained model
        print("\n🧪 Testing trained model...")
        
        test_queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain transformers in NLP"
        ]
        
        # Put model in eval mode
        retro_lora.eval()
        
        with torch.no_grad():
            for i, query in enumerate(test_queries, 1):
                print(f"\n--- Test Query {i}: {query} ---")
                
                # Tokenize query
                inputs = tokenizer(
                    query,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(device)
                
                # Generate response (simplified)
                try:
                    outputs = retro_lora(**inputs)
                    print(f"✅ Model processed query successfully")
                    
                    # Test retrieval
                    if retriever:
                        retrieval_context, metadata = retriever.retrieve(
                            query_text=[query],
                            k=3
                        )
                        print(f"Retrieved {len(metadata)} relevant documents")
                        
                        # Show top retrieved document
                        if metadata:
                            top_doc = metadata[0]
                            score = top_doc.get('score', 0.0)
                            text = top_doc.get('text', '')[:100] + "..."
                            print(f"Top document (score: {score:.3f}): {text}")
                    
                except Exception as e:
                    print(f"⚠️ Error during inference: {e}")
        
        # Step 10: Show final statistics
        print("\n📊 Training Summary:")
        print(f"Base model: {model_name}")
        print(f"Adapter type: RetroLoRA")
        print(f"Trainable parameters: {retro_lora.get_trainable_parameters():,}")
        print(f"Training queries: {len(queries)}")
        print(f"Knowledge base: {len(documents)} documents")
        print(f"Vector index: {index_data['num_chunks']} chunks")
        
        print("\n🎉 Training example completed successfully!")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        import shutil
        if os.path.exists("./demo_checkpoints"):
            shutil.rmtree("./demo_checkpoints")
            print("✅ Cleaned up temporary files")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)