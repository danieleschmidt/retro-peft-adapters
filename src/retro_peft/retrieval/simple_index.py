"""
Simple vector index builder without heavy dependencies.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = object()

from .mock_retriever import MockRetriever


class SimpleVectorIndexBuilder:
    """
    Lightweight vector index builder that works without heavy dependencies.
    
    Uses simple text processing and mock embeddings for development and testing.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        chunk_size: int = 100,  # Number of words per chunk
        overlap: int = 20,
    ):
        """
        Initialize simple vector index builder.
        
        Args:
            embedding_dim: Dimension of mock embeddings
            chunk_size: Maximum words per chunk
            overlap: Overlap words between chunks
        """
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks using simple word splitting.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks
        
        Returns:
            List of chunks with text and metadata
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk_data = {
                "text": chunk_text,
                "start_word": i,
                "end_word": min(i + self.chunk_size, len(words)),
                "metadata": metadata or {},
            }
            chunks.append(chunk_data)
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def build_index(
        self,
        documents: List[Union[str, Dict[str, Any]]],
        output_path: Optional[str] = None,
    ) -> MockRetriever:
        """
        Build a simple index from documents.
        
        Args:
            documents: List of documents (strings or dicts with 'text' field)
            output_path: Optional path to save the index
        
        Returns:
            MockRetriever instance with the indexed documents
        """
        print(f"Building simple index from {len(documents)} documents...")
        
        # Process documents into chunks
        all_chunks = []
        for doc_idx, doc in enumerate(documents):
            if isinstance(doc, str):
                text = doc
                doc_metadata = {"doc_id": doc_idx, "source": f"doc_{doc_idx}"}
            else:
                text = doc.get("text", str(doc))
                doc_metadata = doc.get("metadata", {})
                doc_metadata["doc_id"] = doc_idx
            
            # Chunk document
            chunks = self.chunk_text(text, doc_metadata)
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks")
        
        # Convert chunks to MockRetriever format
        retriever_docs = []
        for chunk in all_chunks:
            doc_entry = {
                "text": chunk["text"],
                "metadata": {
                    **chunk["metadata"],
                    "start_word": chunk["start_word"],
                    "end_word": chunk["end_word"],
                }
            }
            retriever_docs.append(doc_entry)
        
        # Create retriever
        retriever = MockRetriever(
            mock_documents=retriever_docs,
            embedding_dim=self.embedding_dim
        )
        
        # Save index if path provided
        if output_path:
            retriever.save_index(output_path)
            print(f"Index saved to: {output_path}")
        
        return retriever
    
    def load_index(self, load_path: str) -> MockRetriever:
        """Load index from saved file"""
        return MockRetriever.load_index(load_path)
    
    def create_sample_documents(self) -> List[Dict[str, Any]]:
        """Create sample documents for testing"""
        return [
            {
                "text": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
                "metadata": {"topic": "machine_learning", "difficulty": "beginner"}
            },
            {
                "text": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
                "metadata": {"topic": "deep_learning", "difficulty": "intermediate"}
            },
            {
                "text": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
                "metadata": {"topic": "nlp", "difficulty": "intermediate"}
            },
            {
                "text": "Parameter-efficient fine-tuning (PEFT) is a set of techniques that allows for the adaptation of pre-trained models to specific tasks or domains while minimizing the number of parameters that need to be updated. This approach is particularly useful for large language models.",
                "metadata": {"topic": "fine_tuning", "difficulty": "advanced"}
            },
            {
                "text": "Retrieval-augmented generation (RAG) is an AI framework for improving the quality of LLM-generated responses by grounding the model on external sources of knowledge to supplement the LLM's internal representation of information.",
                "metadata": {"topic": "rag", "difficulty": "advanced"}
            }
        ]


# Alias for backward compatibility
VectorIndexBuilder = SimpleVectorIndexBuilder