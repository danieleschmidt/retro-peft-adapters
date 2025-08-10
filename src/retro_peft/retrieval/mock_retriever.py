"""
Mock retriever implementation for basic functionality without heavy dependencies.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = object()


class MockRetriever:
    """
    Simple mock retriever for testing and basic functionality.
    
    Returns predefined responses to demonstrate retrieval-augmented adaptation
    without requiring vector databases or sentence transformers.
    """
    
    def __init__(
        self,
        mock_documents: Optional[List[Dict[str, Any]]] = None,
        embedding_dim: int = 768,
    ):
        """
        Initialize mock retriever.
        
        Args:
            mock_documents: List of mock documents with text and metadata
            embedding_dim: Dimension of mock embeddings
        """
        self.embedding_dim = embedding_dim
        
        # Default mock documents if none provided
        if mock_documents is None:
            self.documents = [
                {
                    "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms that improve through experience.",
                    "metadata": {"source": "ml_intro", "topic": "machine_learning", "score": 0.95}
                },
                {
                    "text": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
                    "metadata": {"source": "dl_intro", "topic": "deep_learning", "score": 0.92}
                },
                {
                    "text": "Natural language processing combines computational linguistics with machine learning to help computers understand human language.",
                    "metadata": {"source": "nlp_intro", "topic": "nlp", "score": 0.88}
                },
                {
                    "text": "Parameter-efficient fine-tuning methods like LoRA allow adaptation of large models with minimal computational cost.",
                    "metadata": {"source": "peft_intro", "topic": "fine_tuning", "score": 0.93}
                },
                {
                    "text": "Retrieval-augmented generation combines language models with external knowledge retrieval for better factual accuracy.",
                    "metadata": {"source": "rag_intro", "topic": "retrieval", "score": 0.90}
                }
            ]
        else:
            self.documents = mock_documents
        
        # Pre-generate mock embeddings for documents
        self.document_embeddings = self._generate_mock_embeddings(len(self.documents))
    
    def _generate_mock_embeddings(self, num_docs: int):
        """Generate mock embeddings for documents"""
        if not _TORCH_AVAILABLE:
            # Return None when torch is not available
            return None
        
        # Generate deterministic but realistic embeddings
        torch.manual_seed(42)
        embeddings = torch.randn(num_docs, self.embedding_dim)
        # Normalize to unit length
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings
    
    def retrieve(
        self,
        query_embeddings: Optional["torch.Tensor"] = None,
        query_text: Optional[List[str]] = None,
        k: int = 5,
    ) -> Tuple["torch.Tensor", List[Dict[str, Any]]]:
        """
        Mock retrieval that returns predefined documents.
        
        Args:
            query_embeddings: Query embeddings tensor [batch_size, hidden_dim]
            query_text: Optional list of query strings
            k: Number of documents to retrieve
        
        Returns:
            Tuple of (context_embeddings, metadata)
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for retrieval functionality")
        
        # Determine batch size
        if query_embeddings is not None:
            batch_size = query_embeddings.size(0)
        else:
            batch_size = len(query_text) if query_text else 1
        
        # Limit k to available documents
        k = min(k, len(self.documents))
        
        # Return top-k documents (mock - just return first k)
        retrieved_docs = self.documents[:k]
        
        # Create mock context embeddings
        if self.document_embeddings is not None:
            # Use pre-generated embeddings
            context_embeddings = self.document_embeddings[:k].unsqueeze(0).expand(
                batch_size, -1, -1
            )
        else:
            # Create zeros if no torch
            context_embeddings = torch.zeros(batch_size, k, self.embedding_dim)
        
        # Prepare metadata for each batch item
        metadata = []
        for _ in range(batch_size):
            metadata.append(retrieved_docs)
        
        return context_embeddings, metadata
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add new documents to the mock retriever"""
        self.documents.extend(documents)
        
        if _TORCH_AVAILABLE:
            # Regenerate embeddings to include new documents
            self.document_embeddings = self._generate_mock_embeddings(len(self.documents))
    
    def save_index(self, save_path: str):
        """Save mock retriever state"""
        state = {
            "documents": self.documents,
            "embedding_dim": self.embedding_dim,
        }
        
        with open(save_path, "w") as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_index(cls, load_path: str):
        """Load mock retriever from saved state"""
        with open(load_path, "r") as f:
            state = json.load(f)
        
        return cls(
            mock_documents=state["documents"],
            embedding_dim=state["embedding_dim"]
        )
    
    def get_document_count(self) -> int:
        """Get number of documents in retriever"""
        return len(self.documents)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Simple text search in document contents.
        
        Args:
            query: Search query text
            k: Number of results to return
        
        Returns:
            List of matching documents with scores
        """
        results = []
        query_words = query.lower().split()
        
        for doc in self.documents:
            # Simple text matching score
            doc_text = doc["text"].lower()
            doc_words = doc_text.split()
            
            # Calculate word overlap score
            matches = 0
            for query_word in query_words:
                for doc_word in doc_words:
                    if query_word in doc_word or doc_word in query_word:
                        matches += 1
                        break
            
            if matches > 0:
                # Score based on word overlap ratio
                score = matches / len(query_words)
                results.append({
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "score": score
                })
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]