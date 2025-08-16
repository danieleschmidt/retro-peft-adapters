"""
Robust retrieval system with error handling and monitoring.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from ..robust_features import (
    RetrievalError, 
    RobustValidator, 
    monitored_operation, 
    resilient_operation,
    get_health_monitor
)
from .mock_retriever import MockRetriever

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = object()


class RobustMockRetriever(MockRetriever):
    """Enhanced MockRetriever with robust error handling and monitoring"""
    
    def __init__(self, *args, **kwargs):
        # Extract robust-specific parameters
        self.enable_caching = kwargs.pop('enable_caching', True)
        
        # Initialize base class
        super().__init__(*args, **kwargs)
        
        # Initialize robust features
        self.health_monitor = get_health_monitor()
        self.query_cache = {}
        self.cache_max_size = 1000
    
    @monitored_operation("retrieval.search")
    @resilient_operation(max_retries=3)
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced search with validation, caching, and monitoring"""
        # Validate inputs with special handling for empty queries
        if not query or len(query.strip()) == 0:
            print("Warning: Empty query provided, returning empty results")
            return []
        
        query = RobustValidator.validate_text_input(query, max_length=1000, min_length=1)
        validated_params = RobustValidator.validate_model_params({'k': k})
        k = validated_params['k']
        
        # Check cache first
        cache_key = f"{query}:{k}"
        if self.enable_caching and cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        try:
            # Perform search
            results = super().search(query, k)
            
            # Validate results
            if not isinstance(results, list):
                raise RetrievalError("Search results must be a list")
            
            for result in results:
                if not isinstance(result, dict) or 'text' not in result:
                    raise RetrievalError("Invalid result format")
            
            # Cache results
            if self.enable_caching:
                self._cache_result(cache_key, results)
            
            return results
            
        except Exception as e:
            raise RetrievalError(f"Search failed: {e}")
    
    @monitored_operation("retrieval.retrieve")
    @resilient_operation(max_retries=2)
    def retrieve(
        self,
        query_embeddings: Optional["torch.Tensor"] = None,
        query_text: Optional[List[str]] = None,
        k: int = 5,
    ) -> Tuple["torch.Tensor", List[Dict[str, Any]]]:
        """Enhanced retrieve with monitoring and error handling"""
        if not _TORCH_AVAILABLE:
            raise RetrievalError("PyTorch is required for tensor retrieval")
        
        # Validate parameters
        validated_params = RobustValidator.validate_model_params({'k': k})
        k = validated_params['k']
        
        if query_text:
            for text in query_text:
                RobustValidator.validate_text_input(text, max_length=500)
        
        try:
            return super().retrieve(query_embeddings, query_text, k)
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {e}")
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache search result with size management"""
        if len(self.query_cache) >= self.cache_max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = result
    
    @monitored_operation("retrieval.add_documents")
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Enhanced document addition with validation"""
        if not isinstance(documents, list):
            raise RetrievalError("Documents must be a list")
        
        for doc in documents:
            if not isinstance(doc, dict) or 'text' not in doc:
                raise RetrievalError("Each document must be a dict with 'text' field")
            
            # Validate document text
            RobustValidator.validate_text_input(doc['text'], max_length=10000)
        
        try:
            super().add_documents(documents)
        except Exception as e:
            raise RetrievalError(f"Failed to add documents: {e}")
    
    @monitored_operation("retrieval.save_index")
    def save_index(self, save_path: str):
        """Enhanced index saving with validation"""
        save_path = RobustValidator.validate_file_path(save_path)
        
        try:
            super().save_index(save_path)
        except Exception as e:
            raise RetrievalError(f"Failed to save index: {e}")
    
    @classmethod
    @monitored_operation("retrieval.load_index")
    def load_index(cls, load_path: str):
        """Enhanced index loading with validation"""
        load_path = RobustValidator.validate_file_path(load_path)
        
        try:
            return super().load_index(load_path)
        except Exception as e:
            raise RetrievalError(f"Failed to load index: {e}")
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get retrieval-specific health metrics"""
        return {
            'document_count': self.get_document_count(),
            'cache_size': len(self.query_cache),
            'cache_enabled': self.enable_caching,
            'performance_summary': self.health_monitor.get_performance_summary('retrieval.search')
        }


class RobustVectorIndexBuilder:
    """Enhanced VectorIndexBuilder with comprehensive error handling"""
    
    def __init__(self, embedding_dim: int = 768, chunk_size: int = 512, overlap: int = 50):
        # Validate parameters
        if embedding_dim <= 0 or embedding_dim > 4096:
            raise RetrievalError(f"Invalid embedding_dim: {embedding_dim}")
        if chunk_size <= 0 or chunk_size > 2048:
            raise RetrievalError(f"Invalid chunk_size: {chunk_size}")
        if overlap < 0 or overlap >= chunk_size:
            raise RetrievalError(f"Invalid overlap: {overlap}")
        
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.health_monitor = get_health_monitor()
    
    @monitored_operation("index_builder.chunk_text")
    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Enhanced text chunking with validation"""
        # Validate input
        text = RobustValidator.validate_text_input(text, max_length=50000)
        
        if metadata and not isinstance(metadata, dict):
            raise RetrievalError("Metadata must be a dictionary")
        
        try:
            words = text.split()
            chunks = []
            
            if not words:
                return []
            
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
            
        except Exception as e:
            raise RetrievalError(f"Text chunking failed: {e}")
    
    @monitored_operation("index_builder.build_index")
    @resilient_operation(max_retries=2)
    def build_index(
        self,
        documents: List[Any],
        output_path: Optional[str] = None,
    ) -> RobustMockRetriever:
        """Enhanced index building with comprehensive error handling"""
        if not isinstance(documents, list):
            raise RetrievalError("Documents must be a list")
        
        if len(documents) == 0:
            raise RetrievalError("Cannot build index from empty document list")
        
        if output_path:
            output_path = RobustValidator.validate_file_path(output_path)
        
        try:
            print(f"Building robust index from {len(documents)} documents...")
            
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
                
                # Chunk document with error handling
                try:
                    chunks = self.chunk_text(text, doc_metadata)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Warning: Failed to chunk document {doc_idx}: {e}")
                    continue
            
            if not all_chunks:
                raise RetrievalError("No valid chunks created from documents")
            
            print(f"Created {len(all_chunks)} chunks")
            
            # Convert chunks to retriever format
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
            
            # Create robust retriever
            retriever = RobustMockRetriever(
                mock_documents=retriever_docs,
                embedding_dim=self.embedding_dim,
                enable_caching=True
            )
            
            # Save index if path provided
            if output_path:
                retriever.save_index(output_path)
                print(f"Index saved to: {output_path}")
            
            return retriever
            
        except Exception as e:
            raise RetrievalError(f"Index building failed: {e}")
    
    @monitored_operation("index_builder.load_index")
    def load_index(self, load_path: str) -> RobustMockRetriever:
        """Enhanced index loading"""
        load_path = RobustValidator.validate_file_path(load_path)
        
        try:
            base_retriever = RobustMockRetriever.load_index(load_path)
            
            # Enhance with robust features
            robust_retriever = RobustMockRetriever(
                mock_documents=base_retriever.documents,
                embedding_dim=base_retriever.embedding_dim
            )
            
            return robust_retriever
            
        except Exception as e:
            raise RetrievalError(f"Failed to load index: {e}")
    
    def create_sample_documents(self) -> List[Dict[str, Any]]:
        """Create validated sample documents"""
        documents = [
            {
                "text": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
                "metadata": {"topic": "machine_learning", "difficulty": "beginner", "validated": True}
            },
            {
                "text": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
                "metadata": {"topic": "deep_learning", "difficulty": "intermediate", "validated": True}
            },
            {
                "text": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
                "metadata": {"topic": "nlp", "difficulty": "intermediate", "validated": True}
            },
            {
                "text": "Parameter-efficient fine-tuning (PEFT) is a set of techniques that allows for the adaptation of pre-trained models to specific tasks or domains while minimizing the number of parameters that need to be updated. This approach is particularly useful for large language models.",
                "metadata": {"topic": "fine_tuning", "difficulty": "advanced", "validated": True}
            },
            {
                "text": "Retrieval-augmented generation (RAG) is an AI framework for improving the quality of LLM-generated responses by grounding the model on external sources of knowledge to supplement the LLM's internal representation of information.",
                "metadata": {"topic": "rag", "difficulty": "advanced", "validated": True}
            }
        ]
        
        # Validate all documents
        for doc in documents:
            RobustValidator.validate_text_input(doc["text"])
        
        return documents


# Export robust retrieval components
__all__ = [
    'RobustMockRetriever',
    'RobustVectorIndexBuilder',
    'RetrievalError'
]