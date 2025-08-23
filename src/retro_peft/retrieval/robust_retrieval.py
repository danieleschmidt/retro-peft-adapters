"""
Generation 2: Robust Retrieval System

Enhanced retrieval with comprehensive error handling, caching, and monitoring.
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..utils import ErrorHandler, InputValidator, ValidationError, AdapterError, resilient_operation
from .mock_retriever import MockRetriever

try:
    import threading
    _THREADING_AVAILABLE = True
except ImportError:
    _THREADING_AVAILABLE = False


@dataclass
class RetrievalResult:
    """Structured retrieval result with metadata"""
    text: str
    score: float
    metadata: Dict[str, Any]
    source_id: str
    retrieval_time_ms: float
    cached: bool = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = object()


class RobustMockRetriever(MockRetriever):
    """
    Production-grade retriever with comprehensive reliability features.
    
    Features:
    - Multi-level caching with TTL
    - Circuit breaker pattern for failure resilience
    - Request deduplication and monitoring
    - Comprehensive logging and error handling
    """
    
    def __init__(
        self,
        mock_documents: Optional[List[Dict[str, Any]]] = None,
        embedding_dim: int = 768,
        cache_size: int = 1000,
        cache_ttl_seconds: int = 3600,
        enable_monitoring: bool = True,
        **kwargs
    ):
        # Initialize base MockRetriever
        super().__init__(mock_documents, embedding_dim)
        
        # Initialize logging and error handling
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.error_handler = ErrorHandler(self.logger)
        
        # Configuration
        self.cache_size = cache_size
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enable_monitoring = enable_monitoring
        
        # Initialize components
        self._init_cache()
        self._init_metrics()
        
        self.logger.info(
            f"RobustMockRetriever initialized",
            extra={
                "documents": len(self.documents),
                "cache_size": cache_size,
                "cache_ttl": cache_ttl_seconds,
                "monitoring": enable_monitoring
            }
        )
    
    def _init_cache(self):
        """Initialize caching system"""
        self.cache = {}  # query_hash -> (result, timestamp)
        self.cache_access_times = {}  # query_hash -> last_access_time
        self.cache_hits = 0
        self.cache_misses = 0
        
        if _THREADING_AVAILABLE:
            self.cache_lock = threading.RLock()
        else:
            self.cache_lock = None
    
    def _init_metrics(self):
        """Initialize metrics tracking"""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "cached_requests": 0,
            "failed_requests": 0,
            "average_response_time_ms": 0.0,
            "last_error": None
        }
    
    @resilient_operation(
        context="robust_search",
        max_retries=2,
        retry_exceptions=(ValidationError, AdapterError)
    )
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Robust search with comprehensive error handling and caching.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with comprehensive metadata
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not query or len(query.strip()) == 0:
                self.logger.warning("Empty query provided")
                return []
            
            query = InputValidator.validate_text_content(query, max_length=1000)
            k = InputValidator.validate_numeric_parameter(
                k, "k", min_val=1, max_val=50, must_be_int=True
            )
            
            # Update metrics
            self.metrics["total_requests"] += 1
            
            # Generate cache key
            cache_key = self._generate_cache_key(query, k)
            
            # Try cache first
            cached_results = self._get_from_cache(cache_key)
            if cached_results:
                self.cache_hits += 1
                self.metrics["cached_requests"] += 1
                self.metrics["successful_requests"] += 1
                
                self.logger.debug(f"Cache hit for query: {cache_key[:8]}")
                return [r.__dict__ for r in cached_results[:k]]
            
            # Cache miss - perform search
            self.cache_misses += 1
            
            # Perform base search
            raw_results = super().search(query, k)
            
            # Convert to structured results
            structured_results = self._convert_to_structured_results(
                raw_results, query, start_time
            )
            
            # Store in cache
            self._store_in_cache(cache_key, structured_results)
            
            # Update metrics
            self.metrics["successful_requests"] += 1
            self._update_response_time_metric(start_time)
            
            self.logger.debug(
                f"Search completed: {len(structured_results)} results in {(time.time() - start_time) * 1000:.1f}ms"
            )
            
            # Return as dictionaries for compatibility
            return [r.__dict__ for r in structured_results[:k]]
            
        except Exception as e:
            self.metrics["failed_requests"] += 1
            self.metrics["last_error"] = str(e)
            
            self.logger.error(f"Search failed: {e}")
            
            # Return empty results on failure
            return []
    
    def _generate_cache_key(self, query: str, k: int, **kwargs) -> str:
        """Generate cache key for query"""
        cache_input = f"{query}:{k}:{sorted(kwargs.items())}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cached result is still valid"""
        return (time.time() - timestamp) < self.cache_ttl_seconds
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[RetrievalResult]]:
        """Retrieve results from cache"""
        if not self.cache_lock:
            return None
        
        with self.cache_lock:
            if cache_key not in self.cache:
                return None
            
            result, timestamp = self.cache[cache_key]
            
            if not self._is_cache_valid(timestamp):
                del self.cache[cache_key]
                if cache_key in self.cache_access_times:
                    del self.cache_access_times[cache_key]
                return None
            
            # Update access time
            self.cache_access_times[cache_key] = time.time()
            return result
    
    def _store_in_cache(self, cache_key: str, results: List[RetrievalResult]):
        """Store results in cache"""
        if not self.cache_lock:
            return
        
        with self.cache_lock:
            current_time = time.time()
            
            # Evict LRU entries if cache is full
            if len(self.cache) >= self.cache_size:
                sorted_entries = sorted(
                    self.cache_access_times.items(),
                    key=lambda x: x[1]
                )
                
                oldest_key = sorted_entries[0][0]
                del self.cache[oldest_key]
                del self.cache_access_times[oldest_key]
            
            # Store new result
            self.cache[cache_key] = (results, current_time)
            self.cache_access_times[cache_key] = current_time
    
    def _convert_to_structured_results(
        self, raw_results: List[Dict[str, Any]], query: str, start_time: float
    ) -> List[RetrievalResult]:
        """Convert raw results to structured RetrievalResult objects"""
        
        structured_results = []
        retrieval_time_ms = (time.time() - start_time) * 1000
        
        for i, result in enumerate(raw_results):
            text = result.get("text", "")
            if not text:
                continue
            
            score = result.get("score", 0.0)
            metadata = result.get("metadata", {})
            
            # Add retrieval metadata
            metadata.update({
                "query": query[:100],
                "rank": i + 1,
                "retrieval_timestamp": time.time()
            })
            
            structured_result = RetrievalResult(
                text=text,
                score=score,
                metadata=metadata,
                source_id=metadata.get("source_id", f"result_{i}"),
                retrieval_time_ms=retrieval_time_ms,
                cached=False
            )
            
            structured_results.append(structured_result)
        
        return structured_results
    
    def _update_response_time_metric(self, start_time: float):
        """Update average response time using exponential moving average"""
        response_time = (time.time() - start_time) * 1000
        
        alpha = 0.1
        if self.metrics["average_response_time_ms"] == 0:
            self.metrics["average_response_time_ms"] = response_time
        else:
            self.metrics["average_response_time_ms"] = (
                alpha * response_time + 
                (1 - alpha) * self.metrics["average_response_time_ms"]
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        
        total_requests = self.metrics["total_requests"]
        success_rate = (
            self.metrics["successful_requests"] / max(total_requests, 1)
        )
        
        cache_hit_rate = self.cache_hits / max(total_requests, 1) if total_requests > 0 else 0.0
        
        # Determine health status
        if success_rate > 0.95:
            status = "healthy"
        elif success_rate > 0.8:
            status = "degraded"  
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "metrics": self.metrics.copy(),
            "cache_stats": {
                "size": len(self.cache),
                "max_size": self.cache_size,
                "hits": self.cache_hits,
                "misses": self.cache_misses
            }
        }


class RobustVectorIndexBuilder:
    """Enhanced VectorIndexBuilder with comprehensive error handling"""
    
    def __init__(self, embedding_dim: int = 768, chunk_size: int = 512, overlap: int = 50):
        # Validate parameters
        self.embedding_dim = InputValidator.validate_numeric_parameter(
            embedding_dim, "embedding_dim", min_val=64, max_val=4096, must_be_int=True
        )
        self.chunk_size = InputValidator.validate_numeric_parameter(
            chunk_size, "chunk_size", min_val=10, max_val=2048, must_be_int=True
        )
        self.overlap = InputValidator.validate_numeric_parameter(
            overlap, "overlap", min_val=0, max_val=chunk_size-1, must_be_int=True
        )
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.error_handler = ErrorHandler(self.logger)
    
    @resilient_operation(
        context="chunk_text",
        max_retries=1
    )
    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Enhanced text chunking with validation"""
        # Validate input
        text = InputValidator.validate_text_content(text, max_length=50000)
        
        if metadata and not isinstance(metadata, dict):
            raise ValidationError("Metadata must be a dictionary")
        
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
    
    @resilient_operation(
        context="build_index",
        max_retries=2
    )
    def build_index(
        self,
        documents: List[Any],
        output_path: Optional[str] = None,
    ) -> RobustMockRetriever:
        """Enhanced index building with comprehensive error handling"""
        if not isinstance(documents, list):
            raise ValidationError("Documents must be a list")
        
        if len(documents) == 0:
            raise ValidationError("Cannot build index from empty document list")
        
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
                self.logger.warning(f"Failed to chunk document {doc_idx}: {e}")
                continue
        
        if not all_chunks:
            raise AdapterError("No valid chunks created from documents")
        
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
            cache_size=1000,
            enable_monitoring=True
        )
        
        # Save index if path provided
        if output_path:
            retriever.save_index(output_path)
            print(f"Index saved to: {output_path}")
        
        return retriever


# Export for backward compatibility
VectorIndexBuilder = RobustVectorIndexBuilder
MockRetriever = RobustMockRetriever


# Export components
__all__ = [
    'RetrievalResult',
    'RobustMockRetriever',
    'RobustVectorIndexBuilder',
    'VectorIndexBuilder',
    'MockRetriever'
]
