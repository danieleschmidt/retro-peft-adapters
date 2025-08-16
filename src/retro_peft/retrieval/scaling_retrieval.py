"""
Scaling-optimized retrieval system with performance enhancements.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from ..scaling_features import (
    HighPerformanceCache,
    AdaptiveResourcePool,
    LoadBalancer,
    AsyncBatchProcessor,
    performance_optimized,
    get_performance_optimizer
)
from ..robust_features import (
    RetrievalError,
    RobustValidator,
    monitored_operation,
    resilient_operation
)
from .robust_retrieval import RobustMockRetriever

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = object()


class ScalingMockRetriever(RobustMockRetriever):
    """High-performance retriever with advanced scaling features"""
    
    def __init__(self, *args, **kwargs):
        # Extract scaling-specific parameters
        self.enable_batch_processing = kwargs.pop('enable_batch_processing', True)
        self.batch_size = kwargs.pop('batch_size', 32)
        self.max_concurrent_searches = kwargs.pop('max_concurrent_searches', 10)
        self.enable_result_caching = kwargs.pop('enable_result_caching', True)
        
        super().__init__(*args, **kwargs)
        
        # Initialize scaling components
        self.performance_cache = HighPerformanceCache(
            max_size=10000,
            default_ttl=1800.0,  # 30 minutes
            max_memory_mb=100.0
        )
        
        # Thread pool for concurrent operations
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_concurrent_searches)
        
        # Batch processor for multiple searches
        if self.enable_batch_processing:
            self.batch_processor = AsyncBatchProcessor(
                processor_func=self._batch_search_impl,
                batch_size=self.batch_size,
                max_wait_time=0.05,  # 50ms batching window
                max_concurrent_batches=4
            )
        
        self.performance_optimizer = get_performance_optimizer()
    
    @performance_optimized(cache_ttl=900.0)  # 15 minutes cache
    @monitored_operation("scaling_retrieval.search")
    @resilient_operation(max_retries=3)
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """High-performance search with caching and optimization"""
        # Handle empty queries
        if not query or len(query.strip()) == 0:
            return []
        
        # Validate inputs
        query = RobustValidator.validate_text_input(query, max_length=1000, min_length=1)
        validated_params = RobustValidator.validate_model_params({'k': k})
        k = validated_params['k']
        
        # Check performance cache first
        cache_key = f"search:{query}:{k}"
        if self.enable_result_caching:
            cached_result = self.performance_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        try:
            # Perform optimized search
            results = self._optimized_search_impl(query, k)
            
            # Cache results
            if self.enable_result_caching:
                self.performance_cache.set(cache_key, results, ttl=1800.0)
            
            return results
            
        except Exception as e:
            raise RetrievalError(f"Optimized search failed: {e}")
    
    def _optimized_search_impl(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Optimized search implementation with parallel processing"""
        # Use parent's search for basic functionality but with optimizations
        start_time = time.time()
        
        # For large k values, use parallel processing
        if k > 10 and len(self.documents) > 100:
            results = self._parallel_search(query, k)
        else:
            results = super().search(query, k)
        
        duration = time.time() - start_time
        
        # Record performance metrics
        self.performance_optimizer.metrics['operation_times']['optimized_search'].append(duration)
        
        return results
    
    def _parallel_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Parallel search across document chunks"""
        # Split documents into chunks for parallel processing
        chunk_size = max(10, len(self.documents) // self.max_concurrent_searches)
        doc_chunks = [
            self.documents[i:i + chunk_size] 
            for i in range(0, len(self.documents), chunk_size)
        ]
        
        # Search each chunk in parallel
        futures = []
        for chunk in doc_chunks:
            future = self.thread_pool.submit(self._search_chunk, query, chunk, k)
            futures.append(future)
        
        # Collect and merge results
        all_results = []
        for future in futures:
            try:
                chunk_results = future.result(timeout=5.0)
                all_results.extend(chunk_results)
            except Exception:
                continue  # Skip failed chunks
        
        # Sort by score and return top k
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results[:k]
    
    def _search_chunk(self, query: str, documents: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Search within a document chunk"""
        results = []
        query_words = query.lower().split()
        
        for doc in documents:
            doc_text = doc["text"].lower()
            doc_words = doc_text.split()
            
            # Calculate relevance score
            matches = 0
            for query_word in query_words:
                for doc_word in doc_words:
                    if query_word in doc_word or doc_word in query_word:
                        matches += 1
                        break
            
            if matches > 0:
                score = matches / len(query_words)
                results.append({
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "score": score
                })
        
        # Return top results for this chunk
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
    
    async def batch_search(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        """Asynchronous batch search processing"""
        if not self.enable_batch_processing:
            # Fallback to sequential processing
            return [self.search(query, k) for query in queries]
        
        # Process queries in batches
        tasks = []
        for query in queries:
            task = self.batch_processor.process_item({'query': query, 'k': k})
            tasks.append(task)
        
        # Wait for all results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append([])  # Empty result for failed queries
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _batch_search_impl(self, batch_items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Batch search implementation"""
        results = []
        for item in batch_items:
            try:
                query = item['query']
                k = item['k']
                result = self._optimized_search_impl(query, k)
                results.append(result)
            except Exception:
                results.append([])  # Empty result for failed searches
        
        return results
    
    @monitored_operation("scaling_retrieval.add_documents")
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Enhanced document addition with cache invalidation"""
        super().add_documents(documents)
        
        # Invalidate search cache since document set changed
        if hasattr(self, 'performance_cache'):
            self.performance_cache.clear()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        base_metrics = self.get_health_metrics()
        
        performance_metrics = {
            'performance_cache_stats': self.performance_cache.get_stats(),
            'thread_pool_active_threads': self.thread_pool._threads.__len__() if hasattr(self.thread_pool, '_threads') else 0,
            'optimization_features': {
                'batch_processing_enabled': self.enable_batch_processing,
                'result_caching_enabled': self.enable_result_caching,
                'parallel_search_enabled': True,
                'batch_size': self.batch_size,
                'max_concurrent_searches': self.max_concurrent_searches
            }
        }
        
        return {**base_metrics, **performance_metrics}


class ScalingVectorIndexBuilder:
    """High-performance index builder with scaling optimizations"""
    
    def __init__(
        self, 
        embedding_dim: int = 768, 
        chunk_size: int = 512, 
        overlap: int = 50,
        enable_parallel_processing: bool = True,
        max_workers: int = 4
    ):
        # Validate parameters (reuse from robust version)
        if embedding_dim <= 0 or embedding_dim > 4096:
            raise RetrievalError(f"Invalid embedding_dim: {embedding_dim}")
        if chunk_size <= 0 or chunk_size > 2048:
            raise RetrievalError(f"Invalid chunk_size: {chunk_size}")
        if overlap < 0 or overlap >= chunk_size:
            raise RetrievalError(f"Invalid overlap: {overlap}")
        
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enable_parallel_processing = enable_parallel_processing
        self.max_workers = max_workers
        
        # Initialize performance components
        self.performance_optimizer = get_performance_optimizer()
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers) if enable_parallel_processing else None
    
    @performance_optimized(cache_ttl=3600.0)  # 1 hour cache
    @monitored_operation("scaling_index_builder.chunk_text")
    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """High-performance text chunking"""
        # Validate input
        text = RobustValidator.validate_text_input(text, max_length=100000)
        
        if metadata and not isinstance(metadata, dict):
            raise RetrievalError("Metadata must be a dictionary")
        
        # Use parallel processing for large texts
        if self.enable_parallel_processing and len(text) > 10000:
            return self._parallel_chunk_text(text, metadata)
        else:
            return self._sequential_chunk_text(text, metadata)
    
    def _sequential_chunk_text(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sequential text chunking"""
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
    
    def _parallel_chunk_text(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parallel text chunking for large texts"""
        words = text.split()
        if len(words) <= self.chunk_size:
            return self._sequential_chunk_text(text, metadata)
        
        # Split text into segments for parallel processing
        segment_size = max(1000, len(words) // self.max_workers)
        segments = []
        
        for i in range(0, len(words), segment_size):
            segment_words = words[i:i + segment_size + self.overlap]  # Add overlap for continuity
            segment_text = " ".join(segment_words)
            segments.append((segment_text, i, metadata))
        
        # Process segments in parallel
        futures = []
        for segment_text, start_offset, seg_metadata in segments:
            future = self.thread_pool.submit(
                self._chunk_text_segment, 
                segment_text, 
                start_offset, 
                seg_metadata
            )
            futures.append(future)
        
        # Collect results
        all_chunks = []
        for future in futures:
            try:
                segment_chunks = future.result(timeout=30.0)
                all_chunks.extend(segment_chunks)
            except Exception as e:
                print(f"Warning: Segment chunking failed: {e}")
                continue
        
        return all_chunks
    
    def _chunk_text_segment(
        self, 
        text: str, 
        start_offset: int, 
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Chunk a text segment"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk_data = {
                "text": chunk_text,
                "start_word": start_offset + i,
                "end_word": start_offset + min(i + self.chunk_size, len(words)),
                "metadata": metadata or {},
            }
            chunks.append(chunk_data)
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    @monitored_operation("scaling_index_builder.build_index")
    @resilient_operation(max_retries=2)
    def build_index(
        self,
        documents: List[Any],
        output_path: Optional[str] = None,
        enable_scaling_features: bool = True
    ) -> ScalingMockRetriever:
        """High-performance index building"""
        if not isinstance(documents, list):
            raise RetrievalError("Documents must be a list")
        
        if len(documents) == 0:
            raise RetrievalError("Cannot build index from empty document list")
        
        if output_path:
            output_path = RobustValidator.validate_file_path(output_path)
        
        try:
            print(f"Building high-performance index from {len(documents)} documents...")
            start_time = time.time()
            
            # Process documents with parallel chunking if enabled
            if self.enable_parallel_processing and len(documents) > 10:
                all_chunks = self._parallel_process_documents(documents)
            else:
                all_chunks = self._sequential_process_documents(documents)
            
            if not all_chunks:
                raise RetrievalError("No valid chunks created from documents")
            
            processing_time = time.time() - start_time
            print(f"Created {len(all_chunks)} chunks in {processing_time:.2f}s")
            
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
            
            # Create scaling retriever with optimization features
            retriever_config = {
                'mock_documents': retriever_docs,
                'embedding_dim': self.embedding_dim,
                'enable_caching': True,
                'enable_batch_processing': enable_scaling_features,
                'batch_size': 32,
                'max_concurrent_searches': 8,
                'enable_result_caching': enable_scaling_features
            }
            
            retriever = ScalingMockRetriever(**retriever_config)
            
            # Save index if path provided
            if output_path:
                retriever.save_index(output_path)
                print(f"Index saved to: {output_path}")
            
            total_time = time.time() - start_time
            print(f"Index building completed in {total_time:.2f}s")
            
            return retriever
            
        except Exception as e:
            raise RetrievalError(f"Scaling index building failed: {e}")
    
    def _sequential_process_documents(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """Sequential document processing"""
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            try:
                if isinstance(doc, str):
                    text = doc
                    doc_metadata = {"doc_id": doc_idx, "source": f"doc_{doc_idx}"}
                else:
                    text = doc.get("text", str(doc))
                    doc_metadata = doc.get("metadata", {})
                    doc_metadata["doc_id"] = doc_idx
                
                chunks = self.chunk_text(text, doc_metadata)
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Warning: Failed to process document {doc_idx}: {e}")
                continue
        
        return all_chunks
    
    def _parallel_process_documents(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """Parallel document processing"""
        # Submit all documents for parallel processing
        futures = []
        for doc_idx, doc in enumerate(documents):
            future = self.thread_pool.submit(self._process_single_document, doc, doc_idx)
            futures.append(future)
        
        # Collect results
        all_chunks = []
        for future in futures:
            try:
                doc_chunks = future.result(timeout=60.0)
                all_chunks.extend(doc_chunks)
            except Exception as e:
                print(f"Warning: Document processing failed: {e}")
                continue
        
        return all_chunks
    
    def _process_single_document(self, doc: Any, doc_idx: int) -> List[Dict[str, Any]]:
        """Process a single document"""
        try:
            if isinstance(doc, str):
                text = doc
                doc_metadata = {"doc_id": doc_idx, "source": f"doc_{doc_idx}"}
            else:
                text = doc.get("text", str(doc))
                doc_metadata = doc.get("metadata", {})
                doc_metadata["doc_id"] = doc_idx
            
            return self.chunk_text(text, doc_metadata)
            
        except Exception as e:
            print(f"Warning: Failed to process document {doc_idx}: {e}")
            return []
    
    def create_sample_documents(self) -> List[Dict[str, Any]]:
        """Create optimized sample documents for testing"""
        documents = [
            {
                "text": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Modern machine learning algorithms can process vast amounts of data and discover complex patterns that would be impossible for humans to identify manually.",
                "metadata": {"topic": "machine_learning", "difficulty": "beginner", "optimized": True, "length": "extended"}
            },
            {
                "text": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep neural networks have revolutionized computer vision, natural language processing, and many other domains by achieving human-level performance on complex tasks.",
                "metadata": {"topic": "deep_learning", "difficulty": "intermediate", "optimized": True, "length": "extended"}
            },
            {
                "text": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. Modern NLP systems use transformer architectures and attention mechanisms to achieve state-of-the-art performance.",
                "metadata": {"topic": "nlp", "difficulty": "intermediate", "optimized": True, "length": "extended"}
            },
            {
                "text": "Parameter-efficient fine-tuning (PEFT) is a set of techniques that allows for the adaptation of pre-trained models to specific tasks or domains while minimizing the number of parameters that need to be updated. This approach is particularly useful for large language models where full fine-tuning would be computationally expensive. Methods like LoRA, AdaLoRA, and IA3 have shown remarkable success in achieving competitive performance with minimal parameter updates.",
                "metadata": {"topic": "fine_tuning", "difficulty": "advanced", "optimized": True, "length": "extended"}
            },
            {
                "text": "Retrieval-augmented generation (RAG) is an AI framework for improving the quality of LLM-generated responses by grounding the model on external sources of knowledge to supplement the LLM's internal representation of information. RAG combines the power of retrieval systems with generative models, enabling more accurate and factually grounded responses. This approach has become essential for building reliable AI systems in production environments.",
                "metadata": {"topic": "rag", "difficulty": "advanced", "optimized": True, "length": "extended"}
            }
        ]
        
        # Validate all documents
        for doc in documents:
            RobustValidator.validate_text_input(doc["text"])
        
        return documents


# Export scaling retrieval components
__all__ = [
    'ScalingMockRetriever',
    'ScalingVectorIndexBuilder'
]