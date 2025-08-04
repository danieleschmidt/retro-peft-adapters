"""
Retrieval backends for vector-based document retrieval.

Supports multiple vector database backends with unified interface.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class BaseRetriever(ABC):
    """
    Base class for all retrieval backends.
    
    Provides unified interface for vector-based document retrieval
    across different backend implementations.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        encoder: Optional[Union[str, SentenceTransformer]] = None,
        device: str = "auto"
    ):
        self.embedding_dim = embedding_dim
        
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize encoder if provided
        if encoder is not None:
            if isinstance(encoder, str):
                self.encoder = SentenceTransformer(encoder, device=self.device)
            else:
                self.encoder = encoder
        else:
            self.encoder = None
    
    @abstractmethod
    def add_documents(
        self, 
        embeddings: np.ndarray, 
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Add documents to the retrieval index"""
        pass
    
    @abstractmethod
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Search for similar documents"""
        pass
    
    def retrieve(
        self,
        query_embeddings: Optional[torch.Tensor] = None,
        query_text: Optional[List[str]] = None,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        High-level retrieval interface.
        
        Args:
            query_embeddings: Pre-computed query embeddings [batch_size, dim]
            query_text: Query texts to encode (if embeddings not provided)
            k: Number of documents to retrieve
            filter_dict: Optional filters for retrieval
            
        Returns:
            Tuple of (context_embeddings, metadata_list)
        """
        # Encode query text if embeddings not provided
        if query_embeddings is None:
            if query_text is None:
                raise ValueError("Either query_embeddings or query_text must be provided")
            
            if self.encoder is None:
                raise ValueError("Encoder required for text queries")
            
            # Encode queries
            encoded_queries = self.encoder.encode(
                query_text, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            query_embeddings = torch.from_numpy(encoded_queries)
        
        # Convert to numpy for backend search
        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings_np = query_embeddings.cpu().numpy()
        else:
            query_embeddings_np = query_embeddings
        
        batch_size = query_embeddings_np.shape[0]
        
        # Retrieve for each query in batch
        all_contexts = []
        all_metadata = []
        
        for i in range(batch_size):
            query_emb = query_embeddings_np[i]
            
            # Search backend
            results, scores = self.search(query_emb, k=k, filter_dict=filter_dict)
            
            # Extract embeddings and metadata
            context_embeddings = []
            metadata_list = []
            
            for result in results:
                if 'embedding' in result:
                    context_embeddings.append(result['embedding'])
                else:
                    # Fallback: encode text if embedding not stored
                    if 'text' in result and self.encoder is not None:
                        emb = self.encoder.encode([result['text']], convert_to_numpy=True)[0]
                        context_embeddings.append(emb)
                    else:
                        # Zero embedding as fallback
                        context_embeddings.append(np.zeros(self.embedding_dim))
                
                metadata_list.append(result)
            
            # Pad to k documents if needed
            while len(context_embeddings) < k:
                context_embeddings.append(np.zeros(self.embedding_dim))
                metadata_list.append({"text": "", "score": 0.0})
            
            all_contexts.append(np.stack(context_embeddings))
            all_metadata.extend(metadata_list)
        
        # Convert to torch tensor
        context_tensor = torch.from_numpy(np.stack(all_contexts)).float()
        
        return context_tensor, all_metadata


class FAISSRetriever(BaseRetriever):
    """
    FAISS-based retrieval backend.
    
    Supports both flat and IVF indices for efficient similarity search.
    """
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        encoder: Optional[Union[str, SentenceTransformer]] = None,
        **kwargs
    ):
        # Import FAISS
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")
        
        self.index_path = index_path
        self.index = None
        self.metadata = []
        
        # Load existing index if provided
        if index_path and os.path.exists(index_path):
            self._load_index(index_path)
            embedding_dim = self.index.d
        
        if embedding_dim is None:
            raise ValueError("embedding_dim must be provided if not loading existing index")
        
        super().__init__(embedding_dim=embedding_dim, encoder=encoder, **kwargs)
        
        # Create new index if not loaded
        if self.index is None:
            self.index = self.faiss.IndexFlatIP(embedding_dim)
    
    def _load_index(self, index_path: str):
        """Load FAISS index and metadata"""
        self.index = self.faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = index_path.replace('.faiss', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                self.metadata = data.get('chunks', [])
        
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
    
    def add_documents(
        self, 
        embeddings: np.ndarray, 
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Add documents to FAISS index"""
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Normalize for cosine similarity (since we use IndexFlatIP)
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata.extend(metadata)
        
        print(f"Added {len(embeddings)} documents to FAISS index")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Search FAISS index"""
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1)
        self.faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        scores = scores[0].tolist()
        indices = indices[0].tolist()
        
        # Retrieve metadata
        results = []
        result_scores = []
        
        for idx, score in zip(indices, scores):
            if idx >= 0 and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(score)
                
                # Apply filters if specified
                if filter_dict is None or self._matches_filter(result, filter_dict):
                    results.append(result)
                    result_scores.append(float(score))
        
        return results, result_scores
    
    def _matches_filter(self, result: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if result matches filter criteria"""
        for key, value in filter_dict.items():
            if key in result.get('metadata', {}):
                if result['metadata'][key] != value:
                    return False
            elif key in result:
                if result[key] != value:
                    return False
        return True
    
    def save_index(self, save_path: str):
        """Save FAISS index to disk"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save index
        self.faiss.write_index(self.index, save_path)
        
        # Save metadata
        metadata_path = save_path.replace('.faiss', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'chunks': self.metadata,
                'embedding_dim': self.embedding_dim,
                'num_chunks': len(self.metadata)
            }, f, indent=2)
        
        print(f"FAISS index saved to: {save_path}")


class QdrantRetriever(BaseRetriever):
    """
    Qdrant-based retrieval backend.
    
    Supports filtering and payload-based retrieval.
    """
    
    def __init__(
        self,
        collection_name: str,
        host: str = "localhost:6333",
        embedding_dim: Optional[int] = None,
        encoder: Optional[Union[str, SentenceTransformer]] = None,
        **kwargs
    ):
        # Import Qdrant
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, Range
            self.qdrant_client = QdrantClient
            self.Distance = Distance
            self.VectorParams = VectorParams
            self.Filter = Filter
            self.FieldCondition = FieldCondition
            self.Range = Range
        except ImportError:
            raise ImportError("Qdrant client not installed. Run: pip install qdrant-client")
        
        self.collection_name = collection_name
        self.host = host
        
        # Initialize client
        self.client = self.qdrant_client(host=host)
        
        # Get collection info
        try:
            collection_info = self.client.get_collection(collection_name)
            if embedding_dim is None:
                embedding_dim = collection_info.config.params.vectors.size
        except:
            if embedding_dim is None:
                raise ValueError("embedding_dim must be provided for new collections")
        
        super().__init__(embedding_dim=embedding_dim, encoder=encoder, **kwargs)
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure collection exists"""
        try:
            self.client.get_collection(self.collection_name)
        except:
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.VectorParams(
                    size=self.embedding_dim,
                    distance=self.Distance.COSINE
                )
            )
            print(f"Created Qdrant collection: {self.collection_name}")
    
    def add_documents(
        self, 
        embeddings: np.ndarray, 
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Add documents to Qdrant collection"""
        from qdrant_client.models import PointStruct
        
        points = []
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            point = PointStruct(
                id=len(points),  # Simple incremental ID
                vector=embedding.tolist(),
                payload={
                    "text": meta.get("text", ""),
                    "metadata": meta.get("metadata", {})
                }
            )
            points.append(point)
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name, 
                points=batch
            )
        
        print(f"Added {len(embeddings)} documents to Qdrant")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Search Qdrant collection"""
        # Prepare filter
        search_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                condition = self.FieldCondition(
                    key=f"metadata.{key}",
                    match={"value": value}
                )
                conditions.append(condition)
            
            if conditions:
                search_filter = self.Filter(must=conditions)
        
        # Search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=search_filter,
            limit=k,
            with_payload=True
        )
        
        # Format results
        results = []
        scores = []
        
        for result in search_results:
            formatted_result = {
                "text": result.payload.get("text", ""),
                "metadata": result.payload.get("metadata", {}),
                "score": float(result.score)
            }
            results.append(formatted_result)
            scores.append(float(result.score))
        
        return results, scores


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval combining dense and sparse retrieval methods.
    
    Fuses results from multiple retrievers using reciprocal rank fusion
    or other combination strategies.
    """
    
    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: Optional[BaseRetriever] = None,
        fusion_method: str = "reciprocal_rank",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        **kwargs
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_method = fusion_method
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        super().__init__(
            embedding_dim=dense_retriever.embedding_dim,
            encoder=dense_retriever.encoder,
            **kwargs
        )
    
    def add_documents(
        self, 
        embeddings: np.ndarray, 
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Add documents to both retrievers"""
        self.dense_retriever.add_documents(embeddings, metadata)
        
        if self.sparse_retriever is not None:
            self.sparse_retriever.add_documents(embeddings, metadata)
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        dense_k: Optional[int] = None,
        sparse_k: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Hybrid search with result fusion"""
        dense_k = dense_k or min(k * 2, 20)  # Retrieve more for fusion
        sparse_k = sparse_k or min(k * 2, 20)
        
        # Dense retrieval
        dense_results, dense_scores = self.dense_retriever.search(
            query_embedding, k=dense_k, filter_dict=filter_dict
        )
        
        results = dense_results
        scores = dense_scores
        
        # Sparse retrieval (if available)
        if self.sparse_retriever is not None:
            sparse_results, sparse_scores = self.sparse_retriever.search(
                query_embedding, k=sparse_k, filter_dict=filter_dict
            )
            
            # Fuse results
            results, scores = self._fuse_results(
                dense_results, dense_scores,
                sparse_results, sparse_scores
            )
        
        # Return top-k results
        if len(results) > k:
            # Sort by score and take top-k
            sorted_pairs = sorted(
                zip(results, scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            results = [pair[0] for pair in sorted_pairs[:k]]
            scores = [pair[1] for pair in sorted_pairs[:k]]
        
        return results, scores
    
    def _fuse_results(
        self,
        dense_results: List[Dict[str, Any]],
        dense_scores: List[float],
        sparse_results: List[Dict[str, Any]],
        sparse_scores: List[float]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Fuse dense and sparse retrieval results"""
        if self.fusion_method == "reciprocal_rank":
            return self._reciprocal_rank_fusion(
                dense_results, dense_scores,
                sparse_results, sparse_scores
            )
        elif self.fusion_method == "weighted_score":
            return self._weighted_score_fusion(
                dense_results, dense_scores,
                sparse_results, sparse_scores
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        dense_scores: List[float],
        sparse_results: List[Dict[str, Any]],
        sparse_scores: List[float]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Reciprocal rank fusion"""
        k = 60  # RRF parameter
        
        # Create result map by text (simple deduplication)
        result_map = {}
        
        # Add dense results
        for rank, (result, score) in enumerate(zip(dense_results, dense_scores)):
            text_key = result.get("text", "")
            if text_key not in result_map:
                result_map[text_key] = {
                    "result": result,
                    "dense_rank": rank + 1,
                    "sparse_rank": None,
                    "dense_score": score,
                    "sparse_score": 0.0
                }
        
        # Add sparse results
        for rank, (result, score) in enumerate(zip(sparse_results, sparse_scores)):
            text_key = result.get("text", "")
            if text_key in result_map:
                result_map[text_key]["sparse_rank"] = rank + 1
                result_map[text_key]["sparse_score"] = score
            else:
                result_map[text_key] = {
                    "result": result,
                    "dense_rank": None,
                    "sparse_rank": rank + 1,
                    "dense_score": 0.0,
                    "sparse_score": score
                }
        
        # Compute RRF scores
        fused_results = []
        for text_key, data in result_map.items():
            rrf_score = 0.0
            
            if data["dense_rank"] is not None:
                rrf_score += self.dense_weight / (k + data["dense_rank"])
            
            if data["sparse_rank"] is not None:
                rrf_score += self.sparse_weight / (k + data["sparse_rank"])
            
            data["result"]["fused_score"] = rrf_score
            fused_results.append((data["result"], rrf_score))
        
        # Sort by fused score
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        results = [pair[0] for pair in fused_results]
        scores = [pair[1] for pair in fused_results]
        
        return results, scores
    
    def _weighted_score_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        dense_scores: List[float],
        sparse_results: List[Dict[str, Any]],
        sparse_scores: List[float]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Weighted score fusion"""
        # Normalize scores to [0, 1]
        def normalize_scores(scores):
            if not scores:
                return scores
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return [1.0] * len(scores)
            return [(s - min_score) / (max_score - min_score) for s in scores]
        
        norm_dense_scores = normalize_scores(dense_scores)
        norm_sparse_scores = normalize_scores(sparse_scores)
        
        # Create result map
        result_map = {}
        
        # Add dense results
        for result, score in zip(dense_results, norm_dense_scores):
            text_key = result.get("text", "")
            result_map[text_key] = {
                "result": result,
                "dense_score": score,
                "sparse_score": 0.0
            }
        
        # Add sparse results
        for result, score in zip(sparse_results, norm_sparse_scores):
            text_key = result.get("text", "")
            if text_key in result_map:
                result_map[text_key]["sparse_score"] = score
            else:
                result_map[text_key] = {
                    "result": result,
                    "dense_score": 0.0,
                    "sparse_score": score
                }
        
        # Compute weighted scores
        fused_results = []
        for data in result_map.values():
            weighted_score = (
                self.dense_weight * data["dense_score"] +
                self.sparse_weight * data["sparse_score"]
            )
            data["result"]["fused_score"] = weighted_score
            fused_results.append((data["result"], weighted_score))
        
        # Sort by weighted score
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        results = [pair[0] for pair in fused_results]
        scores = [pair[1] for pair in fused_results]
        
        return results, scores