"""
High-performance caching system for retro-peft components.

Provides multi-level caching, memory management, and cache optimization.
"""

import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple, Union


class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set key-value pair with optional TTL"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get cache size"""
        pass
    
    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class MemoryCache(CacheBackend):
    """
    High-performance in-memory cache with LRU eviction.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: Optional[int] = 3600,
        cleanup_interval: int = 300
    ):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default TTL in seconds
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        # Thread-safe storage
        self._cache = OrderedDict()
        self._lock = RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._last_cleanup = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key with LRU update"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            # Check TTL
            value, expiry = self._cache[key]
            if expiry and time.time() > expiry:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            
            # Periodic cleanup
            self._maybe_cleanup()
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set key-value pair with TTL"""
        with self._lock:
            # Calculate expiry
            expiry = None
            if ttl is not None:
                expiry = time.time() + ttl
            elif self.default_ttl:
                expiry = time.time() + self.default_ttl
            
            # Remove old entry if exists
            if key in self._cache:
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = (value, expiry)
            
            # Evict if over capacity
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            return True
    
    def size(self) -> int:
        """Get cache size"""
        with self._lock:
            return len(self._cache)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / max(total_requests, 1)
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "utilization": len(self._cache) / self.max_size
            }
    
    def _maybe_cleanup(self):
        """Periodic cleanup of expired entries"""
        current_time = time.time()
        if current_time - self._last_cleanup > self.cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = current_time
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, (value, expiry) in self._cache.items():
            if expiry and current_time > expiry:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]


class MultiLevelCache:
    """
    Multi-level cache system with automatic promotion/demotion.
    
    Implements a hierarchy: L1 (fast, small) -> L2 (medium) -> L3 (large, slower)
    """
    
    def __init__(
        self,
        l1_size: int = 1000,
        l2_size: int = 10000,
        l3_size: int = 100000,
        promotion_threshold: int = 2
    ):
        """
        Initialize multi-level cache.
        
        Args:
            l1_size: L1 cache size (fastest)
            l2_size: L2 cache size
            l3_size: L3 cache size (largest)
            promotion_threshold: Access count for promotion
        """
        self.l1 = MemoryCache(max_size=l1_size, default_ttl=1800)  # 30 min
        self.l2 = MemoryCache(max_size=l2_size, default_ttl=3600)  # 1 hour
        self.l3 = MemoryCache(max_size=l3_size, default_ttl=7200)  # 2 hours
        
        self.promotion_threshold = promotion_threshold
        
        # Access tracking for promotion decisions
        self._access_counts = {}
        self._lock = RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value with automatic promotion"""
        # Try L1 first
        value = self.l1.get(key)
        if value is not None:
            return value
        
        # Try L2
        value = self.l2.get(key)
        if value is not None:
            self._track_access(key)
            # Maybe promote to L1
            if self._should_promote(key):
                self.l1.set(key, value)
            return value
        
        # Try L3
        value = self.l3.get(key)
        if value is not None:
            self._track_access(key)
            # Maybe promote to L2
            if self._should_promote(key):
                self.l2.set(key, value)
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in appropriate cache level"""
        # Always set in L3 first
        success = self.l3.set(key, value, ttl)
        
        # Reset access count
        with self._lock:
            self._access_counts[key] = 0
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete from all cache levels"""
        deleted = False
        deleted |= self.l1.delete(key)
        deleted |= self.l2.delete(key)
        deleted |= self.l3.delete(key)
        
        with self._lock:
            self._access_counts.pop(key, None)
        
        return deleted
    
    def clear(self) -> bool:
        """Clear all cache levels"""
        self.l1.clear()
        self.l2.clear()
        self.l3.clear()
        
        with self._lock:
            self._access_counts.clear()
        
        return True
    
    def _track_access(self, key: str):
        """Track access count for promotion decisions"""
        with self._lock:
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
    
    def _should_promote(self, key: str) -> bool:
        """Check if key should be promoted to higher cache level"""
        with self._lock:
            return self._access_counts.get(key, 0) >= self.promotion_threshold
    
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            "l1": self.l1.stats(),
            "l2": self.l2.stats(),
            "l3": self.l3.stats(),
            "tracked_keys": len(self._access_counts),
            "promotion_threshold": self.promotion_threshold
        }


class EmbeddingCache:
    """
    Specialized cache for embedding vectors and retrieval results.
    """
    
    def __init__(
        self,
        max_memory_mb: int = 1000,
        compression: bool = True,
        similarity_threshold: float = 0.95
    ):
        """
        Initialize embedding cache.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            compression: Whether to compress embeddings
            similarity_threshold: Threshold for similar query caching
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.compression = compression
        self.similarity_threshold = similarity_threshold
        
        # Storage
        self._embeddings = {}
        self._retrieval_results = {}
        self._query_embeddings = {}
        self._memory_usage = 0
        self._lock = RLock()
        
        # Statistics
        self._embedding_hits = 0
        self._embedding_misses = 0
        self._retrieval_hits = 0
        self._retrieval_misses = 0
    
    def cache_embedding(
        self, 
        text: str, 
        embedding: List[float]
    ) -> bool:
        """Cache text embedding"""
        with self._lock:
            if self._check_memory_limit():
                key = self._hash_text(text)
                data = self._compress_data(embedding) if self.compression else embedding
                
                self._embeddings[key] = {
                    'data': data,
                    'text': text,
                    'timestamp': time.time(),
                    'access_count': 0
                }
                
                self._memory_usage += len(pickle.dumps(data))
                return True
            return False
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text"""
        with self._lock:
            key = self._hash_text(text)
            
            if key in self._embeddings:
                entry = self._embeddings[key]
                entry['access_count'] += 1
                
                data = entry['data']
                if self.compression:
                    data = self._decompress_data(data)
                
                self._embedding_hits += 1
                return data
            
            # Check for similar queries
            similar_key = self._find_similar_query(text)
            if similar_key:
                entry = self._embeddings[similar_key]
                entry['access_count'] += 1
                
                data = entry['data']
                if self.compression:
                    data = self._decompress_data(data)
                
                self._embedding_hits += 1
                return data
            
            self._embedding_misses += 1
            return None
    
    def cache_retrieval_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]],
        k: int
    ) -> bool:
        """Cache retrieval results"""
        with self._lock:
            if self._check_memory_limit():
                key = f"{self._hash_text(query)}:{k}"
                
                self._retrieval_results[key] = {
                    'results': results,
                    'query': query,
                    'k': k,
                    'timestamp': time.time(),
                    'access_count': 0
                }
                
                self._memory_usage += len(pickle.dumps(results))
                return True
            return False
    
    def get_retrieval_results(
        self, 
        query: str, 
        k: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached retrieval results"""
        with self._lock:
            key = f"{self._hash_text(query)}:{k}"
            
            if key in self._retrieval_results:
                entry = self._retrieval_results[key]
                entry['access_count'] += 1
                self._retrieval_hits += 1
                return entry['results']
            
            self._retrieval_misses += 1
            return None
    
    def _hash_text(self, text: str) -> str:
        """Create hash key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data using pickle"""
        try:
            import gzip
            return gzip.compress(pickle.dumps(data))
        except ImportError:
            return pickle.dumps(data)
    
    def _decompress_data(self, data: bytes) -> Any:
        """Decompress data"""
        try:
            import gzip
            return pickle.loads(gzip.decompress(data))
        except (ImportError, OSError):
            return pickle.loads(data)
    
    def _check_memory_limit(self) -> bool:
        """Check if we're within memory limits"""
        if self._memory_usage > self.max_memory_bytes:
            self._evict_oldest()
        return self._memory_usage <= self.max_memory_bytes
    
    def _evict_oldest(self):
        """Evict oldest/least accessed entries"""
        # Combine all entries for eviction decision
        all_entries = []
        
        for key, entry in self._embeddings.items():
            all_entries.append((key, entry, 'embedding'))
        
        for key, entry in self._retrieval_results.items():
            all_entries.append((key, entry, 'retrieval'))
        
        # Sort by access count (ascending) and timestamp (ascending)
        all_entries.sort(
            key=lambda x: (x[1]['access_count'], x[1]['timestamp'])
        )
        
        # Evict oldest 10% or until under memory limit
        evict_count = max(1, len(all_entries) // 10)
        
        for i in range(min(evict_count, len(all_entries))):
            key, entry, entry_type = all_entries[i]
            
            if entry_type == 'embedding':
                data_size = len(pickle.dumps(self._embeddings[key]['data']))
                del self._embeddings[key]
            else:
                data_size = len(pickle.dumps(self._retrieval_results[key]['results']))
                del self._retrieval_results[key]
            
            self._memory_usage -= data_size
            
            if self._memory_usage <= self.max_memory_bytes:
                break
    
    def _find_similar_query(self, query: str) -> Optional[str]:
        """Find similar cached query using simple text similarity"""
        query_words = set(query.lower().split())
        
        best_key = None
        best_similarity = 0.0
        
        for key, entry in self._embeddings.items():
            cached_words = set(entry['text'].lower().split())
            
            if not cached_words or not query_words:
                continue
            
            # Jaccard similarity
            intersection = len(query_words & cached_words)
            union = len(query_words | cached_words)
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_key = key
        
        return best_key
    
    def clear(self):
        """Clear all cached data"""
        with self._lock:
            self._embeddings.clear()
            self._retrieval_results.clear()
            self._query_embeddings.clear()
            self._memory_usage = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_embedding_requests = self._embedding_hits + self._embedding_misses
            total_retrieval_requests = self._retrieval_hits + self._retrieval_misses
            
            return {
                "memory_usage_mb": self._memory_usage / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "memory_utilization": self._memory_usage / self.max_memory_bytes,
                "embeddings_cached": len(self._embeddings),
                "retrieval_results_cached": len(self._retrieval_results),
                "embedding_hit_rate": (
                    self._embedding_hits / max(total_embedding_requests, 1)
                ),
                "retrieval_hit_rate": (
                    self._retrieval_hits / max(total_retrieval_requests, 1)
                ),
                "compression_enabled": self.compression,
                "similarity_threshold": self.similarity_threshold
            }


class CacheManager:
    """
    Unified cache manager that coordinates different cache types.
    """
    
    def __init__(self):
        self.caches = {}
        self._lock = RLock()
    
    def register_cache(self, name: str, cache: CacheBackend):
        """Register a cache backend"""
        with self._lock:
            self.caches[name] = cache
    
    def get_cache(self, name: str) -> Optional[CacheBackend]:
        """Get cache by name"""
        with self._lock:
            return self.caches.get(name)
    
    def clear_all(self):
        """Clear all registered caches"""
        with self._lock:
            for cache in self.caches.values():
                cache.clear()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches"""
        with self._lock:
            stats = {}
            for name, cache in self.caches.items():
                stats[name] = cache.stats()
            return stats