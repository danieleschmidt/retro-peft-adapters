"""
Advanced multi-level caching system for retrieval-augmented inference.

Provides memory, disk, and distributed caching with intelligent
eviction policies and cache warming strategies.
"""

import time
import json
import hashlib
import pickle
import threading
import os
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path
import sqlite3
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from ..utils.logging import get_global_logger


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() > (self.timestamp + self.ttl)
    
    def age_seconds(self) -> float:
        """Get age in seconds"""
        return time.time() - self.timestamp


class LRUCache:
    """
    Thread-safe LRU cache with TTL support.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self.logger = get_global_logger()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return default
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self.misses += 1
                return default
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.access_count += 1
            self.hits += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache"""
        with self._lock:
            # Calculate size estimate
            size_bytes = self._estimate_size(value)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl
            )
            
            # Add/update entry
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                self._cache[key] = entry
                
                # Evict if necessary
                while len(self._cache) > self.max_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    self.evictions += 1
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries"""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get number of entries"""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "total_size_bytes": total_size,
                "avg_entry_size": total_size / len(self._cache) if self._cache else 0
            }
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            if isinstance(obj, (str, bytes)):
                return len(obj)
            elif isinstance(obj, (int, float, bool)):
                return 8  # Rough estimate
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v) 
                    for k, v in obj.items()
                )
            elif hasattr(obj, '__sizeof__'):
                return obj.__sizeof__()
            else:
                return len(pickle.dumps(obj))
        except Exception:
            return 1024  # Fallback estimate


class DiskCache:
    """
    Persistent disk-based cache with SQLite backend.
    """
    
    def __init__(
        self, 
        cache_dir: str,
        max_size_mb: int = 1000,
        cleanup_interval: int = 3600  # 1 hour
    ):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in MB
            cleanup_interval: Cleanup interval in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cleanup_interval = cleanup_interval
        
        # Database setup
        self.db_path = self.cache_dir / "cache.db"
        self._init_database()
        
        self.logger = get_global_logger()
        self._last_cleanup = time.time()
        self._lock = threading.RLock()
    
    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    ttl REAL,
                    size_bytes INTEGER NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_access_count ON cache_entries(access_count)
            """)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from disk cache"""
        with self._lock:
            self._maybe_cleanup()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT filename, timestamp, ttl, access_count FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return default
                
                filename, timestamp, ttl, access_count = row
                
                # Check expiration
                if ttl and time.time() > (timestamp + ttl):
                    self._delete_entry(key, filename)
                    return default
                
                # Load value from file
                file_path = self.cache_dir / filename
                if not file_path.exists():
                    self._delete_entry(key, filename)
                    return default
                
                try:
                    with open(file_path, 'rb') as f:
                        value = pickle.load(f)
                    
                    # Update access count
                    conn.execute(
                        "UPDATE cache_entries SET access_count = ? WHERE key = ?",
                        (access_count + 1, key)
                    )
                    
                    return value
                    
                except Exception as e:
                    self.logger.error(f"Error loading cache entry {key}: {e}")
                    self._delete_entry(key, filename)
                    return default
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in disk cache"""
        with self._lock:
            self._maybe_cleanup()
            
            # Generate filename
            key_hash = hashlib.md5(key.encode()).hexdigest()
            filename = f"cache_{key_hash}.pkl"
            file_path = self.cache_dir / filename
            
            try:
                # Save value to file
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                
                file_size = file_path.stat().st_size
                
                # Update database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (key, filename, timestamp, ttl, size_bytes, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        key, filename, time.time(), ttl, file_size, 
                        json.dumps({})
                    ))
                
                # Check size limit
                self._enforce_size_limit()
                
            except Exception as e:
                self.logger.error(f"Error saving cache entry {key}: {e}")
                if file_path.exists():
                    file_path.unlink()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT filename FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    filename = row[0]
                    self._delete_entry(key, filename)
                    return True
                return False
    
    def clear(self) -> None:
        """Clear all entries"""
        with self._lock:
            # Delete all files
            for file_path in self.cache_dir.glob("cache_*.pkl"):
                try:
                    file_path.unlink()
                except Exception:
                    pass
            
            # Clear database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as entry_count,
                    SUM(size_bytes) as total_size,
                    AVG(size_bytes) as avg_size,
                    SUM(access_count) as total_accesses
                FROM cache_entries
            """)
            row = cursor.fetchone()
            
            entry_count, total_size, avg_size, total_accesses = row
            total_size = total_size or 0
            avg_size = avg_size or 0
            total_accesses = total_accesses or 0
            
            return {
                "entry_count": entry_count,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "avg_entry_size": avg_size,
                "total_accesses": total_accesses,
                "max_size_mb": self.max_size_bytes / (1024 * 1024)
            }
    
    def _delete_entry(self, key: str, filename: str):
        """Delete cache entry and file"""
        try:
            file_path = self.cache_dir / filename
            if file_path.exists():
                file_path.unlink()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                
        except Exception as e:
            self.logger.error(f"Error deleting cache entry {key}: {e}")
    
    def _enforce_size_limit(self):
        """Enforce cache size limit by evicting old entries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
            total_size = cursor.fetchone()[0] or 0
            
            if total_size > self.max_size_bytes:
                # Evict least recently used entries
                cursor = conn.execute("""
                    SELECT key, filename FROM cache_entries 
                    ORDER BY access_count ASC, timestamp ASC
                """)
                
                for key, filename in cursor:
                    self._delete_entry(key, filename)
                    
                    # Check if we're under the limit
                    cursor2 = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
                    current_size = cursor2.fetchone()[0] or 0
                    
                    if current_size <= self.max_size_bytes * 0.8:  # Leave some headroom
                        break
    
    def _maybe_cleanup(self):
        """Run cleanup if needed"""
        now = time.time()
        if now - self._last_cleanup > self.cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = now
    
    def _cleanup_expired(self):
        """Clean up expired entries"""
        now = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT key, filename FROM cache_entries 
                WHERE ttl IS NOT NULL AND ? > (timestamp + ttl)
            """, (now,))
            
            expired_entries = cursor.fetchall()
            
            for key, filename in expired_entries:
                self._delete_entry(key, filename)
            
            if expired_entries:
                self.logger.info(f"Cleaned up {len(expired_entries)} expired cache entries")


class VectorCache:
    """
    Specialized cache for vector embeddings with similarity search.
    """
    
    def __init__(
        self, 
        max_vectors: int = 100000,
        vector_dim: int = 768,
        similarity_threshold: float = 0.9
    ):
        """
        Initialize vector cache.
        
        Args:
            max_vectors: Maximum number of vectors to store
            vector_dim: Vector dimensionality
            similarity_threshold: Minimum similarity for cache hits
        """
        self.max_vectors = max_vectors
        self.vector_dim = vector_dim
        self.similarity_threshold = similarity_threshold
        
        # Storage
        self.vectors = np.zeros((max_vectors, vector_dim), dtype=np.float32)
        self.metadata: List[Dict[str, Any]] = []
        self.keys: List[str] = []
        self.count = 0
        
        self._lock = threading.RLock()
        self.logger = get_global_logger()
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def get_similar(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Find similar vectors in cache.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            
        Returns:
            List of (key, metadata, similarity_score) tuples
        """
        with self._lock:
            if self.count == 0:
                self.misses += 1
                return []
            
            # Compute similarities
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0:
                return []
            
            query_normalized = query_vector / query_norm
            
            # Get stored vectors (only up to count)
            stored_vectors = self.vectors[:self.count]
            stored_norms = np.linalg.norm(stored_vectors, axis=1)
            
            # Avoid division by zero
            valid_indices = stored_norms > 0
            if not np.any(valid_indices):
                self.misses += 1
                return []
            
            similarities = np.zeros(self.count)
            similarities[valid_indices] = np.dot(
                stored_vectors[valid_indices] / stored_norms[valid_indices, np.newaxis],
                query_normalized
            )
            
            # Find top-k similar vectors above threshold
            indices = np.where(similarities >= self.similarity_threshold)[0]
            if len(indices) == 0:
                self.misses += 1
                return []
            
            # Sort by similarity
            sorted_indices = indices[np.argsort(similarities[indices])[::-1]]
            top_indices = sorted_indices[:top_k]
            
            results = []
            for idx in top_indices:
                results.append((
                    self.keys[idx],
                    self.metadata[idx].copy(),
                    float(similarities[idx])
                ))
            
            if results:
                self.hits += 1
            else:
                self.misses += 1
            
            return results
    
    def put(
        self, 
        key: str, 
        vector: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> None:
        """
        Store vector in cache.
        
        Args:
            key: Cache key
            vector: Vector to store
            metadata: Associated metadata
        """
        with self._lock:
            if vector.shape[0] != self.vector_dim:
                raise ValueError(f"Vector dimension {vector.shape[0]} != {self.vector_dim}")
            
            if self.count < self.max_vectors:
                # Add new vector
                idx = self.count
                self.count += 1
            else:
                # Replace oldest vector (FIFO)
                idx = 0
                self.vectors[:-1] = self.vectors[1:]
                self.metadata[:-1] = self.metadata[1:]
                self.keys[:-1] = self.keys[1:]
                idx = self.count - 1
            
            # Store vector and metadata
            self.vectors[idx] = vector.astype(np.float32)
            
            if idx < len(self.metadata):
                self.metadata[idx] = metadata.copy()
                self.keys[idx] = key
            else:
                self.metadata.append(metadata.copy())
                self.keys.append(key)
    
    def clear(self) -> None:
        """Clear all vectors"""
        with self._lock:
            self.count = 0
            self.metadata.clear()
            self.keys.clear()
            self.vectors.fill(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "count": self.count,
                "max_vectors": self.max_vectors,
                "vector_dim": self.vector_dim,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "similarity_threshold": self.similarity_threshold,
                "memory_usage_mb": (
                    self.vectors.nbytes + 
                    len(self.metadata) * 1024 +  # Rough estimate
                    len(self.keys) * 100
                ) / (1024 * 1024)
            }


class MultiLevelCache:
    """
    Multi-level cache system combining memory, disk, and vector caches.
    """
    
    def __init__(
        self,
        memory_cache_size: int = 1000,
        disk_cache_dir: str = "./cache",
        disk_cache_size_mb: int = 1000,
        vector_cache_size: int = 10000,
        enable_vector_cache: bool = True
    ):
        """
        Initialize multi-level cache.
        
        Args:
            memory_cache_size: Size of memory cache
            disk_cache_dir: Directory for disk cache
            disk_cache_size_mb: Size of disk cache in MB
            vector_cache_size: Size of vector cache
            enable_vector_cache: Whether to enable vector cache
        """
        self.memory_cache = LRUCache(max_size=memory_cache_size)
        self.disk_cache = DiskCache(disk_cache_dir, disk_cache_size_mb)
        
        if enable_vector_cache:
            self.vector_cache = VectorCache(max_vectors=vector_cache_size)
        else:
            self.vector_cache = None
        
        self.logger = get_global_logger()
        self._lock = threading.RLock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache (memory -> disk)"""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.put(key, value)
            return value
        
        return default
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache"""
        # Store in both memory and disk
        self.memory_cache.put(key, value, ttl)
        self.disk_cache.put(key, value, ttl)
    
    def get_similar_vectors(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """Get similar vectors from vector cache"""
        if self.vector_cache is None:
            return []
        
        return self.vector_cache.get_similar(query_vector, top_k)
    
    def put_vector(
        self, 
        key: str, 
        vector: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> None:
        """Store vector in vector cache"""
        if self.vector_cache is not None:
            self.vector_cache.put(key, vector, metadata)
    
    def delete(self, key: str) -> bool:
        """Delete key from all caches"""
        deleted_memory = self.memory_cache.delete(key)
        deleted_disk = self.disk_cache.delete(key)
        return deleted_memory or deleted_disk
    
    def clear(self) -> None:
        """Clear all caches"""
        self.memory_cache.clear()
        self.disk_cache.clear()
        if self.vector_cache:
            self.vector_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            "memory_cache": self.memory_cache.get_stats(),
            "disk_cache": self.disk_cache.get_stats(),
        }
        
        if self.vector_cache:
            stats["vector_cache"] = self.vector_cache.get_stats()
        
        return stats


# Global cache manager
_global_cache_manager = None


def get_cache_manager() -> MultiLevelCache:
    """Get global cache manager instance"""
    global _global_cache_manager
    
    if _global_cache_manager is None:
        _global_cache_manager = MultiLevelCache()
    
    return _global_cache_manager