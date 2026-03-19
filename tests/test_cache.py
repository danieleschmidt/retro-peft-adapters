"""Tests for KeyValueCache."""

import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retro_peft import KeyValueCache, CacheBuilder


class TestKeyValueCache:
    def _make_cache(self, n=16, d_key=32, d_val=64):
        keys = torch.randn(n, d_key)
        values = torch.randn(n, d_val)
        return KeyValueCache(keys, values)

    def test_basic_construction(self):
        cache = self._make_cache()
        assert cache.size == 16
        assert cache.key_dim == 32
        assert cache.value_dim == 64

    def test_mismatched_sizes_raises(self):
        with pytest.raises(ValueError):
            KeyValueCache(torch.randn(10, 8), torch.randn(12, 8))

    def test_retrieve_shape_1d_query(self):
        cache = self._make_cache(n=16, d_key=32, d_val=64)
        query = torch.randn(32)  # single query
        values, scores = cache.retrieve(query, k=4)
        assert values.shape == (4, 64)
        assert scores.shape == (4,)

    def test_retrieve_shape_batch(self):
        cache = self._make_cache(n=16, d_key=32, d_val=64)
        query = torch.randn(8, 32)  # batch of 8 queries
        values, scores = cache.retrieve(query, k=4)
        assert values.shape == (8, 4, 64)
        assert scores.shape == (8, 4)

    def test_retrieve_shape_3d(self):
        """Handles (batch, seq_len, d_key) queries."""
        cache = self._make_cache(n=16, d_key=32, d_val=64)
        query = torch.randn(4, 10, 32)
        values, scores = cache.retrieve(query, k=3)
        assert values.shape == (4, 10, 3, 64)
        assert scores.shape == (4, 10, 3)

    def test_retrieve_scores_are_cosine_similarities(self):
        """Scores should be in [-1, 1]."""
        cache = self._make_cache(n=32, d_key=16, d_val=8)
        query = torch.randn(5, 16)
        _, scores = cache.retrieve(query, k=4)
        assert scores.min() >= -1.01
        assert scores.max() <= 1.01

    def test_retrieve_k_clamp(self):
        """k larger than cache size returns cache.size neighbours."""
        cache = self._make_cache(n=5, d_key=8, d_val=4)
        query = torch.randn(8)
        values, scores = cache.retrieve(query, k=100)
        assert values.shape[0] == 5

    def test_exact_retrieval(self):
        """A query identical to a key should retrieve that key's value as top-1."""
        keys = torch.eye(8)       # 8 orthonormal keys
        values = torch.eye(8) * 10.0
        cache = KeyValueCache(keys, values)
        for i in range(8):
            q = keys[i]
            vals, scores = cache.retrieve(q, k=1)
            assert scores[0].item() > 0.99
            assert torch.allclose(vals[0], values[i], atol=1e-4)

    def test_to_device(self):
        cache = self._make_cache()
        cache.to(torch.device("cpu"))
        assert cache.device == torch.device("cpu")


class TestCacheBuilder:
    def test_from_tensors(self):
        keys = torch.randn(20, 16)
        values = torch.randn(20, 32)
        cache = CacheBuilder.from_tensors(keys, values)
        assert cache.size == 20

    def test_from_pairs(self):
        pairs = [(torch.randn(16), torch.randn(32)) for _ in range(10)]
        cache = CacheBuilder.from_pairs(pairs)
        assert cache.size == 10
        assert cache.key_dim == 16
        assert cache.value_dim == 32

    def test_add_and_build(self):
        builder = CacheBuilder()
        for _ in range(3):
            builder.add(torch.randn(5, 16), torch.randn(5, 32))
        cache = builder.build()
        assert cache.size == 15

    def test_build_empty_raises(self):
        builder = CacheBuilder()
        with pytest.raises(RuntimeError):
            builder.build()

    def test_3d_input_mean_pooling(self):
        """3D tensors (batch, seq, dim) are mean-pooled to (batch, dim)."""
        builder = CacheBuilder(pool="mean")
        builder.add(torch.randn(4, 10, 16), torch.randn(4, 10, 32))
        cache = builder.build()
        assert cache.size == 4
        assert cache.key_dim == 16

    def test_3d_input_cls_pooling(self):
        builder = CacheBuilder(pool="cls")
        builder.add(torch.randn(4, 10, 16), torch.randn(4, 10, 32))
        cache = builder.build()
        assert cache.size == 4

    def test_with_encoder(self):
        """Verify encoder callable is invoked."""
        calls = []
        def enc(x):
            calls.append(x)
            return torch.randn(len(x), 8)

        builder = CacheBuilder(key_encoder=enc, value_encoder=enc)
        builder.add(["a", "b", "c"], ["x", "y", "z"])
        assert len(calls) == 2  # key_encoder + value_encoder
