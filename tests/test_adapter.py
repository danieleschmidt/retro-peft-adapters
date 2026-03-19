"""Tests for RetroAdapter."""

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retro_peft import KeyValueCache, CacheBuilder, RetroAdapter


def make_cache(n=32, d_key=16, d_val=32):
    keys = torch.randn(n, d_key)
    values = torch.randn(n, d_val)
    return KeyValueCache(keys, values)


class TestRetroAdapter:
    def test_output_shape_no_cache(self):
        adapter = RetroAdapter(in_features=16, out_features=32, rank=4)
        x = torch.randn(8, 16)
        out = adapter(x)
        assert out.shape == (8, 32)

    def test_output_shape_with_cache(self):
        cache = make_cache(n=32, d_key=16, d_val=32)
        adapter = RetroAdapter(in_features=16, out_features=32, rank=4, cache=cache)
        x = torch.randn(8, 16)
        out = adapter(x)
        assert out.shape == (8, 32)

    def test_3d_input(self):
        cache = make_cache(n=32, d_key=16, d_val=32)
        adapter = RetroAdapter(in_features=16, out_features=32, rank=4, cache=cache)
        x = torch.randn(4, 10, 16)
        out = adapter(x)
        assert out.shape == (4, 10, 32)

    def test_base_is_frozen(self):
        adapter = RetroAdapter(in_features=16, out_features=32, rank=4, freeze_base=True)
        for p in adapter.base.parameters():
            assert not p.requires_grad

    def test_lora_is_trainable(self):
        adapter = RetroAdapter(in_features=16, out_features=32, rank=4)
        for p in adapter.lora_A.parameters():
            assert p.requires_grad
        for p in adapter.lora_B.parameters():
            assert p.requires_grad

    def test_retrieval_gate_is_trainable(self):
        adapter = RetroAdapter(in_features=16, out_features=32, rank=4)
        assert adapter.retrieval_gate.requires_grad

    def test_set_and_clear_cache(self):
        adapter = RetroAdapter(in_features=16, out_features=32, rank=4)
        assert not adapter.has_cache

        cache = make_cache(d_key=16, d_val=32)
        adapter.set_cache(cache)
        assert adapter.has_cache

        adapter.clear_cache()
        assert not adapter.has_cache

    def test_retrieval_changes_output(self):
        """With cache active, output should differ from LoRA-only output."""
        torch.manual_seed(42)
        adapter = RetroAdapter(in_features=16, out_features=32, rank=4,
                               retrieval_gate_init=5.0)  # high gate → strong retrieval
        x = torch.randn(4, 16)
        out_no_cache = adapter(x).detach().clone()

        cache = make_cache(d_key=16, d_val=32)
        adapter.set_cache(cache)
        out_with_cache = adapter(x).detach().clone()

        assert not torch.allclose(out_no_cache, out_with_cache)

    def test_value_proj(self):
        """value_proj_dim projects d_val → out_features."""
        cache = make_cache(n=32, d_key=16, d_val=64)
        adapter = RetroAdapter(
            in_features=16, out_features=32, rank=4,
            cache=cache, value_proj_dim=64,
        )
        x = torch.randn(8, 16)
        out = adapter(x)
        assert out.shape == (8, 32)

    def test_lora_init_zero_B(self):
        """lora_B is initialized to zero (standard LoRA)."""
        adapter = RetroAdapter(in_features=16, out_features=32, rank=4)
        assert torch.all(adapter.lora_B.weight == 0)

    def test_gradients_flow_through_lora(self):
        """Gradients should flow through LoRA parameters."""
        adapter = RetroAdapter(in_features=16, out_features=32, rank=4)
        x = torch.randn(4, 16)
        loss = adapter(x).sum()
        loss.backward()
        assert adapter.lora_A.weight.grad is not None
        assert adapter.lora_B.weight.grad is not None

    def test_gradients_flow_through_gate(self):
        """Gradients flow through retrieval gate when cache is active."""
        cache = make_cache(d_key=16, d_val=32)
        adapter = RetroAdapter(in_features=16, out_features=32, rank=4, cache=cache)
        x = torch.randn(4, 16)
        loss = adapter(x).sum()
        loss.backward()
        assert adapter.retrieval_gate.grad is not None

    def test_base_has_no_grad(self):
        """Gradients do NOT flow to frozen base weights."""
        adapter = RetroAdapter(in_features=16, out_features=32, rank=4, freeze_base=True)
        x = torch.randn(4, 16)
        loss = adapter(x).sum()
        loss.backward()
        assert adapter.base.weight.grad is None
