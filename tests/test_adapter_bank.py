"""Tests for AdapterBank."""

import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retro_peft import KeyValueCache, AdapterBank


def make_cache(n=20, d_key=16, d_val=32):
    return KeyValueCache(torch.randn(n, d_key), torch.randn(n, d_val))


class TestAdapterBank:
    def test_register_and_activate(self):
        bank = AdapterBank(16, 32, rank=4)
        cache = make_cache()
        bank.register_domain("math", cache)
        bank.activate("math")
        assert bank.active_domain == "math"

    def test_forward_no_cache(self):
        bank = AdapterBank(16, 32, rank=4)
        bank.register_domain("code")
        bank.activate("code")
        x = torch.randn(4, 16)
        out = bank(x)
        assert out.shape == (4, 32)

    def test_forward_with_cache(self):
        cache = make_cache()
        bank = AdapterBank(16, 32, rank=4)
        bank.register_domain("science", cache)
        bank.activate("science")
        x = torch.randn(4, 16)
        out = bank(x)
        assert out.shape == (4, 32)

    def test_domain_swap_changes_output(self):
        """Hot-swapping cache changes output without touching LoRA weights."""
        cache_a = make_cache()
        cache_b = make_cache()
        bank = AdapterBank(16, 32, rank=4, retrieval_gate_init=5.0)
        bank.register_domain("a", cache_a)
        bank.register_domain("b", cache_b)

        x = torch.randn(4, 16)
        bank.activate("a")
        out_a = bank(x).detach().clone()

        bank.activate("b")
        out_b = bank(x).detach().clone()

        # Different caches → different retrieval → different outputs
        assert not torch.allclose(out_a, out_b)

    def test_duplicate_domain_raises(self):
        bank = AdapterBank(16, 32, rank=4)
        bank.register_domain("x")
        with pytest.raises(ValueError):
            bank.register_domain("x")

    def test_activate_unknown_raises(self):
        bank = AdapterBank(16, 32, rank=4)
        with pytest.raises(KeyError):
            bank.activate("unknown")

    def test_set_cache(self):
        bank = AdapterBank(16, 32, rank=4)
        bank.register_domain("d1")
        cache = make_cache()
        bank.set_cache("d1", cache)
        bank.activate("d1")
        x = torch.randn(4, 16)
        out = bank(x)
        assert out.shape == (4, 32)

    def test_deactivate_removes_cache(self):
        cache = make_cache()
        bank = AdapterBank(16, 32, rank=4)
        bank.register_domain("d", cache)
        bank.activate("d")
        assert bank.adapter.has_cache
        bank.deactivate()
        assert not bank.adapter.has_cache

    def test_domains_list(self):
        bank = AdapterBank(16, 32, rank=4, domains=["a", "b", "c"])
        assert set(bank.domains) == {"a", "b", "c"}

    def test_per_domain_adapters(self):
        """shared_lora=False gives each domain its own adapter."""
        bank = AdapterBank(16, 32, rank=4, shared_lora=False)
        bank.register_domain("d1", make_cache())
        bank.register_domain("d2", make_cache())
        bank.activate("d1")
        x = torch.randn(4, 16)
        out = bank(x)
        assert out.shape == (4, 32)

    def test_per_domain_no_active_raises(self):
        bank = AdapterBank(16, 32, rank=4, shared_lora=False)
        bank.register_domain("d1")
        x = torch.randn(4, 16)
        with pytest.raises(RuntimeError):
            bank(x)
