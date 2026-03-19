"""Tests for RetroPEFT model wrapper."""

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retro_peft import KeyValueCache, CacheBuilder, RetroPEFT


def make_model():
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    )


def make_cache(n=20, d_key=64, d_val=64):
    return KeyValueCache(torch.randn(n, d_key), torch.randn(n, d_val))


class TestRetroPEFT:
    def test_wraps_target_layers(self):
        model = make_model()
        peft = RetroPEFT(model, target_modules=["0", "2"], rank=4)
        assert "0" in peft.adapters
        assert "2" in peft.adapters

    def test_forward_shape(self):
        model = make_model()
        peft = RetroPEFT(model, target_modules=["0", "2"], rank=4)
        x = torch.randn(8, 32)
        out = peft(x)
        assert out.shape == (8, 16)

    def test_base_params_frozen(self):
        model = make_model()
        peft = RetroPEFT(model, target_modules=["0"], rank=4)
        for name, p in peft.named_parameters():
            # The frozen base linear layer is at ".base.weight" / ".base.bias"
            if name.endswith(".base.weight") or name.endswith(".base.bias"):
                assert not p.requires_grad, f"{name} should be frozen"

    def test_lora_params_trainable(self):
        model = make_model()
        peft = RetroPEFT(model, target_modules=["0"], rank=4)
        lora_params = [n for n, p in peft.named_parameters()
                       if p.requires_grad and "lora" in n]
        assert len(lora_params) > 0

    def test_trainable_param_count_less_than_total(self):
        model = make_model()
        peft = RetroPEFT(model, target_modules=["0", "2"], rank=4)
        assert peft.trainable_param_count() < peft.total_param_count()

    def test_compression_ratio(self):
        model = make_model()
        peft = RetroPEFT(model, target_modules=["0", "2"], rank=4)
        ratio = peft.compression_ratio()
        assert 0.0 < ratio < 1.0

    def test_set_cache_global(self):
        model = make_model()
        peft = RetroPEFT(model, target_modules=["0"], rank=4)
        cache = make_cache(d_key=32, d_val=64)
        peft.set_cache(cache)
        for adapter in peft.adapters.values():
            assert adapter.has_cache

    def test_set_cache_per_layer(self):
        model = make_model()
        peft = RetroPEFT(model, target_modules=["0", "2"], rank=4)
        peft.set_cache({
            "0": make_cache(d_key=32, d_val=64),
            "2": make_cache(d_key=64, d_val=16),
        })
        assert peft.adapters["0"].has_cache
        assert peft.adapters["2"].has_cache

    def test_set_cache_bad_key_raises(self):
        model = make_model()
        peft = RetroPEFT(model, target_modules=["0"], rank=4)
        with pytest.raises(KeyError):
            peft.set_cache({"nonexistent": make_cache()})

    def test_clear_caches(self):
        model = make_model()
        peft = RetroPEFT(model, target_modules=["0"], rank=4)
        peft.set_cache(make_cache(d_key=32, d_val=64))
        peft.clear_caches()
        for adapter in peft.adapters.values():
            assert not adapter.has_cache

    def test_gradients_flow_end_to_end(self):
        model = make_model()
        peft = RetroPEFT(model, target_modules=["0", "2"], rank=4)
        x = torch.randn(4, 32)
        loss = peft(x).sum()
        loss.backward()
        # LoRA weights should have gradients
        for adapter in peft.adapters.values():
            assert adapter.lora_A.weight.grad is not None

    def test_partial_target_modules(self):
        """Targeting only one layer leaves the other as-is."""
        model = make_model()
        peft = RetroPEFT(model, target_modules=["0"], rank=4)
        assert "0" in peft.adapters
        assert "2" not in peft.adapters

    def test_weights_copied_from_base(self):
        """Adapter base layer should have the same weights as the original Linear."""
        model = make_model()
        orig_weight = model[0].weight.data.clone()
        peft = RetroPEFT(model, target_modules=["0"], rank=4)
        adapter_weight = peft.adapters["0"].base.weight.data
        assert torch.allclose(orig_weight, adapter_weight)
