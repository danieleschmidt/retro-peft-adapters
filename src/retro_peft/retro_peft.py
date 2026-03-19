"""
RetroPEFT: full model wrapper that injects RetroAdapters at specified layers.

Wraps any nn.Module (a base model) and replaces target nn.Linear layers with
RetroAdapters. Target layers are specified by name substring matching — e.g.,
target_modules=["fc", "proj"] replaces any Linear whose name contains "fc" or "proj".

Caches can be attached globally or per-layer after injection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from .adapter import RetroAdapter
from .cache import KeyValueCache


class RetroPEFT(nn.Module):
    """
    Wrap a base model and inject RetroAdapters at target linear layers.

    After wrapping, only the RetroAdapter parameters (LoRA + retrieval gate)
    are trainable. The rest of the base model is frozen.

    Args:
        base_model: Any nn.Module to wrap.
        target_modules: List of name substrings. Any nn.Linear whose name
                        contains one of these strings will be replaced.
        rank: LoRA rank for all injected adapters.
        lora_alpha: LoRA scaling factor.
        k: Top-k retrieval neighbours.
        dropout: LoRA dropout.
        **adapter_kwargs: Extra kwargs passed to RetroAdapter.

    Example:
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
        peft = RetroPEFT(model, target_modules=["0", "2"], rank=4)
        peft.adapters  # dict of injected RetroAdapters
    """

    def __init__(
        self,
        base_model: nn.Module,
        target_modules: List[str],
        rank: int = 8,
        lora_alpha: float = 16.0,
        k: int = 4,
        dropout: float = 0.0,
        **adapter_kwargs,
    ) -> None:
        super().__init__()

        self.base_model = base_model
        self.target_modules = target_modules
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.k = k

        # Freeze all base model parameters first
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Inject adapters
        self._adapters: Dict[str, RetroAdapter] = {}
        self._inject(
            self.base_model, "",
            rank=rank, lora_alpha=lora_alpha, k=k, dropout=dropout,
            **adapter_kwargs,
        )

    def _inject(self, module: nn.Module, prefix: str, **adapter_kwargs) -> None:
        """Recursively replace matching Linear layers with RetroAdapters."""
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}".lstrip(".")
            if isinstance(child, nn.Linear) and self._is_target(full_name):
                adapter = RetroAdapter(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    freeze_base=True,
                    **adapter_kwargs,
                )
                # Copy base model weights into the adapter's frozen base layer
                adapter.base.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    adapter.base.bias.data.copy_(child.bias.data)
                else:
                    nn.init.zeros_(adapter.base.bias)

                setattr(module, name, adapter)
                self._adapters[full_name] = adapter
            else:
                self._inject(child, full_name, **adapter_kwargs)

    def _is_target(self, name: str) -> bool:
        return any(t in name for t in self.target_modules)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def set_cache(
        self,
        cache: Union[KeyValueCache, Dict[str, KeyValueCache]],
    ) -> None:
        """
        Attach retrieval caches to injected adapters.

        Args:
            cache: Either a single KeyValueCache (applied to all adapters)
                   or a dict mapping layer name → KeyValueCache.
        """
        if isinstance(cache, KeyValueCache):
            for adapter in self._adapters.values():
                adapter.set_cache(cache)
        else:
            for layer_name, layer_cache in cache.items():
                if layer_name not in self._adapters:
                    raise KeyError(
                        f"No adapter at layer '{layer_name}'. "
                        f"Available: {list(self._adapters)}"
                    )
                self._adapters[layer_name].set_cache(layer_cache)

    def clear_caches(self) -> None:
        """Remove all caches (revert to LoRA-only mode)."""
        for adapter in self._adapters.values():
            adapter.clear_cache()

    # ------------------------------------------------------------------
    # Adapter access
    # ------------------------------------------------------------------

    @property
    def adapters(self) -> Dict[str, RetroAdapter]:
        """Dict of injected RetroAdapters, keyed by layer name."""
        return self._adapters

    def adapter_names(self) -> List[str]:
        return list(self._adapters)

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> List[nn.Parameter]:
        """All parameters that require grad (LoRA + gates)."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def compression_ratio(self) -> float:
        total = self.total_param_count()
        trainable = self.trainable_param_count()
        return trainable / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"RetroPEFT(\n"
            f"  adapters={list(self._adapters)},\n"
            f"  trainable_params={self.trainable_param_count():,},\n"
            f"  total_params={self.total_param_count():,},\n"
            f"  compression={self.compression_ratio():.2%}\n"
            f")"
        )
