"""
RetroAdapter: a PEFT adapter that augments a linear layer with retrieval.

The output is a learned weighted combination of:
  1. The standard LoRA output (low-rank adaptation of the frozen base layer).
  2. Retrieved value vectors from a KeyValueCache (weighted by similarity scores).

This allows fast domain adaptation: swap the cache to adapt to a new domain
without re-training — the adapter learns to blend retrieval and parametric updates.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import KeyValueCache


class RetroAdapter(nn.Module):
    """
    Retrieval-augmented LoRA adapter for a linear layer.

    Replaces (or wraps) an nn.Linear with:
        output = base(x) + LoRA(x) + retrieval_blend(x, cache)

    where retrieval_blend computes a weighted sum of top-k retrieved values,
    gated by a learned scalar.

    Args:
        in_features: Input dimension of the wrapped linear layer.
        out_features: Output dimension of the wrapped linear layer.
        rank: LoRA rank (default 8).
        lora_alpha: LoRA scaling factor. Effective scale = lora_alpha / rank.
        dropout: Dropout on LoRA path.
        cache: Optional KeyValueCache. Can be set later via .set_cache().
        k: Number of neighbours to retrieve.
        retrieval_gate_init: Initial value for the retrieval gate (sigmoid'd).
                              0.0 → gate≈0.5, large negative → gate≈0 (disabled).
        value_proj_dim: If not None, project values from d_val → out_features via
                        a learned linear. If None, values must already be out_features.
        freeze_base: If True (default), the base weight is not trainable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.0,
        cache: Optional[KeyValueCache] = None,
        k: int = 4,
        retrieval_gate_init: float = -2.0,
        value_proj_dim: Optional[int] = None,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_alpha / rank
        self.k = k

        # ------------------------------------------------------------------
        # Base linear (frozen)
        # ------------------------------------------------------------------
        self.base = nn.Linear(in_features, out_features, bias=True)
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        # ------------------------------------------------------------------
        # LoRA matrices (trainable)
        # ------------------------------------------------------------------
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        # Initialize: A ~ N(0, 1/sqrt(in)), B = 0  (standard LoRA init)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # ------------------------------------------------------------------
        # Retrieval components
        # ------------------------------------------------------------------
        self._cache: Optional[KeyValueCache] = cache
        self._value_dim = value_proj_dim  # None means cache.value_dim == out_features

        # Project retrieved values to out_features if needed
        if value_proj_dim is not None:
            self.value_proj = nn.Linear(value_proj_dim, out_features, bias=False)
        else:
            self.value_proj = None

        # Learned gate: scalar weight on the retrieval term.
        # Initialize small (close to 0) so retrieval starts off gently.
        self.retrieval_gate = nn.Parameter(
            torch.full((1,), retrieval_gate_init)
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def set_cache(self, cache: KeyValueCache) -> None:
        """Attach or replace the retrieval cache."""
        self._cache = cache

    def clear_cache(self) -> None:
        """Detach the retrieval cache (disables retrieval path)."""
        self._cache = None

    @property
    def has_cache(self) -> bool:
        return self._cache is not None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Base (frozen) pass
        out = self.base(x)

        # LoRA pass
        lora_out = self.lora_B(self.lora_dropout(self.lora_A(x))) * self.lora_scale
        out = out + lora_out

        # Retrieval pass (if cache attached)
        if self._cache is not None:
            out = out + self._retrieval_term(x)

        return out

    def _retrieval_term(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the retrieval augmentation term."""
        # Use x as query into the cache
        values, scores = self._cache.retrieve(x, k=self.k)
        # values: (..., k, d_val), scores: (..., k)

        # Softmax-weight the retrieved values by similarity score
        weights = F.softmax(scores, dim=-1)  # (..., k)
        # Weighted sum: (..., d_val)
        retrieved = (weights.unsqueeze(-1) * values).sum(dim=-2)

        # Project to out_features if needed
        if self.value_proj is not None:
            retrieved = self.value_proj(retrieved)

        # Gate: learned scalar controlling retrieval influence
        gate = torch.sigmoid(self.retrieval_gate)
        return gate * retrieved

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def lora_parameters(self):
        """Return only the LoRA parameters (for targeted optimizers)."""
        return list(self.lora_A.parameters()) + list(self.lora_B.parameters())

    def retrieval_parameters(self):
        """Return retrieval-specific parameters."""
        params = [self.retrieval_gate]
        if self.value_proj is not None:
            params += list(self.value_proj.parameters())
        return params

    def trainable_parameters(self):
        return self.lora_parameters() + self.retrieval_parameters()

    def __repr__(self) -> str:
        return (
            f"RetroAdapter(in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, k={self.k}, "
            f"has_cache={self.has_cache})"
        )
