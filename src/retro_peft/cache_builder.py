"""
CacheBuilder: build a KeyValueCache from a corpus of (key, value) pairs.

In a real scenario, keys might be sentence embeddings (from a frozen encoder)
and values might be hidden-state vectors or domain-specific representations.
CacheBuilder handles batching, average pooling, and optional deduplication.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple

from .cache import KeyValueCache


class CacheBuilder:
    """
    Build a KeyValueCache from raw (input, output) corpus pairs.

    Usage pattern:
        builder = CacheBuilder(key_encoder=my_encoder, value_encoder=my_encoder)
        for batch_inputs, batch_outputs in corpus:
            builder.add(batch_inputs, batch_outputs)
        cache = builder.build()

    Or from pre-computed tensors:
        cache = CacheBuilder.from_tensors(keys, values)

    Args:
        key_encoder: Callable that maps raw inputs → key embeddings (N, d_key).
                     If None, raw inputs must already be tensors.
        value_encoder: Callable that maps raw outputs → value tensors (N, d_val).
                       If None, raw outputs must already be tensors.
        pool: Pooling strategy for sequence inputs. "mean" (default) or "cls".
        device: Device for the final cache.
    """

    def __init__(
        self,
        key_encoder: Optional[Callable] = None,
        value_encoder: Optional[Callable] = None,
        pool: str = "mean",
        device: Optional[torch.device] = None,
    ) -> None:
        self.key_encoder = key_encoder
        self.value_encoder = value_encoder
        self.pool = pool
        self.device = device or torch.device("cpu")

        self._keys: List[torch.Tensor] = []
        self._values: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Building from raw pairs
    # ------------------------------------------------------------------

    def add(self, inputs, outputs) -> "CacheBuilder":
        """
        Add a batch of (input, output) pairs.

        Args:
            inputs: Raw inputs (passed to key_encoder) or a tensor of shape (N, d_key).
            outputs: Raw outputs (passed to value_encoder) or tensor of shape (N, d_val).

        Returns:
            self, for chaining.
        """
        if self.key_encoder is not None:
            keys = self.key_encoder(inputs)
        else:
            keys = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs)

        if self.value_encoder is not None:
            values = self.value_encoder(outputs)
        else:
            values = outputs if isinstance(outputs, torch.Tensor) else torch.tensor(outputs)

        keys = self._pool(keys)
        values = self._pool(values)

        self._keys.append(keys.float().cpu())
        self._values.append(values.float().cpu())
        return self

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce sequence dimension if present."""
        if x.dim() == 3:
            if self.pool == "cls":
                return x[:, 0, :]
            else:  # mean
                return x.mean(dim=1)
        return x

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self) -> KeyValueCache:
        """Concatenate all accumulated pairs and return a KeyValueCache."""
        if not self._keys:
            raise RuntimeError("No data has been added to the CacheBuilder.")
        keys = torch.cat(self._keys, dim=0)
        values = torch.cat(self._values, dim=0)
        return KeyValueCache(keys, values, device=self.device)

    @staticmethod
    def from_tensors(
        keys: torch.Tensor,
        values: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> KeyValueCache:
        """
        Convenience constructor: build a KeyValueCache directly from tensors.

        Args:
            keys: Shape (N, d_key).
            values: Shape (N, d_val).
            device: Target device.
        """
        return KeyValueCache(keys, values, device=device)

    @staticmethod
    def from_pairs(
        pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        device: Optional[torch.device] = None,
    ) -> KeyValueCache:
        """
        Build a KeyValueCache from a list of (key, value) tensor pairs.

        Each pair is a tuple of 1-D tensors (d_key,) and (d_val,).
        """
        keys = torch.stack([p[0] for p in pairs], dim=0)
        values = torch.stack([p[1] for p in pairs], dim=0)
        return KeyValueCache(keys, values, device=device)
