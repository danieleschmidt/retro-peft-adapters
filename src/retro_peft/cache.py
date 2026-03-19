"""
KeyValueCache: frozen key/value memory store for retrieval-augmented PEFT.

Stores (key_embedding, value) pairs and retrieves top-k by cosine similarity.
The cache is frozen at inference time — keys and values are not updated during
fine-tuning, which is what makes this PEFT-compatible.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


class KeyValueCache:
    """
    Frozen key/value memory store.

    Keys are embedding vectors; values are arbitrary tensors (e.g., hidden states,
    class embeddings, domain vectors). At retrieval time, a query embedding is
    compared to all keys via cosine similarity and the top-k values are returned.

    The cache is "frozen" in the sense that it is not updated by gradient descent.
    It is built once (see CacheBuilder) and then used read-only during adaptation.

    Args:
        keys: Float tensor of shape (N, d_key) — the key embeddings.
        values: Float tensor of shape (N, d_val) — the associated values.
        device: Target device. Defaults to CPU.
    """

    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> None:
        if keys.shape[0] != values.shape[0]:
            raise ValueError(
                f"keys and values must have the same number of entries: "
                f"got keys={keys.shape[0]}, values={values.shape[0]}"
            )
        device = device or torch.device("cpu")
        # Normalize keys up-front for fast cosine similarity via dot product.
        self._keys = F.normalize(keys.float(), dim=-1).to(device)
        self._values = values.float().to(device)
        self.device = device

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of (key, value) pairs in the cache."""
        return self._keys.shape[0]

    @property
    def key_dim(self) -> int:
        return self._keys.shape[1]

    @property
    def value_dim(self) -> int:
        return self._values.shape[1]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: torch.Tensor,
        k: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the top-k values for one or more query embeddings.

        Args:
            query: Float tensor of shape (..., d_key). The query is normalized
                   internally so raw or pre-normalized vectors both work.
            k: Number of nearest neighbours to return.

        Returns:
            values: Tensor of shape (..., k, d_val) — retrieved value vectors,
                    ordered by descending similarity.
            scores: Tensor of shape (..., k) — cosine similarity scores.
        """
        k = min(k, self.size)
        original_shape = query.shape[:-1]
        query_flat = query.reshape(-1, query.shape[-1]).to(self.device)
        query_norm = F.normalize(query_flat.float(), dim=-1)

        # Cosine similarity via matrix multiply (keys are already normalized).
        # sim: (B, N)
        sim = query_norm @ self._keys.T

        scores_flat, indices_flat = sim.topk(k, dim=-1)
        values_flat = self._values[indices_flat]  # (B, k, d_val)

        values_out = values_flat.reshape(*original_shape, k, self.value_dim)
        scores_out = scores_flat.reshape(*original_shape, k)

        return values_out, scores_out

    def to(self, device: torch.device) -> "KeyValueCache":
        """Move cache to a device (returns self for chaining)."""
        self._keys = self._keys.to(device)
        self._values = self._values.to(device)
        self.device = device
        return self

    def __repr__(self) -> str:
        return (
            f"KeyValueCache(size={self.size}, "
            f"key_dim={self.key_dim}, value_dim={self.value_dim}, "
            f"device={self.device})"
        )
