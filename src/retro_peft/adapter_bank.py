"""
AdapterBank: manages multiple RetroAdapters for different domains.

Allows hot-swapping caches at inference time — load a different domain cache
without touching the LoRA weights. The LoRA weights capture general adaptation
patterns; the cache carries domain-specific knowledge.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .adapter import RetroAdapter
from .cache import KeyValueCache


class AdapterBank(nn.Module):
    """
    A named collection of RetroAdapters with hot-swappable domain caches.

    One RetroAdapter is "active" at any point. Switching domains means:
      1. Load the domain's KeyValueCache into the active adapter.
      2. (Optionally) swap in domain-specific LoRA weights if available.

    By default, all adapters share a single set of LoRA weights (parameter
    efficient). With ``shared_lora=False``, each domain gets its own LoRA.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        domains: List of domain names to pre-register (caches added later).
        shared_lora: If True, one adapter's LoRA weights are shared across all
                     domains (default). If False, each domain gets its own adapter.
        rank: LoRA rank.
        lora_alpha: LoRA scaling factor.
        k: Top-k neighbours to retrieve.
        **adapter_kwargs: Additional kwargs forwarded to RetroAdapter.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        domains: Optional[List[str]] = None,
        shared_lora: bool = True,
        rank: int = 8,
        lora_alpha: float = 16.0,
        k: int = 4,
        **adapter_kwargs,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.shared_lora = shared_lora
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.k = k

        self._domains: Dict[str, Optional[KeyValueCache]] = {}
        self._active_domain: Optional[str] = None

        if shared_lora:
            # One adapter whose cache is swapped per domain
            self.adapter = RetroAdapter(
                in_features, out_features,
                rank=rank, lora_alpha=lora_alpha, k=k,
                **adapter_kwargs,
            )
            self._domain_adapters: Optional[nn.ModuleDict] = None
        else:
            self.adapter = None  # type: ignore[assignment]
            self._domain_adapters = nn.ModuleDict()

        if domains:
            for d in domains:
                self.register_domain(d)

    # ------------------------------------------------------------------
    # Domain management
    # ------------------------------------------------------------------

    def register_domain(
        self,
        name: str,
        cache: Optional[KeyValueCache] = None,
    ) -> None:
        """
        Register a domain (optionally with its cache).

        Args:
            name: Domain identifier string.
            cache: Optional pre-built KeyValueCache for this domain.
        """
        if name in self._domains:
            raise ValueError(f"Domain '{name}' is already registered.")
        self._domains[name] = cache

        if not self.shared_lora:
            # Create a dedicated adapter for this domain
            assert self._domain_adapters is not None
            self._domain_adapters[name] = RetroAdapter(
                self.in_features, self.out_features,
                rank=self.rank, lora_alpha=self.lora_alpha, k=self.k,
                cache=cache,
            )

    def set_cache(self, domain: str, cache: KeyValueCache) -> None:
        """Attach or replace the cache for a registered domain."""
        if domain not in self._domains:
            raise KeyError(f"Domain '{domain}' is not registered. Call register_domain first.")
        self._domains[domain] = cache
        if not self.shared_lora and self._domain_adapters is not None:
            self._domain_adapters[domain].set_cache(cache)  # type: ignore[union-attr]

    def activate(self, domain: str) -> None:
        """
        Switch the active domain.

        For shared-LoRA mode: swaps the cache on the single adapter.
        For per-domain mode: routes forward() to the domain's adapter.
        """
        if domain not in self._domains:
            raise KeyError(f"Domain '{domain}' not found. Registered: {list(self._domains)}")
        self._active_domain = domain
        if self.shared_lora:
            cache = self._domains[domain]
            if cache is not None:
                self.adapter.set_cache(cache)
            else:
                self.adapter.clear_cache()

    def deactivate(self) -> None:
        """Deactivate domain retrieval (LoRA-only mode)."""
        self._active_domain = None
        if self.shared_lora:
            self.adapter.clear_cache()

    @property
    def active_domain(self) -> Optional[str]:
        return self._active_domain

    @property
    def domains(self) -> List[str]:
        return list(self._domains)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shared_lora:
            return self.adapter(x)
        else:
            assert self._domain_adapters is not None
            if self._active_domain is None:
                raise RuntimeError(
                    "No active domain set. Call activate(domain) before forward()."
                )
            return self._domain_adapters[self._active_domain](x)

    def __repr__(self) -> str:
        return (
            f"AdapterBank(domains={self.domains}, "
            f"active='{self.active_domain}', "
            f"shared_lora={self.shared_lora})"
        )
