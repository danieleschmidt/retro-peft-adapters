# retro-peft-adapters

Retrieval-Augmented Parameter-Efficient Fine-Tuning (PEFT).

Combines LoRA-style low-rank adaptation with a frozen key/value retrieval cache. The adapter output is a learned blend of:
1. **LoRA output** — parametric adaptation via trainable low-rank matrices
2. **Retrieved value vectors** — non-parametric domain knowledge from a frozen cache

Swapping the cache at inference time adapts the model to a new domain without any retraining.

## Concepts

```
Input x
  │
  ├──► base(x)            ← frozen pre-trained linear
  │
  ├──► lora_B(lora_A(x)) ← trainable LoRA, scale = α/r
  │
  └──► gate · Σ wᵢ vᵢ    ← retrieval: top-k values from cache,
                              weighted by cosine similarity,
                              gated by a learned scalar
```

The retrieval cache is built once from a corpus and frozen. Only the LoRA matrices and the retrieval gate are trained.

## Components

| Class | Description |
|---|---|
| `KeyValueCache` | Frozen store of (key_embedding, value) pairs. Retrieves top-k by cosine similarity. |
| `CacheBuilder` | Builds a `KeyValueCache` from raw (input, output) corpus pairs with optional encoder and pooling. |
| `RetroAdapter` | Drop-in replacement for `nn.Linear`. LoRA + retrieval, with a learned gate. |
| `AdapterBank` | Manages multiple `RetroAdapter`s for different domains. Hot-swap caches at inference time. |
| `RetroPEFT` | Wraps any `nn.Module`, injects `RetroAdapter`s at target layers by name. |

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.8 and PyTorch ≥ 1.12.

## Quick Start

```python
import torch
import torch.nn as nn
from retro_peft import KeyValueCache, CacheBuilder, RetroAdapter, RetroPEFT

# 1. Build a cache from domain-specific key/value pairs
keys   = torch.randn(100, 64)   # 100 prototype embeddings
values = torch.randn(100, 128)  # associated value vectors
cache  = CacheBuilder.from_tensors(keys, values)

# 2. Wrap a model — injects RetroAdapters at layers matching "fc" or "proj"
base_model = nn.Sequential(
    nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)
)
peft = RetroPEFT(base_model, target_modules=["0", "2"], rank=8, k=4)

# 3. Attach cache to the first layer (key_dim=64, value_dim=128)
peft.set_cache({"0": cache})

# 4. Only LoRA + gate params are trainable
print(f"Trainable: {peft.trainable_param_count():,} / {peft.total_param_count():,}")

# 5. Forward pass
x = torch.randn(32, 64)
out = peft(x)  # shape: (32, 10)
```

## Domain Hot-Swapping with AdapterBank

```python
from retro_peft import AdapterBank, CacheBuilder

bank = AdapterBank(in_features=64, out_features=128, rank=8, k=4)

# Register domains with their caches
for domain, (keys, values) in domain_data.items():
    cache = CacheBuilder.from_tensors(keys, values)
    bank.register_domain(domain, cache)

# Activate a domain — no retraining needed
bank.activate("medical")
out = bank(x)

bank.activate("legal")
out = bank(x)   # same LoRA weights, different cache
```

## Building Caches

```python
from retro_peft import CacheBuilder

# From pre-computed tensors
cache = CacheBuilder.from_tensors(keys, values)

# From a list of (key, value) tensor pairs
cache = CacheBuilder.from_pairs([(k1, v1), (k2, v2), ...])

# With an encoder and incremental batches
builder = CacheBuilder(key_encoder=my_encoder, pool="mean")
for batch_inputs, batch_outputs in corpus:
    builder.add(batch_inputs, batch_outputs)
cache = builder.build()
```

## Demo

```bash
python examples/demo.py
```

Trains a small classifier with full fine-tuning, vanilla LoRA, and RetroLoRA on a 4-domain toy task. Shows that RetroLoRA matches full fine-tuning accuracy using ~22% of the parameters.

## Tests

```bash
pytest tests/ -v
```

53 tests covering `KeyValueCache`, `CacheBuilder`, `RetroAdapter`, `AdapterBank`, and `RetroPEFT`.

## Design Decisions

**Why freeze the cache?**  
LoRA handles the parametric adaptation that generalizes across examples. The cache carries frozen domain knowledge — it's cheap to swap and requires no gradient updates.

**Why cosine similarity?**  
Keys are normalized at construction time. Retrieval is a matrix multiply — fast and scale-invariant.

**Why a learned gate?**  
The gate lets the model learn how much to trust retrieval vs LoRA per adapter. It starts near zero and grows only if retrieval is useful for the task.

**Why shared LoRA in AdapterBank?**  
One set of LoRA weights captures general adaptation patterns; domain specificity lives in the cache. This minimizes parameters when serving many domains.

## License

MIT
