"""
retro-peft-adapters demo
========================
Demonstrates retrieval-augmented adaptation vs vanilla LoRA on a toy task.

Task: given a 32-dim input, predict which of 4 "domain buckets" it belongs to.
      Each domain has a distinct bias pattern. The retrieval cache stores
      domain-specific prototype vectors (key=input cluster center, value=bias).

Run with:
    ~/anaconda3/bin/python3 examples/demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from retro_peft import (
    KeyValueCache, CacheBuilder, RetroAdapter, AdapterBank, RetroPEFT
)

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# 1. Toy dataset: 4 domains, each with a distinct centroid in 32-dim space
# ---------------------------------------------------------------------------
D_IN = 32
D_HID = 64
D_OUT = 4
N_DOMAINS = 4
N_TRAIN = 200
N_TEST = 50

DOMAIN_CENTERS = torch.randn(N_DOMAINS, D_IN) * 3

def make_data(n: int, domain: int):
    """Generate n samples from a given domain."""
    center = DOMAIN_CENTERS[domain]
    x = center.unsqueeze(0) + torch.randn(n, D_IN) * 0.5
    # Label = domain index
    y = torch.full((n,), domain, dtype=torch.long)
    return x, y


def eval_model(model, all_x, all_y):
    model.eval()
    with torch.no_grad():
        logits = model(all_x)
        preds = logits.argmax(dim=-1)
        acc = (preds == all_y).float().mean().item()
    return acc


def train(model, x_train, y_train, n_epochs=200, lr=1e-2, label="model"):
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()
    return model


# ---------------------------------------------------------------------------
# 2. Build dataset
# ---------------------------------------------------------------------------
print("=" * 60)
print("retro-peft-adapters demo")
print("=" * 60)
print(f"\nTask: classify inputs into {N_DOMAINS} domains ({D_IN}-dim input)")
print(f"      Each domain has a distinct cluster center.\n")

all_x_train, all_y_train, all_x_test, all_y_test = [], [], [], []
for d in range(N_DOMAINS):
    x_tr, y_tr = make_data(N_TRAIN // N_DOMAINS, d)
    x_te, y_te = make_data(N_TEST // N_DOMAINS, d)
    all_x_train.append(x_tr); all_y_train.append(y_tr)
    all_x_test.append(x_te);  all_y_test.append(y_te)

x_train = torch.cat(all_x_train)
y_train = torch.cat(all_y_train)
x_test  = torch.cat(all_x_test)
y_test  = torch.cat(all_y_test)

# ---------------------------------------------------------------------------
# 3. Baseline: full fine-tuning
# ---------------------------------------------------------------------------
print("─" * 40)
print("[1] Full fine-tuning (all params trainable)")

base = nn.Sequential(
    nn.Linear(D_IN, D_HID), nn.ReLU(), nn.Linear(D_HID, D_OUT)
)
train(base, x_train, y_train, label="full")
acc_full = eval_model(base, x_test, y_test)
n_params_full = sum(p.numel() for p in base.parameters())
print(f"    Accuracy: {acc_full:.1%}  |  Params: {n_params_full:,}")

# ---------------------------------------------------------------------------
# 4. Vanilla LoRA (RetroPEFT without cache)
# ---------------------------------------------------------------------------
print("\n─" * 40)
print("[2] Vanilla LoRA via RetroPEFT (no retrieval cache)")

base2 = nn.Sequential(
    nn.Linear(D_IN, D_HID), nn.ReLU(), nn.Linear(D_HID, D_OUT)
)
peft_lora = RetroPEFT(base2, target_modules=["0", "2"], rank=4, lora_alpha=8.0)
train(peft_lora, x_train, y_train, lr=1e-2, label="lora")
acc_lora = eval_model(peft_lora, x_test, y_test)
n_train_lora = peft_lora.trainable_param_count()
n_total_lora = peft_lora.total_param_count()
print(f"    Accuracy: {acc_lora:.1%}  |  "
      f"Trainable: {n_train_lora:,}/{n_total_lora:,} "
      f"({peft_lora.compression_ratio():.1%} of params)")

# ---------------------------------------------------------------------------
# 5. RetroLoRA — LoRA + retrieval cache
# ---------------------------------------------------------------------------
print("\n─" * 40)
print("[3] RetroLoRA (LoRA + retrieval-augmented cache)")

# Build the cache: keys = domain centers, values = domain bias vectors
# In practice, you'd build this from real corpus embeddings.
# Here: keys are the domain cluster centers (32-dim),
#       values are random domain-specific "knowledge vectors" (64-dim).
torch.manual_seed(7)
domain_keys   = DOMAIN_CENTERS                         # (4, 32) — the cluster centers
domain_values = torch.randn(N_DOMAINS, D_HID) * 2.0   # (4, 64) — domain knowledge

cache = CacheBuilder.from_tensors(domain_keys, domain_values)
print(f"    Cache: {cache}")

base3 = nn.Sequential(
    nn.Linear(D_IN, D_HID), nn.ReLU(), nn.Linear(D_HID, D_OUT)
)
peft_retro = RetroPEFT(
    base3, target_modules=["0", "2"], rank=4, lora_alpha=8.0, k=2,
    retrieval_gate_init=-1.0,  # start with low gate, let training tune it
)
# Attach cache only to layer "0" (maps D_IN → D_HID, same dims as cache)
peft_retro.set_cache({"0": cache})

train(peft_retro, x_train, y_train, lr=1e-2, label="retro")
acc_retro = eval_model(peft_retro, x_test, y_test)
gate_val = torch.sigmoid(peft_retro.adapters["0"].retrieval_gate).item()
print(f"    Accuracy: {acc_retro:.1%}  |  "
      f"Trainable: {peft_retro.trainable_param_count():,}/{peft_retro.total_param_count():,} "
      f"({peft_retro.compression_ratio():.1%} of params)")
print(f"    Learned retrieval gate: {gate_val:.3f}  "
      f"({'active' if gate_val > 0.4 else 'suppressed'})")

# ---------------------------------------------------------------------------
# 6. AdapterBank: hot-swap domains without retraining
# ---------------------------------------------------------------------------
print("\n─" * 40)
print("[4] AdapterBank: per-domain caches, shared LoRA weights")

bank = AdapterBank(D_IN, D_HID, rank=4, k=2, retrieval_gate_init=0.0)
for d in range(N_DOMAINS):
    # Each domain gets its own small cache (just 1 entry here for illustration)
    d_cache = CacheBuilder.from_tensors(
        DOMAIN_CENTERS[d].unsqueeze(0),  # (1, 32)
        domain_values[d].unsqueeze(0),   # (1, 64)
    )
    bank.register_domain(f"domain_{d}", d_cache)

print(f"    Bank domains: {bank.domains}")
for d in range(N_DOMAINS):
    bank.activate(f"domain_{d}")
    x_d, _ = make_data(5, d)
    out = bank(x_d)
    print(f"    domain_{d}: output shape={tuple(out.shape)}  "
          f"(gate={torch.sigmoid(bank.adapter.retrieval_gate).item():.3f})")

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
rows = [
    ("Full fine-tuning",       acc_full,  n_params_full, "—"),
    ("Vanilla LoRA",           acc_lora,  n_train_lora,  f"{peft_lora.compression_ratio():.1%}"),
    ("RetroLoRA",              acc_retro, peft_retro.trainable_param_count(), f"{peft_retro.compression_ratio():.1%}"),
]
print(f"{'Method':<22} {'Accuracy':>10} {'Trainable params':>18} {'Ratio':>8}")
print("-" * 62)
for method, acc, params, ratio in rows:
    print(f"{method:<22} {acc:>10.1%} {params:>18,} {ratio:>8}")

print("\n✓ RetroLoRA adapts with <{:.0f}% of full model params.".format(
    peft_retro.compression_ratio() * 100
))
print("✓ Cache swap (AdapterBank) requires zero retraining.")
