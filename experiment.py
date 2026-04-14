"""
LoRA — Low-Rank Adaptation of GPT-2 from Scratch
==================================================
Implements LoRA (Hu et al. 2022) entirely from scratch — no PEFT library —
and applies it to fine-tune GPT-2 Small on Tiny Shakespeare.

Core idea:
  Instead of updating a weight matrix W ∈ R^(d×k) directly, LoRA adds a
  low-rank residual:  W' = W + (A @ B) * (alpha / rank)
  where A ∈ R^(d×r) and B ∈ R^(r×k), with r ≪ d.
  Only A and B are trained; W is frozen. Trainable parameters scale as
  O(r * (d + k)) instead of O(d * k).

Experiment:
  1. Load GPT-2 Small (117M params); freeze all weights.
  2. Inject LoRA adapters into all attention Q, K, V, O projections.
  3. Fine-tune on Shakespeare (train split) for 2,000 steps per rank.
  4. Sweep rank r ∈ {1, 2, 4, 8, 16, 32}.
  5. Measure validation perplexity before fine-tuning and after each rank.
  6. Compare trainable parameter count vs perplexity reduction.
  7. Run one full fine-tune (all params) as an upper-bound reference.

Key result expected:
  LoRA rank 4–8 reaches within ~5% of full fine-tune perplexity while
  training only ~0.3% of total parameters.

Device: MPS (Apple Silicon M4) → falls back to CPU if unavailable.
Hardware: ~8 min on M4 MacBook Air (all ranks + full fine-tune).
Reference: Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language
           Models. ICLR 2022.
"""
import os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D   # GPT-2 uses this, not nn.Linear

# ── Device ────────────────────────────────────────────────────────────────────
# M4 note: MPS is fast for pure-inference workloads, but custom nn.Module
# subclasses (like our LoRALayer) trigger many small CPU-MPS syncs during
# autograd, making MPS SLOWER than CPU for training in this setup.
# M4 has 4 high-performance cores ~4.4 GHz; CPU training is more efficient here.
DEVICE = torch.device("cpu")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
# Structure: MI-Projects/Finetuning/LoRA/experiment.py
#            MI-Projects/MinimalTransformer/data/shakespeare.txt
CORPUS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../MinimalTransformer/data/shakespeare.txt"))

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
LORA_RANKS   = [1, 4, 16, 64]  # 4 ranks — enough for a clear efficiency curve
LORA_ALPHA   = 32              # scaling factor; scale = alpha / rank
TRAIN_STEPS  = 300             # per rank — ~14 min total on M4 CPU
FULL_STEPS   = 300             # for full fine-tune reference
BATCH        = 8               # M4 CPU handles this well
SEQ_LEN      = 32              # short context: 32² attn = 4x faster than 64²
LR_LORA      = 3e-4
LR_FULL      = 1e-5            # lower LR for full fine-tune to avoid instability
EVAL_BATCHES = 10              # batches for val perplexity estimate

print("=" * 60)
print("LoRA Fine-tuning — GPT-2 Small on Tiny Shakespeare")
print("=" * 60)
print(f"  Device : {DEVICE}")
print(f"  Ranks  : {LORA_RANKS}")
print(f"  Steps  : {TRAIN_STEPS} per rank + {FULL_STEPS} full fine-tune")

# ── Data ──────────────────────────────────────────────────────────────────────
print("\nLoading tokenizer and corpus...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

text   = open(CORPUS, encoding="utf-8").read()
tokens = tokenizer.encode(text)
tokens = torch.tensor(tokens, dtype=torch.long)
split  = int(0.9 * len(tokens))
train_tok = tokens[:split]
val_tok   = tokens[split:]
print(f"  Train tokens: {len(train_tok):,}  |  Val tokens: {len(val_tok):,}")

def get_batch(split="train"):
    src = train_tok if split == "train" else val_tok
    ix  = torch.randint(len(src) - SEQ_LEN - 1, (BATCH,))
    x   = torch.stack([src[i : i + SEQ_LEN]     for i in ix]).to(DEVICE)
    y   = torch.stack([src[i + 1 : i + SEQ_LEN + 1] for i in ix]).to(DEVICE)
    return x, y

# ── LoRA module ───────────────────────────────────────────────────────────────
class LoRALayer(nn.Module):
    """
    Wraps a frozen Conv1D or nn.Linear and adds a trainable low-rank residual.

    GPT-2 (HuggingFace) uses Conv1D whose weight is stored as (d_in, d_out),
    so the forward is  x @ W + b.  nn.Linear stores weight as (d_out, d_in)
    and applies  x @ W.T + b.  Both produce the same output shape; we detect
    which we have and read dimensions accordingly.

    LoRA residual: (x @ A @ B) * scale   — same shape as base output.
    A: Kaiming init  |  B: zero init  → ΔW = 0 at step 0.
    """
    def __init__(self, layer, rank: int, alpha: float):
        super().__init__()
        self.layer = layer
        for p in self.layer.parameters():
            p.requires_grad = False

        if isinstance(layer, Conv1D):
            d_in, d_out = layer.weight.shape   # Conv1D: (d_in, d_out)
        else:
