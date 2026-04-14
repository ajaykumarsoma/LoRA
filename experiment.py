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
