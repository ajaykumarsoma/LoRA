# LoRA

**LoRA from scratch on GPT-2 Small: rank-1 (0.044% of parameters) reduces perplexity from 9,237 to 316, and rank-64 (2.8%) beats full fine-tuning — demonstrating both parameter efficiency and LoRA's regularisation effect.**

---

## What Is LoRA?

Low-Rank Adaptation (Hu et al. 2021) freezes all pre-trained weights and adds a small trainable residual to each targeted linear layer:

```
W_new = W_frozen + ΔW    where ΔW = B @ A · (α / r)
```

- **A** ∈ ℝ^(d_in × r) — Kaiming initialised
- **B** ∈ ℝ^(r × d_out) — zero initialised → ΔW = 0 at step 0
- **r** (rank) controls the expressiveness of the update
- **α** is a fixed scaling factor (α/r normalises the learning rate across ranks)

The key insight: most weight updates during fine-tuning lie in a low-dimensional subspace. Rank 4–16 captures most of the relevant adaptation; higher ranks add diminishing returns.

---

## Implementation

`LoRALayer` wraps either `nn.Linear` or HuggingFace's `Conv1D` (which GPT-2 uses instead of `nn.Linear` — different weight shape convention):

```python
def forward(self, x):
    return self.layer(x) + (x @ self.lora_A @ self.lora_B) * self.scale
```

LoRA is applied to the attention `c_attn` (QKV) and `c_proj` (output) projections in every layer. The embedding matrix, layer norms, and FFN weights remain completely frozen.

---

## Experiment

- **Model:** GPT-2 Small (124M parameters)
- **Task:** Language modelling on Tiny Shakespeare (~300k tokens)
- **Ranks swept:** r = 1, 4, 16, 64
- **Steps:** 300 per configuration (≈ 150 seconds / rank on M4 CPU)
- **Baseline:** GPT-2 zero-shot on Shakespeare — PPL 9,237

---

## Results

| Method | Trainable params | % of total | PPL | ΔPPL vs baseline |
|---|---|---|---|---|
| Baseline (no FT) | 0 | 0% | 9,237 | — |
| LoRA r=1 | 55,296 | **0.044%** | 316 | −8,921 |
| LoRA r=4 | 221,184 | 0.178% | **242** | −8,995 |
| LoRA r=16 | 884,736 | 0.711% | 254 | −8,982 |
| LoRA r=64 | 3,538,944 | 2.844% | **229** | −9,008 |
| Full fine-tune | 124,439,808 | 100% | 260 | −8,977 |

**Key findings:**

1. **LoRA r=1 (0.044% of parameters) drops PPL from 9,237 to 316 — a 29× improvement.** The vast majority of the adaptation signal lies in an extremely low-dimensional subspace.

2. **LoRA r=64 beats full fine-tuning (229 vs 260 PPL) with 97% fewer trainable parameters.** At this step count, the low-rank constraint acts as implicit regularisation, preventing the full model from updating in noisy high-dimensional directions.

3. **Sweet spot at r=4:** achieves the second-best PPL (242) with only 0.178% of parameters — the best efficiency on the parameter-PPL curve.

4. **r=16 is slightly worse than r=4**, suggesting the benefit of higher rank is dataset- and step-count-dependent. At longer training, higher ranks typically converge to better performance.

---

## Plots

| Plot | What it shows |
|---|---|
| `ppl_vs_params.png` | PPL vs trainable parameters (log scale) — LoRA r=64 left of and below Full FT |
| `ppl_drop.png` | Bar chart: PPL improvement per method — r=64 is tallest bar |
| `efficiency_frontier.png` | PPL per million trainable params — r=1 is the most parameter-efficient |
| `recovery_pct.png` | What % of full FT's PPL gain each LoRA rank recovers |

---

## Limitations

- **Short training run (300 steps).** Results are directionally clear but full convergence would require 2,000–10,000 steps. At longer runs, the full FT model typically surpasses lower LoRA ranks.
- **Shakespeare is a small, specialised corpus.** The baseline PPL is very high because GPT-2 was trained on web text, not Early Modern English. The improvement magnitude is partly a reflection of this large distribution shift.
- **Only attention projections.** LoRA could also be applied to FFN layers; this experiment targets only `c_attn` and `c_proj` for simplicity.
- **CPU training.** M4 MPS triggers CPU-MPS syncs on custom `nn.Module` subclasses, making CPU faster for this pattern.

---

## Reference

Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. https://arxiv.org/abs/2106.09685
