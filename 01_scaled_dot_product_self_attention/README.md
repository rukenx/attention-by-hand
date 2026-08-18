# Exercise 01: Scaled Dot-Product Self-Attention

Compute one complete single-head self-attention pass with small matrices that can be checked by hand.

This exercise isolates:

```math
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

There is no causal mask, positional encoding, multi-head structure, or training.

## What you compute

`X -> Q, K, V -> QK^T -> scale -> softmax -> output`

## Files

- `problem.md` — numerical setup and tasks
- `solution_by_hand.pdf` — worked derivation
- `solution_by_hand.tex` — LaTeX source
- `answer_key.py` — deterministic PyTorch verification

Run from the repository root:

```bash
python 01_scaled_dot_product_self_attention/answer_key.py
```
