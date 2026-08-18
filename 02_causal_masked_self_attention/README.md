# Exercise 02: Causal Masked Self-Attention

Add autoregressive visibility constraints to the scaled dot-product attention from Exercise 01.

The key operation is to add a causal mask to the scaled logits **before** softmax so position `i` can attend only to positions `j <= i`.

## What you compute

`QK^T -> scale -> causal mask -> softmax -> output`

The exercise verifies that future-token probabilities are exactly zero and explains why masking after softmax is not equivalent.

## Files

- `problem.md` — numerical setup and tasks
- `solution_by_hand.pdf` — worked derivation
- `solution_by_hand.tex` — LaTeX source
- `answer_key.py` — deterministic PyTorch verification

Run from the repository root:

```bash
python 02_causal_masked_self_attention/answer_key.py
```
