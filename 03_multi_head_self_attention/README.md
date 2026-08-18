# Exercise 03: Multi-Head Self-Attention

Compute self-attention with two independent heads that receive the same input but use different projection matrices.

The exercise isolates three ideas:

1. each head has its own `Q`, `K`, and `V` projections,
2. different projections can produce different attention patterns,
3. head outputs are concatenated and passed through `W_O`.

The toy `W_O` is the identity matrix so the effect of concatenation stays visible. No semantic role is assigned to either hand-designed head; the example demonstrates geometry, not learned specialization.

## Files

- `problem.md` — numerical setup and tasks
- `solution_by_hand.pdf` — worked derivation
- `solution_by_hand.tex` — LaTeX source
- `answer_key.py` — deterministic PyTorch verification

Run from the repository root:

```bash
python 03_multi_head_self_attention/answer_key.py
```
