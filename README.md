# Attention by Hand

> Transformer attention, worked through from first principles.

Small, deterministic exercises for understanding the matrix operations behind Transformer attention without hiding the arithmetic behind high-level APIs.

The sequence moves from **scaled dot-product self-attention** to modern autoregressive attention mechanisms, introducing one idea at a time.

## Exercises

| # | Mechanism | Key question |
|---:|---|---|
| 01 | [Scaled Dot-Product Self-Attention](01_scaled_dot_product_self_attention/) | How do `Q`, `K`, and `V` become an attention output? |
| 02 | [Causal Masked Self-Attention](02_causal_masked_self_attention/) | How does a causal mask prevent attention to future tokens? |
| 03 | [Multi-Head Self-Attention](03_multi_head_self_attention/) | How can different projection heads produce different attention patterns? |
| 04 | [Rotary Position Embedding (RoPE)](04_rotary_position_embedding_rope/) | How can position-dependent rotations of `Q` and `K` expose relative position? |
| 05 | [KV Cache for Autoregressive Decoding](05_kv_cache_autoregressive_decoding/) | How does cached decoding preserve the current-token output while avoiding repeated prefix work? |
| 06 | [Multi-Query and Grouped-Query Attention](06_multi_query_and_grouped_query_attention/) | How can multiple query heads share fewer K/V heads, and what does that save? |

## How to use the repository

Each exercise contains:

- `problem.md` — the numerical setup and tasks
- `solution_by_hand.pdf` — the worked derivation
- `solution_by_hand.tex` — the LaTeX source for the worked derivation
- `answer_key.py` — a deterministic PyTorch verifier

Recommended workflow:

1. Solve `problem.md` by hand.
2. Compare with `solution_by_hand.pdf`.
3. Run `answer_key.py` to verify the arithmetic, shapes, and key identities.

## Requirements

- Python 3.9+
- PyTorch
- Matplotlib for the RoPE figures in Exercise 04

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Run a verifier from the repository root, for example:

```bash
python 01_scaled_dot_product_self_attention/answer_key.py
```

GitHub Actions runs all six verifiers on pushes and pull requests.

## Design principles

- **One mechanism at a time.** Each exercise isolates a specific attention concept.
- **Small enough to inspect.** Shapes and intermediate values stay visible.
- **Math first.** PyTorch verifies the result; it does not replace the derivation.
- **Deterministic.** Tensors and projection matrices are fixed; nothing is trained.
- **Mechanics, not benchmarking.** The repository is for understanding attention, not reproducing production kernels or building a complete Transformer.
