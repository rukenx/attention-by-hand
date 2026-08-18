# Exercise 04: Rotary Position Embedding (RoPE)

Use a controlled 2D example to see how position-dependent rotations of `Q` and `K` create relative-position structure in their dot products.

For row-vector notation, the exercise uses the identity

```math
q_m^R (k_n^R)^\top
=
q_m R((n-m)\theta) k_n^\top.
```

The simplified cosine pattern in the worked example is specific to identical unit base vectors; the matrix identity above is the general statement used by the exercise.

## Files

- `problem.md` — numerical setup and tasks
- `solution_by_hand.pdf` — worked derivation
- `solution_by_hand.tex` — LaTeX source
- `answer_key.py` — numerical verifier and figure generator
- `figures/` — generated RoPE diagrams used by the solution

Run from the repository root:

```bash
python 04_rotary_position_embedding_rope/answer_key.py
```
