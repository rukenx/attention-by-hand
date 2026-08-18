# Exercise 02: Causal Masked Self-Attention

## Objective

Reuse Exercise 01 and add one new mechanism: a causal mask that prevents each token from attending to future positions.

Use the same sequence, embeddings, dimensions, and projection matrices as Exercise 01:

```text
I like cats
```

```math
X=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},
\qquad
d_k=2.
```

```math
W_Q=
\begin{bmatrix}1&0\\0&1\end{bmatrix},
\quad
W_K=
\begin{bmatrix}1&1\\0&1\end{bmatrix},
\quad
W_V=
\begin{bmatrix}1&0\\1&1\end{bmatrix}.
```

## Causal rule

For row `i` and column `j`, use

```math
M_{ij}=
\begin{cases}
0, & j\le i,\\
-\infty, & j>i.
\end{cases}
```

The masked attention operation is

```math
O_{\mathrm{causal}}
=
\mathrm{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V.
```

The mask is added **before** softmax.

## Tasks

1. Recompute `Q`, `K`, and `V`.
2. Compute `S = QK^T` and `S_scaled = S / sqrt(d_k)`.
3. Construct the `3 x 3` causal mask `M`.
4. Identify each future-token connection removed by the mask.
5. Compute `S_masked = S_scaled + M`.
6. Compute `A_causal = softmax(S_masked)` row-wise.
7. Verify that every forbidden future position has probability exactly `0`.
8. Verify that every row of `A_causal` sums to approximately `1`.
9. Compute `O_causal = A_causal V`.
10. Compare the causal attention matrix and output with Exercise 01.

Write the shape of every matrix as you go.

## Final question

Why is the attention row for `cats` unchanged while the rows for `I` and `like` change?

## Bonus

Why is zeroing forbidden probabilities **after** softmax not equivalent to masking their logits with `-inf` **before** softmax?
