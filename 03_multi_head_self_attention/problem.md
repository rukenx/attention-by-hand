# Exercise 03: Multi-Head Self-Attention

## Objective

Extend self-attention from one head to two independent heads, concatenate their outputs, and apply an output projection.

Both heads receive the same input `X`, but they use different fixed projection matrices. The example demonstrates how different projected geometries can produce different attention patterns; the hand-designed heads are not assigned learned semantic roles.

## Input

```text
I like cats
```

```math
X=
\begin{bmatrix}
1&0&1&0\\
0&1&1&1\\
1&1&0&1
\end{bmatrix}
\in\mathbb{R}^{3\times4}.
```

Use

```math
H=2,
\qquad
d_k=d_v=2,
\qquad d_{\mathrm{model}}=4.
```

## Head 1

```math
W_Q^{(1)}=W_K^{(1)}=W_V^{(1)}=
\begin{bmatrix}
1&0\\
0&1\\
0&0\\
0&0
\end{bmatrix}.
```

## Head 2

```math
W_Q^{(2)}=
\begin{bmatrix}
0&0\\
0&0\\
1&0\\
0&1
\end{bmatrix},
\quad
W_K^{(2)}=
\begin{bmatrix}
0&0\\
0&0\\
0&1\\
1&0
\end{bmatrix},
\quad
W_V^{(2)}=
\begin{bmatrix}
0&0\\
0&0\\
1&0\\
0&1
\end{bmatrix}.
```

## Output projection

```math
W_O=I_4.
```

The identity is only a toy simplification; a trained Transformer uses a learned output projection.

## Formula

For each head `h`,

```math
Q_h=XW_Q^{(h)},\qquad
K_h=XW_K^{(h)},\qquad
V_h=XW_V^{(h)},
```

```math
\mathrm{head}_h
=
\mathrm{softmax}\left(
\frac{Q_hK_h^\top}{\sqrt{d_k}}
\right)V_h.
```

Then

```math
O_{\mathrm{MHA}}
=
\mathrm{Concat}(\mathrm{head}_1,\mathrm{head}_2)W_O.
```

## Tasks

1. Compute `Q_1`, `K_1`, `V_1`.
2. Compute Head 1 scores, attention weights, and `O_1`.
3. Compute `Q_2`, `K_2`, `V_2`.
4. Compute Head 2 scores, attention weights, and `O_2`.
5. Compare `A_1` and `A_2` and explain why they differ.
6. Concatenate `O_1` and `O_2` along the feature dimension.
7. Apply `W_O`.
8. Verify all shapes and attention row sums.

## Final question

Why can the two heads produce different attention weights even though they receive the same input matrix `X`?

## Bonus

If this were causal multi-head self-attention, where would the causal mask from Exercise 02 be applied?
