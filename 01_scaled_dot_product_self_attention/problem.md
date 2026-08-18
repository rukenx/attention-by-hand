# Exercise 01: Scaled Dot-Product Self-Attention

## Objective

Compute one complete single-head self-attention operation by hand.

Use the three-token sequence:

```text
I like cats
```

with row-token notation and

```math
N=3,
\qquad
d_{\mathrm{model}}=d_k=d_v=2.
```

## Token embeddings

```math
x_I=\begin{bmatrix}1&0\end{bmatrix},
\qquad
x_{\mathrm{like}}=\begin{bmatrix}0&1\end{bmatrix},
\qquad
x_{\mathrm{cats}}=\begin{bmatrix}1&1\end{bmatrix}.
```

Therefore

```math
X=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}.
```

These are fixed toy vectors chosen so the arithmetic stays inspectable.

## Projection matrices

```math
W_Q=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\qquad
W_K=
\begin{bmatrix}
1&1\\
0&1
\end{bmatrix},
\qquad
W_V=
\begin{bmatrix}
1&0\\
1&1
\end{bmatrix}.
```

## Tasks

Compute, in order:

1. `Q = XW_Q`
2. `K = XW_K`
3. `V = XW_V`
4. `K^T`
5. raw scores `S = QK^T`
6. scaled scores `S_scaled = QK^T / sqrt(d_k)`
7. row-wise attention weights `A = softmax(S_scaled)`
8. the row sums of `A`
9. final output `O = AV`
10. one row of `A` in words

Write the shape of every matrix as you go.

## Final question

For the token `cats`, which token receives the largest attention weight?

## Bonus

Does the token receiving the largest attention weight necessarily correspond to the value vector with the largest norm? Explain why or why not.
