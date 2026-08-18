# Exercise 04: Rotary Position Embedding (RoPE)

## Objective

Use a controlled 2D example to see how position-dependent rotations of `Q` and `K` make their dot products depend on relative position.

## Controlled setup

Use three positions `p = 0, 1, 2` and identical content vectors:

```math
Q=K=
\begin{bmatrix}
1&0\\
1&0\\
1&0
\end{bmatrix},
\qquad d_k=2.
```

Because content is held constant, any difference after RoPE comes from position alone. `V` is omitted because this exercise isolates query-key geometry.

## Baseline without position

```math
S_{\mathrm{plain}}=QK^\top=
\begin{bmatrix}
1&1&1\\
1&1&1\\
1&1&1
\end{bmatrix}.
```

After scaling and row-wise softmax, every row is uniform.

## RoPE convention

Use one 2D frequency with `theta = 1` radian. For position `p`, let

```math
\phi_p=p\theta.
```

For row-vector notation,

```math
\mathrm{RoPE}(q_p,p)=q_pR(\phi_p)^\top,
```

where

```math
R(\phi)=
\begin{bmatrix}
\cos\phi&-\sin\phi\\
\sin\phi&\cos\phi
\end{bmatrix}.
```

Real RoPE uses multiple dimension pairs and multiple frequencies; this exercise keeps only one pair to expose the geometry.

## Tasks

1. Compute `S_plain = QK^T`.
2. Scale by `sqrt(d_k)` and compute the plain attention matrix.
3. Compute `phi_0`, `phi_1`, and `phi_2`.
4. Construct the rotated `Q_R` and `K_R`.
5. Compute `S_RoPE = Q_R K_R^T`.
6. Compare score entries with equal relative offsets.
7. Verify the row-vector identity

   ```math
   q_m^R(k_n^R)^\top
   =
   q_mR((n-m)\theta)k_n^\top.
   ```
8. Scale `S_RoPE` by `sqrt(d_k)` and apply row-wise softmax.
9. Compare plain attention with RoPE attention.
10. Explain why rotations preserve vector norms.

## Final question

RoPE uses absolute position indices to rotate `Q` and `K`. Why can their resulting dot product nevertheless depend on relative position?
