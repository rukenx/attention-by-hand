# Exercise 06: Multi-Query and Grouped-Query Attention

## Objective

Understand how multiple query heads can share fewer key/value heads, and separate the savings on the K/V side from the work that remains per query head.

The key principle is:

```math
\boxed{\text{reduce K/V heads, not query heads}}
```

## MHA, GQA, and MQA

Let `H_q` be the number of query heads and `H_kv` the number of key/value heads.

```math
\text{MHA: }H_{kv}=H_q,
\qquad
\text{GQA: }1<H_{kv}<H_q,
\qquad
\text{MQA: }H_{kv}=1.
```

Use

```math
H_q=4,
\qquad
H_{kv}^{\mathrm{GQA}}=2,
\qquad
H_{kv}^{\mathrm{MQA}}=1,
```

```math
d_{\mathrm{model}}=8,
\qquad d_k=d_v=2.
```

For GQA, the group size is

```math
G=\frac{H_q}{H_{kv}}=2,
```

so the 0-based mapping is

```text
[0, 0, 1, 1]
```

and MQA uses

```text
[0, 0, 0, 0]
```

## Shared K/V attention

For query head `h`,

```math
\mathrm{head}_h
=
\mathrm{softmax}\left(
\frac{Q_hK_{g(h)}^\top}{\sqrt{d_k}}
\right)V_{g(h)}.
```

Use the small shared-K/V example

```math
q_1=\begin{bmatrix}1&0\end{bmatrix},
\qquad
q_2=\begin{bmatrix}0&1\end{bmatrix},
```

```math
K_{\mathrm{shared}}=
\begin{bmatrix}1&0\\0&1\end{bmatrix},
\qquad
V_{\mathrm{shared}}=
\begin{bmatrix}1&0\\1&1\end{bmatrix}.
```

Different queries should produce different attention distributions even though K/V are shared.

## Projection shapes

Using row-vector mathematical notation:

```math
W_Q\in\mathbb{R}^{d_{\mathrm{model}}\times(H_qd_k)},
```

```math
W_K\in\mathbb{R}^{d_{\mathrm{model}}\times(H_{kv}d_k)},
\qquad
W_V\in\mathbb{R}^{d_{\mathrm{model}}\times(H_{kv}d_v)},
```

```math
W_O\in\mathbb{R}^{(H_qd_v)\times d_{\mathrm{model}}}.
```

For the toy dimensions:

| Architecture | Q width | K width | V width | Combined K/V weights |
|---|---:|---:|---:|---:|
| MHA | 8 | 8 | 8 | 128 |
| GQA | 8 | 4 | 4 | 64 |
| MQA | 8 | 2 | 2 | 32 |

The counts assume bias-free K/V projections.

## Tensor view

Use

```text
Q: [B, H_q,  L, D]
K: [B, H_kv, L, D]
V: [B, H_kv, L, D]
```

with

```text
B=1, H_q=4, H_kv=2, L=3, D=2.
```

The verifier uses `index_select` to **materialize** query-head-shaped K/V tensors for clarity. That is a teaching convenience, not a claim that an optimized GQA implementation must physically duplicate the KV cache. The physical stored K/V state remains `H_kv` heads.

## KV cache

Per layer,

```math
K_{\mathrm{cache}}\in\mathbb{R}^{B\times H_{kv}\times L\times d_k},
\qquad
V_{\mathrm{cache}}\in\mathbb{R}^{B\times H_{kv}\times L\times d_v}.
```

The element count is

```math
BLH_{kv}(d_k+d_v),
```

or across all layers,

```math
BN_{\mathrm{layers}}LH_{kv}(d_k+d_v).
```

For `B=1`, one layer, `L=3`, and `d_k=d_v=2`:

```text
MHA: 48 elements
GQA: 24 elements
MQA: 12 elements
```

## What gets cheaper

Reducing `H_kv` directly reduces:

- K/V projection width and K/V parameter count,
- KV-cache storage,
- K/V memory traffic during decoding.

`H_q` remains fixed, so query-head score/value work does not simply shrink in direct proportion to `H_kv`.

## Tasks

1. Define `H_q` and `H_kv`.
2. Derive MHA, GQA, and MQA head mappings for `H_q=4`.
3. Compute the shared-K/V numerical example for `q_1` and `q_2`.
4. Verify that the two attention distributions and outputs differ.
5. Derive the Q/K/V/W_O projection shapes.
6. Compute combined K/V weight counts for MHA, GQA, and MQA.
7. Write the physical K/V tensor shapes for the GQA example.
8. Explain the difference between the verifier's materialized `K_for_q`/`V_for_q` tensors and the physical KV cache.
9. Derive the per-layer and all-layer KV-cache element formulas.
10. Compute the `48 : 24 : 12` cache counts for the toy setup.
11. Explain what decreases with `H_kv` and what does not.

## Final question

How can MQA use one K/V head while retaining multiple distinct query-head attention patterns?

## Bonus

If sequence length `L` doubles, what happens to the cache sizes and to their relative MHA:GQA:MQA ratios?
