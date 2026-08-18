# Exercise 05: KV Cache for Autoregressive Decoding

## Objective

Compare one autoregressive decode step **with** and **without** a KV cache, and verify that caching changes the amount of repeated work without changing the newest token's attention output.

Start from the three-token prompt used in Exercise 01:

```text
I like cats
```

Treat these tokens as the **prefill**. Then append one artificial decode token:

```text
today
```

The full sequence is therefore:

```text
I like cats today
```

> **Scope note:** RoPE and other positional mechanisms are intentionally omitted so the KV-cache mechanism can be isolated. Real models must cache K/V in the positional form required by their implementation.

## Numerical setup

Use:

```math
d_{\mathrm{model}} = d_k = d_v = 2.
```

Projection matrices:

```math
W_Q =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix},
\qquad
W_K =
\begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
\qquad
W_V =
\begin{bmatrix}
1 & 0 \\
1 & 1
\end{bmatrix}.
```

Embeddings:

```math
x_I=\begin{bmatrix}1&0\end{bmatrix},\quad
x_{\mathrm{like}}=\begin{bmatrix}0&1\end{bmatrix},\quad
x_{\mathrm{cats}}=\begin{bmatrix}1&1\end{bmatrix},\quad
x_{\mathrm{today}}=\begin{bmatrix}0&1\end{bmatrix}.
```

## Part A — Prefill

1. Form
   ```math
   X_{\mathrm{past}}=
   \begin{bmatrix}
   x_I\\x_{\mathrm{like}}\\x_{\mathrm{cats}}
   \end{bmatrix}.
   ```
2. Compute `Q_past`, `K_past`, and `V_past`.
3. State which projected tensors need to be retained for future decode steps.

## Part B — No-cache reference decode

Append `today` and form `X_full` with four rows.

4. Recompute
   ```math
   Q_{\mathrm{full}}=X_{\mathrm{full}}W_Q,\quad
   K_{\mathrm{full}}=X_{\mathrm{full}}W_K,\quad
   V_{\mathrm{full}}=X_{\mathrm{full}}W_V.
   ```
5. Identify the old K/V rows that were recomputed even though their inputs did not change.
6. Compute the **full** scaled score matrix
   ```math
   S_{\mathrm{full}}
   =
   \frac{Q_{\mathrm{full}}K_{\mathrm{full}}^\top}{\sqrt{d_k}}.
   ```
7. Construct the `4 x 4` causal mask and add it before softmax.
8. Compute the full causal attention matrix `A_full`.
9. Compute
   ```math
   O_{\mathrm{full}}=A_{\mathrm{full}}V_{\mathrm{full}}.
   ```
10. Extract the newest-token output
    ```math
    o_t^{(\mathrm{no-cache})}=O_{\mathrm{full}}[-1].
    ```

This branch deliberately recomputes the whole prefix so that its attention interaction work for a length-`t` call is visibly quadratic in sequence length.

## Part C — Cached decode

Return to the prefill state.

11. Initialize
    ```math
    K_{\mathrm{cache}}=K_{\mathrm{past}},\qquad
    V_{\mathrm{cache}}=V_{\mathrm{past}}.
    ```
12. Compute only `q_t`, `k_t`, and `v_t` for `today`.
13. Append `k_t` and `v_t` to the caches.
14. Compute
    ```math
    s_t^{(\mathrm{cache})}
    =
    \frac{q_tK_{\mathrm{cache}}^\top}{\sqrt{d_k}}.
    ```
15. Apply softmax to obtain the newest-token attention weights.
16. Compute
    ```math
    o_t^{(\mathrm{cache})}
    =
    a_t^{(\mathrm{cache})}V_{\mathrm{cache}}.
    ```

## Part D — Equivalence

17. Verify numerically that
    ```math
    o_t^{(\mathrm{no-cache})}
    =
    o_t^{(\mathrm{cache})}.
    ```
18. Verify that the newest row of `A_full` equals the cached attention distribution.
19. Explain why past query rows are not required to compute the newest token's output once past K/V are cached.

## Part E — Repeated work and complexity

Assume three decode steps after the common three-token prefill, producing prefix lengths `4`, `5`, and `6`.

20. Count token rows processed through Q/K/V projections:
    - no cache: `4 + 5 + 6`
    - cache: `1 + 1 + 1`
21. Compute the number of repeated old-token projection rows avoided.
22. Count attention score interactions for the same three calls:
    - full-prefix no cache: `4^2 + 5^2 + 6^2`
    - cached newest-token attention: `4 + 5 + 6`
23. Holding model width fixed, state the per-call sequence-length complexity:
    - full-prefix no cache: `O(t^2)` attention interactions
    - cached decode: `O(t)` attention interactions
24. Explain why cached decoding is still not `O(1)` per token.
25. Derive the cache element count for this one-layer, one-head, batch-size-1 toy setup:
    ```math
    Ld_k + Ld_v = L(d_k+d_v).
    ```
26. Explain the time-memory trade-off introduced by the cache.

## Final conceptual question

Why can KV caching make autoregressive generation cheaper without changing the attention output for the newest token?

## Bonus

If the cached context length doubles, what happens to the number of stored K/V scalar elements in this toy setup?
