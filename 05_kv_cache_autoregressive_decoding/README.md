# Exercise 05: KV Cache for Autoregressive Decoding

Compare a full-prefix causal attention recomputation with a cached single-token decode step.

Both procedures must produce the same output for the newest token. The difference is repeated work:

- **No cache:** recompute projections for the full prefix and evaluate full-prefix causal attention.
- **KV cache:** project only the new token, append its key/value state, and evaluate only the new query against cached keys.

This makes the complexity comparison concrete: holding model width fixed, full-prefix attention has `O(t^2)` token-to-token interactions for a length-`t` decode call, while cached current-token attention has `O(t)` interactions.

## Files

- `problem.md` — numerical setup and tasks
- `solution_by_hand.pdf` — worked derivation
- `solution_by_hand.tex` — LaTeX source
- `answer_key.py` — deterministic PyTorch verification

Run from the repository root:

```bash
python 05_kv_cache_autoregressive_decoding/answer_key.py
```
