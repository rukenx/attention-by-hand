# Exercise 06: Multi-Query and Grouped-Query Attention

Study how multiple query heads can share fewer key/value heads.

The structural continuum is:

- **MHA:** `H_kv = H_q`
- **GQA:** `1 < H_kv < H_q`
- **MQA:** `H_kv = 1`

Reducing `H_kv` reduces K/V projection width, KV-cache storage, and K/V memory traffic. It does **not** reduce the number of query heads, so query-head score/value work does not simply shrink in proportion to `H_kv`.

The tensor verifier uses a GQA example with `H_q = 4` and `H_kv = 2`, mapping query heads as `[0, 0, 1, 1]`. For clarity it materializes query-head-shaped K/V tensors with `index_select`; optimized implementations do not need to duplicate the physical KV cache this way.

## Files

- `problem.md` — definitions, projection shapes, cache formulas, and tasks
- `solution_by_hand.pdf` — worked derivation
- `solution_by_hand.tex` — LaTeX source
- `answer_key.py` — deterministic PyTorch verification, including a full GQA attention-head tensor example

Run from the repository root:

```bash
python 06_multi_query_and_grouped_query_attention/answer_key.py
```
