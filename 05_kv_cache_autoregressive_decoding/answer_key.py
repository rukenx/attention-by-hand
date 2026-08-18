import math

import torch


def main() -> None:
    torch.set_printoptions(precision=4, sci_mode=False)

    d_k = 2
    d_v = 2
    scale = math.sqrt(d_k)

    W_Q = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    W_K = torch.tensor([
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    W_V = torch.tensor([
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    x_I = torch.tensor([1.0, 0.0])
    x_like = torch.tensor([0.0, 1.0])
    x_cats = torch.tensor([1.0, 1.0])
    x_today = torch.tensor([0.0, 1.0])

    print("=" * 70)
    print("PART A - PREFILL")
    print("=" * 70)

    X_past = torch.stack([x_I, x_like, x_cats])
    Q_past = X_past @ W_Q
    K_past = X_past @ W_K
    V_past = X_past @ W_V

    assert Q_past.shape == K_past.shape == V_past.shape == (3, 2)

    K_cache_prefill = K_past.clone()
    V_cache_prefill = V_past.clone()

    print("Q_past:\n", Q_past)
    print("K_past:\n", K_past)
    print("V_past:\n", V_past)

    print("\n" + "=" * 70)
    print("PART B - NO-CACHE FULL-PREFIX CAUSAL DECODE")
    print("=" * 70)

    X_full = torch.stack([x_I, x_like, x_cats, x_today])
    Q_full = X_full @ W_Q
    K_full = X_full @ W_K
    V_full = X_full @ W_V

    # Old K/V rows are exactly the same values, but a no-cache full-prefix
    # implementation recomputes them because it reprocesses the entire prefix.
    torch.testing.assert_close(K_full[:3], K_past)
    torch.testing.assert_close(V_full[:3], V_past)

    scores_full = Q_full @ K_full.T / scale
    causal_mask = torch.triu(
        torch.full_like(scores_full, float("-inf")),
        diagonal=1,
    )
    masked_scores_full = scores_full + causal_mask
    A_full = torch.softmax(masked_scores_full, dim=-1)
    O_full = A_full @ V_full

    o_t_no_cache = O_full[-1:]

    assert scores_full.shape == (4, 4)
    assert A_full.shape == (4, 4)
    assert O_full.shape == (4, 2)
    torch.testing.assert_close(A_full.sum(dim=-1), torch.ones(4))
    assert torch.count_nonzero(torch.triu(A_full, diagonal=1)) == 0

    print("full scaled scores:\n", scores_full)
    print("causal mask:\n", causal_mask)
    print("full causal attention:\n", A_full)
    print("full outputs:\n", O_full)
    print("newest-token no-cache output:\n", o_t_no_cache)

    print("\n" + "=" * 70)
    print("PART C - CACHED DECODE")
    print("=" * 70)

    K_cache = K_cache_prefill.clone()
    V_cache = V_cache_prefill.clone()

    q_t = x_today.unsqueeze(0) @ W_Q
    k_t = x_today.unsqueeze(0) @ W_K
    v_t = x_today.unsqueeze(0) @ W_V

    K_cache = torch.cat([K_cache, k_t], dim=0)
    V_cache = torch.cat([V_cache, v_t], dim=0)

    s_t_cache = q_t @ K_cache.T / scale
    a_t_cache = torch.softmax(s_t_cache, dim=-1)
    o_t_cache = a_t_cache @ V_cache

    torch.testing.assert_close(q_t, Q_full[-1:])
    torch.testing.assert_close(k_t, K_full[-1:])
    torch.testing.assert_close(v_t, V_full[-1:])
    torch.testing.assert_close(K_cache, K_full)
    torch.testing.assert_close(V_cache, V_full)

    # The last row of a causal full-prefix attention call has no future positions,
    # so it must match the single cached query against all available keys.
    torch.testing.assert_close(s_t_cache, scores_full[-1:])
    torch.testing.assert_close(a_t_cache, A_full[-1:])
    torch.testing.assert_close(o_t_cache, o_t_no_cache)

    expected_attention = torch.tensor([
        [0.19888169, 0.19888169, 0.40335493, 0.19888169]
    ])
    expected_output = torch.tensor([
        [1.40335493, 0.80111831]
    ])

    torch.testing.assert_close(a_t_cache, expected_attention)
    torch.testing.assert_close(o_t_cache, expected_output)

    print("cached attention:\n", a_t_cache)
    print("cached output:\n", o_t_cache)
    print("same newest-token output: yes")

    print("\n" + "=" * 70)
    print("PART D - REPEATED WORK")
    print("=" * 70)

    prefix_lengths = [4, 5, 6]

    no_cache_projection_rows = sum(prefix_lengths)
    cache_projection_rows = len(prefix_lengths)
    repeated_projection_rows_avoided = (
        no_cache_projection_rows - cache_projection_rows
    )

    no_cache_attention_interactions = sum(t * t for t in prefix_lengths)
    cache_attention_interactions = sum(prefix_lengths)

    assert no_cache_projection_rows == 15
    assert cache_projection_rows == 3
    assert repeated_projection_rows_avoided == 12
    assert no_cache_attention_interactions == 77
    assert cache_attention_interactions == 15

    print(f"projection rows, no cache: {no_cache_projection_rows}")
    print(f"projection rows, cache: {cache_projection_rows}")
    print(f"repeated projection rows avoided: {repeated_projection_rows_avoided}")
    print(f"attention interactions, no cache: {no_cache_attention_interactions}")
    print(f"attention interactions, cache: {cache_attention_interactions}")

    print("\n" + "=" * 70)
    print("PART E - CACHE SIZE AND COMPLEXITY")
    print("=" * 70)

    L = K_cache.shape[0]
    cache_elements = K_cache.numel() + V_cache.numel()
    expected_cache_elements = L * (d_k + d_v)

    assert cache_elements == expected_cache_elements == 4 * L

    print(f"cache elements at L={L}: {cache_elements}")
    print("per decode call, holding model width fixed:")
    print("  full-prefix no cache attention interactions: O(t^2)")
    print("  cached newest-token attention interactions: O(t)")
    print("cached decoding is not O(1): the newest query still scans t keys/values")

    assert torch.isfinite(o_t_cache).all()
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
