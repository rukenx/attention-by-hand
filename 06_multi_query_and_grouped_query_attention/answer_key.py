import math

import torch


def head_mapping(h_q: int, h_kv: int) -> list[int]:
    assert h_q % h_kv == 0
    group_size = h_q // h_kv
    return [h // group_size for h in range(h_q)]


def main() -> None:
    torch.set_printoptions(precision=4, sci_mode=False)

    H_q = 4
    H_kv_MHA = 4
    H_kv_GQA = 2
    H_kv_MQA = 1
    d_model = 8
    d_k = d_v = D = 2
    scale = math.sqrt(d_k)

    print("=" * 70)
    print("PART A - HEAD MAPPINGS")
    print("=" * 70)

    map_MHA = head_mapping(H_q, H_kv_MHA)
    map_GQA = head_mapping(H_q, H_kv_GQA)
    map_MQA = head_mapping(H_q, H_kv_MQA)

    assert map_MHA == [0, 1, 2, 3]
    assert map_GQA == [0, 0, 1, 1]
    assert map_MQA == [0, 0, 0, 0]

    print("MHA:", map_MHA)
    print("GQA:", map_GQA)
    print("MQA:", map_MQA)

    print("\n" + "=" * 70)
    print("PART B - SHARED K/V DOES NOT FORCE IDENTICAL ATTENTION")
    print("=" * 70)

    q_0 = torch.tensor([[1.0, 0.0]])
    q_1 = torch.tensor([[0.0, 1.0]])
    K_shared = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    V_shared = torch.tensor([
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    scores_0 = q_0 @ K_shared.T / scale
    scores_1 = q_1 @ K_shared.T / scale
    A_0 = torch.softmax(scores_0, dim=-1)
    A_1 = torch.softmax(scores_1, dim=-1)
    O_0 = A_0 @ V_shared
    O_1 = A_1 @ V_shared

    expected_A_0 = torch.tensor([[0.66976155, 0.33023845]])
    expected_A_1 = torch.tensor([[0.33023845, 0.66976155]])
    expected_O_0 = torch.tensor([[1.0, 0.33023845]])
    expected_O_1 = torch.tensor([[1.0, 0.66976155]])

    torch.testing.assert_close(A_0, expected_A_0)
    torch.testing.assert_close(A_1, expected_A_1)
    torch.testing.assert_close(O_0, expected_O_0)
    torch.testing.assert_close(O_1, expected_O_1)
    assert not torch.allclose(A_0, A_1)
    assert not torch.allclose(O_0, O_1)

    print("A_0:\n", A_0)
    print("A_1:\n", A_1)
    print("O_0:\n", O_0)
    print("O_1:\n", O_1)

    print("\n" + "=" * 70)
    print("PART C - PROJECTION WIDTHS AND K/V PARAMETER COUNTS")
    print("=" * 70)

    def widths(h_kv: int) -> tuple[int, int, int]:
        return H_q * d_k, h_kv * d_k, h_kv * d_v

    widths_MHA = widths(H_kv_MHA)
    widths_GQA = widths(H_kv_GQA)
    widths_MQA = widths(H_kv_MQA)

    assert widths_MHA == (8, 8, 8)
    assert widths_GQA == (8, 4, 4)
    assert widths_MQA == (8, 2, 2)

    def kv_parameter_count(h_kv: int) -> int:
        return d_model * h_kv * (d_k + d_v)

    kv_params_MHA = kv_parameter_count(H_kv_MHA)
    kv_params_GQA = kv_parameter_count(H_kv_GQA)
    kv_params_MQA = kv_parameter_count(H_kv_MQA)

    assert (kv_params_MHA, kv_params_GQA, kv_params_MQA) == (128, 64, 32)

    print("widths (Q, K, V):")
    print("  MHA:", widths_MHA)
    print("  GQA:", widths_GQA)
    print("  MQA:", widths_MQA)
    print("combined K/V weight parameters:", kv_params_MHA, kv_params_GQA, kv_params_MQA)

    print("\n" + "=" * 70)
    print("PART D - KV CACHE SIZE")
    print("=" * 70)

    B = 1
    N_layers = 1
    L = 3

    def cache_elements(h_kv: int) -> int:
        return B * N_layers * L * h_kv * (d_k + d_v)

    cache_MHA = cache_elements(H_kv_MHA)
    cache_GQA = cache_elements(H_kv_GQA)
    cache_MQA = cache_elements(H_kv_MQA)

    assert (cache_MHA, cache_GQA, cache_MQA) == (48, 24, 12)
    print("cache elements:", cache_MHA, cache_GQA, cache_MQA)

    print("\n" + "=" * 70)
    print("PART E - GQA ATTENTION-HEAD TENSOR FORWARD")
    print("=" * 70)

    # Practical layout:
    # Q: [B, H_q,  L, D]
    # K: [B, H_kv, L, D]
    # V: [B, H_kv, L, D]
    Q = torch.tensor([[[
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], [
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, -1.0],
    ], [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], [
        [1.0, -1.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ]]])

    K = torch.tensor([[[
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]]])

    V = torch.tensor([[[
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], [
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]]])

    assert Q.shape == (B, H_q, L, D)
    assert K.shape == (B, H_kv_GQA, L, D)
    assert V.shape == (B, H_kv_GQA, L, D)

    head_to_kv = torch.tensor(map_GQA, dtype=torch.long)

    # For clarity, this verifier MATERIALIZES query-head-shaped copies with
    # index_select. That is only a convenient teaching implementation. The physical
    # K/V state above still contains H_kv=2 heads, and optimized GQA kernels do not
    # need to duplicate the KV cache into H_q=4 stored heads.
    K_for_q = K.index_select(dim=1, index=head_to_kv)
    V_for_q = V.index_select(dim=1, index=head_to_kv)

    assert K_for_q.shape == (B, H_q, L, D)
    assert V_for_q.shape == (B, H_q, L, D)

    torch.testing.assert_close(K_for_q[:, 0], K_for_q[:, 1])
    torch.testing.assert_close(V_for_q[:, 0], V_for_q[:, 1])
    torch.testing.assert_close(K_for_q[:, 2], K_for_q[:, 3])
    torch.testing.assert_close(V_for_q[:, 2], V_for_q[:, 3])

    gqa_scores = torch.matmul(Q, K_for_q.transpose(-1, -2)) / scale
    gqa_attention = torch.softmax(gqa_scores, dim=-1)
    gqa_output = torch.matmul(gqa_attention, V_for_q)

    assert gqa_scores.shape == (B, H_q, L, L)
    assert gqa_attention.shape == (B, H_q, L, L)
    assert gqa_output.shape == (B, H_q, L, D)
    torch.testing.assert_close(
        gqa_attention.sum(dim=-1),
        torch.ones(B, H_q, L),
    )

    # Sharing K/V does not make heads identical because Q remains head-specific.
    assert not torch.allclose(gqa_output[:, 0], gqa_output[:, 1])
    assert not torch.allclose(gqa_output[:, 2], gqa_output[:, 3])

    # Verify the vectorized teaching implementation against the canonical per-head
    # formula without materializing a repeated cache.
    outputs_by_head = []
    for h in range(H_q):
        kv_h = map_GQA[h]
        scores_h = Q[:, h] @ K[:, kv_h].transpose(-1, -2) / scale
        attention_h = torch.softmax(scores_h, dim=-1)
        output_h = attention_h @ V[:, kv_h]
        outputs_by_head.append(output_h)

    gqa_output_loop = torch.stack(outputs_by_head, dim=1)
    torch.testing.assert_close(gqa_output, gqa_output_loop)

    print("Q shape:", tuple(Q.shape))
    print("physical K shape:", tuple(K.shape))
    print("physical V shape:", tuple(V.shape))
    print("head -> KV mapping:", map_GQA)
    print("materialized K_for_q shape:", tuple(K_for_q.shape))
    print("attention shape:", tuple(gqa_attention.shape))
    print("output shape:", tuple(gqa_output.shape))

    print("\n" + "=" * 70)
    print("PART F - WHAT GETS CHEAPER")
    print("=" * 70)
    print("Reducing H_kv reduces K/V projection width, K/V parameters, KV-cache size,")
    print("and K/V memory traffic. H_q stays fixed, so query-head attention work")
    print("does not simply shrink in direct proportion to H_kv.")

    assert torch.isfinite(gqa_output).all()
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
