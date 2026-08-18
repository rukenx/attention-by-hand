import torch


def print_matrix(name: str, value: torch.Tensor) -> None:
    print(f"=== {name} ===")
    print(value)
    print("shape:", tuple(value.shape))
    print()


def main() -> None:
    torch.set_printoptions(precision=4, sci_mode=False)

    X = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
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

    d_k = 2
    sqrt_dk = torch.sqrt(torch.tensor(float(d_k)))

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    K_T = K.T
    scores = Q @ K_T
    scaled_scores = scores / sqrt_dk

    causal_mask = torch.tensor([
        [0.0, float("-inf"), float("-inf")],
        [0.0, 0.0,           float("-inf")],
        [0.0, 0.0,           0.0],
    ])

    masked_scores = scaled_scores + causal_mask
    causal_attention_weights = torch.softmax(masked_scores, dim=-1)
    causal_output = causal_attention_weights @ V

    print_matrix("Input embeddings X", X)
    print_matrix("Queries Q", Q)
    print_matrix("Keys K", K)
    print_matrix("Values V", V)
    print_matrix("Transposed keys K^T", K_T)
    print_matrix("Raw attention scores QK^T", scores)
    print_matrix("Scaled attention scores", scaled_scores)
    print_matrix("Causal mask M", causal_mask)
    print_matrix("Masked attention scores", masked_scores)
    print_matrix("Causal attention weights", causal_attention_weights)
    print("row sums:", causal_attention_weights.sum(dim=-1))
    print()
    print_matrix("Causal output", causal_output)

    expected_Q = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    expected_K = torch.tensor([
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 2.0],
    ])
    expected_V = torch.tensor([
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ])
    expected_K_T = torch.tensor([
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 2.0],
    ])
    expected_scores = torch.tensor([
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 2.0],
        [2.0, 1.0, 3.0],
    ])
    expected_scaled_scores = torch.tensor([
        [0.7071, 0.0000, 0.7071],
        [0.7071, 0.7071, 1.4142],
        [1.4142, 0.7071, 2.1213],
    ])
    expected_masked_scores = torch.tensor([
        [0.7071, float("-inf"), float("-inf")],
        [0.7071, 0.7071,        float("-inf")],
        [1.4142, 0.7071,        2.1213],
    ])
    expected_causal_attention = torch.tensor([
        [1.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000],
        [0.2840, 0.1400, 0.5760],
    ])
    expected_causal_output = torch.tensor([
        [1.0000, 0.0000],
        [1.0000, 0.5000],
        [1.5760, 0.7160],
    ])

    torch.testing.assert_close(Q, expected_Q, atol=1e-6, rtol=0)
    torch.testing.assert_close(K, expected_K, atol=1e-6, rtol=0)
    torch.testing.assert_close(V, expected_V, atol=1e-6, rtol=0)
    torch.testing.assert_close(K_T, expected_K_T, atol=1e-6, rtol=0)
    torch.testing.assert_close(scores, expected_scores, atol=1e-6, rtol=0)
    torch.testing.assert_close(scaled_scores, expected_scaled_scores, atol=1e-4, rtol=0)
    torch.testing.assert_close(masked_scores, expected_masked_scores, atol=1e-4, rtol=0)
    torch.testing.assert_close(
        causal_attention_weights,
        expected_causal_attention,
        atol=1e-4,
        rtol=0,
    )
    torch.testing.assert_close(causal_output, expected_causal_output, atol=1e-4, rtol=0)

    assert causal_attention_weights.shape == (3, 3)
    assert causal_output.shape == (3, 2)
    assert torch.allclose(causal_attention_weights.sum(dim=-1), torch.ones(3), atol=1e-4)

    future_positions = torch.triu(
        torch.ones_like(causal_attention_weights, dtype=torch.bool),
        diagonal=1,
    )
    assert torch.allclose(
        causal_attention_weights[future_positions],
        torch.zeros_like(causal_attention_weights[future_positions]),
        atol=1e-6,
    )

    print("All assertions passed.")


if __name__ == "__main__":
    main()
