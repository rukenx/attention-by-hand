import torch


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
    attention_weights = torch.softmax(scaled_scores, dim=-1)
    output = attention_weights @ V

    print("=== Input embeddings X ===")
    print(X)
    print("shape:", tuple(X.shape))
    print()

    print("=== Queries Q ===")
    print(Q)
    print("shape:", tuple(Q.shape))
    print()

    print("=== Keys K ===")
    print(K)
    print("shape:", tuple(K.shape))
    print()

    print("=== Values V ===")
    print(V)
    print("shape:", tuple(V.shape))
    print()

    print("=== Transposed keys K^T ===")
    print(K_T)
    print("shape:", tuple(K_T.shape))
    print()

    print("=== Raw attention scores QK^T ===")
    print(scores)
    print("shape:", tuple(scores.shape))
    print()

    print("=== Scaled attention scores ===")
    print(scaled_scores)
    print("shape:", tuple(scaled_scores.shape))
    print()

    print("=== Attention weights ===")
    print(attention_weights)
    print("shape:", tuple(attention_weights.shape))
    print("row sums:", attention_weights.sum(dim=-1))
    print()

    print("=== Final output ===")
    print(output)
    print("shape:", tuple(output.shape))
    print()

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
    expected_scores = torch.tensor([
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 2.0],
        [2.0, 1.0, 3.0],
    ])
    expected_K_T = torch.tensor([
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 2.0],
    ])
    expected_scaled_scores = torch.tensor([
        [0.7071, 0.0000, 0.7071],
        [0.7071, 0.7071, 1.4142],
        [1.4142, 0.7071, 2.1213],
    ])
    expected_attention_weights = torch.tensor([
        [0.4011, 0.1978, 0.4011],
        [0.2483, 0.2483, 0.5035],
        [0.2840, 0.1400, 0.5760],
    ])
    expected_output = torch.tensor([
        [1.4011, 0.5989],
        [1.5035, 0.7517],
        [1.5760, 0.7160],
    ])

    torch.testing.assert_close(Q, expected_Q, atol=1e-6, rtol=0)
    torch.testing.assert_close(K, expected_K, atol=1e-6, rtol=0)
    torch.testing.assert_close(V, expected_V, atol=1e-6, rtol=0)
    torch.testing.assert_close(K_T, expected_K_T, atol=1e-6, rtol=0)
    torch.testing.assert_close(scores, expected_scores, atol=1e-6, rtol=0)
    torch.testing.assert_close(scaled_scores, expected_scaled_scores, atol=1e-4, rtol=0)
    torch.testing.assert_close(attention_weights, expected_attention_weights, atol=1e-4, rtol=0)
    torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=0)

    assert attention_weights.shape == (3, 3)
    assert output.shape == (3, 2)
    assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(3), atol=1e-4)

    print("All assertions passed.")


if __name__ == "__main__":
    main()
