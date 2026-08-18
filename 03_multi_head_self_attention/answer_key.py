import torch


def attention_head(
    X: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    sqrt_dk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    scores = Q @ K.T / sqrt_dk
    attention = torch.softmax(scores, dim=-1)
    output = attention @ V
    return Q, K, V, attention, output


def main() -> None:
    torch.set_printoptions(precision=4, sci_mode=False)

    X = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ])

    d_k = 2
    sqrt_dk = torch.sqrt(torch.tensor(float(d_k)))

    W_Q_1 = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    W_K_1 = W_Q_1.clone()
    W_V_1 = W_Q_1.clone()

    W_Q_2 = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ])
    W_K_2 = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ])
    W_V_2 = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    Q_1, K_1, V_1, A_1, O_1 = attention_head(
        X, W_Q_1, W_K_1, W_V_1, sqrt_dk
    )
    Q_2, K_2, V_2, A_2, O_2 = attention_head(
        X, W_Q_2, W_K_2, W_V_2, sqrt_dk
    )

    expected_Q_1 = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    expected_A_1 = torch.tensor([
        [0.4011, 0.1978, 0.4011],
        [0.1978, 0.4011, 0.4011],
        [0.2483, 0.2483, 0.5035],
    ])
    expected_O_1 = torch.tensor([
        [0.8022, 0.5989],
        [0.5989, 0.8022],
        [0.7517, 0.7517],
    ])

    expected_Q_2 = torch.tensor([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    expected_K_2 = torch.tensor([
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ])
    expected_V_2 = torch.tensor([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    expected_A_2 = torch.tensor([
        [0.1978, 0.4011, 0.4011],
        [0.2483, 0.5035, 0.2483],
        [0.4011, 0.4011, 0.1978],
    ])
    expected_O_2 = torch.tensor([
        [0.5989, 0.8022],
        [0.7517, 0.7517],
        [0.8022, 0.5989],
    ])

    torch.testing.assert_close(Q_1, expected_Q_1)
    torch.testing.assert_close(K_1, expected_Q_1)
    torch.testing.assert_close(V_1, expected_Q_1)
    torch.testing.assert_close(A_1, expected_A_1, atol=1e-4, rtol=0)
    torch.testing.assert_close(O_1, expected_O_1, atol=1e-4, rtol=0)

    torch.testing.assert_close(Q_2, expected_Q_2)
    torch.testing.assert_close(K_2, expected_K_2)
    torch.testing.assert_close(V_2, expected_V_2)
    torch.testing.assert_close(A_2, expected_A_2, atol=1e-4, rtol=0)
    torch.testing.assert_close(O_2, expected_O_2, atol=1e-4, rtol=0)

    torch.testing.assert_close(A_1.sum(dim=-1), torch.ones(3))
    torch.testing.assert_close(A_2.sum(dim=-1), torch.ones(3))
    assert not torch.allclose(A_1, A_2)

    O_concat = torch.cat([O_1, O_2], dim=-1)
    W_O = torch.eye(4)
    O_mha = O_concat @ W_O

    expected_concat = torch.tensor([
        [0.8022, 0.5989, 0.5989, 0.8022],
        [0.5989, 0.8022, 0.7517, 0.7517],
        [0.7517, 0.7517, 0.8022, 0.5989],
    ])

    torch.testing.assert_close(O_concat, expected_concat, atol=1e-4, rtol=0)
    torch.testing.assert_close(O_mha, O_concat)

    assert Q_1.shape == K_1.shape == V_1.shape == (3, 2)
    assert Q_2.shape == K_2.shape == V_2.shape == (3, 2)
    assert A_1.shape == A_2.shape == (3, 3)
    assert O_concat.shape == O_mha.shape == (3, 4)
    assert torch.isfinite(O_mha).all()

    print("Head 1 attention:\n", A_1)
    print("Head 2 attention:\n", A_2)
    print("Concatenated output:\n", O_concat)
    print("Final MHA output:\n", O_mha)
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
