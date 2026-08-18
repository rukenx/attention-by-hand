import math

import torch


def rotate_rows(vectors: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply standard 2D rotations to row vectors, one angle per row."""
    x = vectors[:, 0]
    y = vectors[:, 1]
    c = torch.cos(angles)
    s = torch.sin(angles)
    return torch.stack([x * c - y * s, x * s + y * c], dim=-1)


def main() -> None:
    torch.set_printoptions(precision=4, sci_mode=False)

    d_k = 2
    sqrt_dk = math.sqrt(d_k)
    theta = 1.0
    positions = torch.tensor([0.0, 1.0, 2.0])

    Q = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    K = Q.clone()

    print("----------------------------------------------------------------------")
    print("PART A - WITHOUT POSITION")
    print("----------------------------------------------------------------------")

    plain_scores = Q @ K.T
    plain_scaled = plain_scores / sqrt_dk
    plain_attention = torch.softmax(plain_scaled, dim=-1)

    expected_plain_scores = torch.ones(3, 3)
    expected_plain_attention = torch.full((3, 3), 1.0 / 3.0)

    torch.testing.assert_close(plain_scores, expected_plain_scores)
    torch.testing.assert_close(plain_attention, expected_plain_attention)

    print("plain scores:\n", plain_scores)
    print("plain attention:\n", plain_attention)

    print("\n----------------------------------------------------------------------")
    print("PART B - APPLY ROPE ROTATIONS")
    print("----------------------------------------------------------------------")

    angles = positions * theta
    Q_rope = rotate_rows(Q, angles)
    K_rope = rotate_rows(K, angles)

    expected_rotated = torch.tensor([
        [1.0, 0.0],
        [math.cos(1.0), math.sin(1.0)],
        [math.cos(2.0), math.sin(2.0)],
    ])

    torch.testing.assert_close(Q_rope, expected_rotated)
    torch.testing.assert_close(K_rope, expected_rotated)

    print("Q_rope:\n", Q_rope)
    print("K_rope:\n", K_rope)

    print("\n----------------------------------------------------------------------")
    print("PART C - RELATIVE-POSITION SCORE STRUCTURE")
    print("----------------------------------------------------------------------")

    rope_scores = Q_rope @ K_rope.T
    expected_rope_scores = torch.tensor([
        [1.0, math.cos(1.0), math.cos(2.0)],
        [math.cos(1.0), 1.0, math.cos(1.0)],
        [math.cos(2.0), math.cos(1.0), 1.0],
    ])

    torch.testing.assert_close(rope_scores, expected_rope_scores, atol=1e-6, rtol=0)

    # Equal relative offsets produce equal scores in this controlled setup.
    torch.testing.assert_close(rope_scores[0, 1], rope_scores[1, 2])
    torch.testing.assert_close(rope_scores[1, 0], rope_scores[2, 1])

    # The diagonal is 1 only because Q == K and every base vector is unit norm.
    torch.testing.assert_close(torch.diag(rope_scores), torch.ones(3))

    print("RoPE scores:\n", rope_scores)

    print("\n----------------------------------------------------------------------")
    print("PART D - SCALE AND SOFTMAX")
    print("----------------------------------------------------------------------")

    rope_scaled = rope_scores / sqrt_dk
    rope_attention = torch.softmax(rope_scaled, dim=-1)

    expected_attention = torch.tensor([
        [0.4785, 0.3457, 0.1758],
        [0.2955, 0.4090, 0.2955],
        [0.1758, 0.3457, 0.4785],
    ])

    torch.testing.assert_close(rope_attention, expected_attention, atol=1e-4, rtol=0)
    torch.testing.assert_close(
        rope_attention.sum(dim=-1),
        torch.ones(3),
    )

    print("scaled RoPE scores:\n", rope_scaled)
    print("RoPE attention:\n", rope_attention)

    print("\n----------------------------------------------------------------------")
    print("PART E - NORM PRESERVATION")
    print("----------------------------------------------------------------------")

    torch.testing.assert_close(
        torch.linalg.vector_norm(Q_rope, dim=-1),
        torch.linalg.vector_norm(Q, dim=-1),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(K_rope, dim=-1),
        torch.linalg.vector_norm(K, dim=-1),
    )

    assert torch.isfinite(rope_attention).all()

    print("all rotated Q/K rows preserve unit norm")
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
