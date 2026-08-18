import math
from pathlib import Path

import matplotlib.pyplot as plt


FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def unit_circle_points(steps: int = 360) -> tuple[list[float], list[float]]:
    angles = [2.0 * math.pi * i / steps for i in range(steps + 1)]
    return [math.cos(a) for a in angles], [math.sin(a) for a in angles]


def draw_vector(ax, angle: float, label: str) -> None:
    x = math.cos(angle)
    y = math.sin(angle)
    ax.arrow(0.0, 0.0, x, y, width=0.012, length_includes_head=True)
    ax.text(1.08 * x, 1.08 * y, label, ha="center", va="center")


def configure_axes(ax) -> None:
    circle_x, circle_y = unit_circle_points()
    ax.plot(circle_x, circle_y)
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def generate_unit_circle() -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    configure_axes(ax)

    for position in (0, 1, 2):
        draw_vector(ax, float(position), f"p={position}")

    ax.set_title("RoPE rotations at positions 0, 1, and 2")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rope_unit_circle.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_relative_angle() -> None:
    m = 1
    n = 2

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    configure_axes(ax)
    draw_vector(ax, float(m), "query: m=1")
    draw_vector(ax, float(n), "key: n=2")

    midpoint = 0.5 * (m + n)
    ax.text(
        0.62 * math.cos(midpoint),
        0.62 * math.sin(midpoint),
        r"$(n-m)\theta = 1$ rad",
        ha="center",
        va="center",
    )
    ax.set_title("Absolute rotations expose a relative angle")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rope_relative_angle.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_unit_circle()
    generate_relative_angle()
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
