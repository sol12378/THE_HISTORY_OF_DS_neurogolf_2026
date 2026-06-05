from __future__ import annotations

import matplotlib.pyplot as plt


def plot_grid(grid: list[list[int]]) -> None:
    plt.imshow(grid, interpolation="nearest")
    plt.axis("off")
