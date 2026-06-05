from __future__ import annotations


def describe_grid(grid: list[list[int]]) -> dict[str, int]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    return {"height": height, "width": width}
