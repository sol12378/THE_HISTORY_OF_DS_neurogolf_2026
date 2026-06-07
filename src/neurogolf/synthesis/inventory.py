from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from neurogolf.synthesis.dsl import infer_color_map


@dataclass(frozen=True)
class ExampleSummary:
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    input_colors: tuple[int, ...]
    output_colors: tuple[int, ...]
    changed_cells: int | None
    changed_ratio: float | None
    input_nonzero_ratio: float
    output_nonzero_ratio: float
    line_score: int
    component_count: int


def grid(example_grid: list[list[int]]) -> np.ndarray:
    return np.asarray(example_grid, dtype=np.int64)


def colors(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in np.unique(arr)))


def nonzero_ratio(arr: np.ndarray) -> float:
    return float(np.count_nonzero(arr) / max(1, arr.size))


def line_score(arr: np.ndarray) -> int:
    score = 0
    for value in colors(arr):
        if value == 0:
            continue
        mask = arr == value
        score += int(np.sum(np.all(mask, axis=1)))
        score += int(np.sum(np.all(mask, axis=0)))
    return score


def connected_components(arr: np.ndarray, background: int = 0) -> int:
    mask = arr != background
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return count


def summarize_example(example: dict[str, Any]) -> ExampleSummary:
    x = grid(example["input"])
    y = grid(example["output"])
    changed_cells: int | None = None
    changed_ratio: float | None = None
    if x.shape == y.shape:
        changed_cells = int(np.sum(x != y))
        changed_ratio = float(changed_cells / max(1, x.size))
    return ExampleSummary(
        input_shape=tuple(int(v) for v in x.shape),
        output_shape=tuple(int(v) for v in y.shape),
        input_colors=colors(x),
        output_colors=colors(y),
        changed_cells=changed_cells,
        changed_ratio=changed_ratio,
        input_nonzero_ratio=nonzero_ratio(x),
        output_nonzero_ratio=nonzero_ratio(y),
        line_score=line_score(x),
        component_count=connected_components(x),
    )


def train_pairs(task: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(grid(example["input"]), grid(example["output"])) for example in task.get("train", [])]


def has_consistent_color_map(task: dict[str, Any]) -> bool:
    mapping = infer_color_map(train_pairs(task))
    if mapping is None:
        return False
    return any(src != dst for src, dst in mapping.items())


def family_vote(task: dict[str, Any], route: str, source: str, template_name: str) -> str:
    summaries = [summarize_example(example) for example in task.get("train", [])]
    if not summaries:
        return "unknown_program_synthesis"

    same_shape = all(item.input_shape == item.output_shape for item in summaries)
    shape_changes = [item for item in summaries if item.input_shape != item.output_shape]
    avg_changed = _avg(item.changed_ratio for item in summaries)
    avg_line_score = _avg(item.line_score for item in summaries)
    avg_components = _avg(item.component_count for item in summaries)
    avg_input_nonzero = _avg(item.input_nonzero_ratio for item in summaries)
    avg_output_nonzero = _avg(item.output_nonzero_ratio for item in summaries)

    source_text = f"{source} {template_name}".lower()
    if "signature" in source_text:
        return "signature_lookup_current"
    if shape_changes:
        return "crop_resize"
    if same_shape and avg_line_score >= 2 and avg_changed >= 0.10:
        return "region_partition_fill"
    if same_shape and avg_line_score >= 1:
        return "line_grid_fill"
    if same_shape and has_consistent_color_map(task):
        return "color_map"
    if same_shape and avg_changed <= 0.20:
        return "sparse_edit_or_object_completion"
    if avg_components <= 8 and avg_input_nonzero < avg_output_nonzero:
        return "point_to_line_pattern"
    if route:
        return route
    return "unknown_program_synthesis"


def _avg(values: Any) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def task_feature_row(
    task_id: int,
    task: dict[str, Any],
    route: str,
    source: str,
    template_name: str,
    cost: int,
    points: float,
) -> dict[str, object]:
    summaries = [summarize_example(example) for example in task.get("train", [])]
    train_count = len(task.get("train", []))
    test_count = len(task.get("test", []))
    arc_gen_count = len(task.get("arc-gen", []))
    shape_pairs = Counter(f"{s.input_shape}->{s.output_shape}" for s in summaries)
    in_colors = sorted(set().union(*(set(s.input_colors) for s in summaries)) if summaries else set())
    out_colors = sorted(set().union(*(set(s.output_colors) for s in summaries)) if summaries else set())
    family = family_vote(task, route, source, template_name)
    return {
        "task_id": task_id,
        "synthesis_family": family,
        "route_prediction": route,
        "current_source": source,
        "current_template": template_name,
        "current_cost": cost,
        "current_points": f"{points:.6f}",
        "train_count": train_count,
        "test_count": test_count,
        "arc_gen_count": arc_gen_count,
        "shape_pairs": ";".join(f"{k}:{v}" for k, v in sorted(shape_pairs.items())),
        "input_colors": " ".join(str(x) for x in in_colors),
        "output_colors": " ".join(str(x) for x in out_colors),
        "avg_changed_ratio": f"{_avg(s.changed_ratio for s in summaries):.6f}",
        "avg_input_nonzero_ratio": f"{_avg(s.input_nonzero_ratio for s in summaries):.6f}",
        "avg_output_nonzero_ratio": f"{_avg(s.output_nonzero_ratio for s in summaries):.6f}",
        "avg_line_score": f"{_avg(s.line_score for s in summaries):.3f}",
        "avg_component_count": f"{_avg(s.component_count for s in summaries):.3f}",
        "needs_program_synthesis": int(family not in {"color_map", "signature_lookup_current"}),
    }
