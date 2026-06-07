from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp073_task251_closed_component_shape_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 251


@dataclass(frozen=True)
class ComponentRow:
    example_idx: int
    component_idx: int
    is_positive_component: bool
    size: int
    bbox_h: int
    bbox_w: int
    bbox_area: int
    is_rectangle: bool
    touches_grid_border: bool
    boundary_colors: str
    ray4_all_cells: bool
    adj_any_all_cells: bool
    holes_in_bbox: int


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows) + 1, max(cols) + 1


def zero_components(x: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = x == 0
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = x.shape
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def boundary_colors(x: np.ndarray, cells: list[tuple[int, int]]) -> set[int]:
    colors: set[int] = set()
    h, w = x.shape
    cell_set = set(cells)
    for r, c in cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and x[nr, nc] != 0:
                colors.add(int(x[nr, nc]))
    return colors


def cell_ray4(x: np.ndarray, r: int, c: int) -> bool:
    h, w = x.shape
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        cr, cc = r + dr, c + dc
        found = False
        while 0 <= cr < h and 0 <= cc < w:
            if x[cr, cc] != 0:
                found = True
                break
            cr += dr
            cc += dc
        if not found:
            return False
    return True


def cell_adj_any(x: np.ndarray, r: int, c: int) -> bool:
    h, w = x.shape
    return any(0 <= r + dr < h and 0 <= c + dc < w and x[r + dr, c + dc] != 0 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])


def audit() -> list[ComponentRow]:
    rows: list[ComponentRow] = []
    examples = examples_for(load_task(TASK_ID), -1)
    for ex_idx, ex in enumerate(examples):
        x = arr(ex["input"])
        y = arr(ex["output"])
        diff = x != y
        for comp_idx, cells in enumerate(zero_components(x)):
            r0, c0, r1, c1 = bbox(cells)
            cell_set = set(cells)
            component_diff = any(bool(diff[r, c]) for r, c in cells)
            touches = any(r == 0 or c == 0 or r == x.shape[0] - 1 or c == x.shape[1] - 1 for r, c in cells)
            area = (r1 - r0) * (c1 - c0)
            rect = area == len(cells)
            hole_count = area - len(cells)
            rows.append(
                ComponentRow(
                    example_idx=ex_idx,
                    component_idx=comp_idx,
                    is_positive_component=component_diff,
                    size=len(cells),
                    bbox_h=r1 - r0,
                    bbox_w=c1 - c0,
                    bbox_area=area,
                    is_rectangle=rect,
                    touches_grid_border=touches,
                    boundary_colors=json.dumps(sorted(boundary_colors(x, cells)), ensure_ascii=False),
                    ray4_all_cells=all(cell_ray4(x, r, c) for r, c in cell_set),
                    adj_any_all_cells=all(cell_adj_any(x, r, c) for r, c in cell_set),
                    holes_in_bbox=hole_count,
                )
            )
    return rows


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    rows = audit()
    with (EXP_DIR / "component_shape_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ComponentRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])

    pos = [row for row in rows if row.is_positive_component]
    neg = [row for row in rows if not row.is_positive_component]
    summary = {
        "positive_components": len(pos),
        "negative_components": len(neg),
        "positive_rectangle": sum(row.is_rectangle for row in pos),
        "negative_rectangle": sum(row.is_rectangle for row in neg),
        "positive_bbox_shape_hist": Counter(f"{row.bbox_h}x{row.bbox_w}" for row in pos).most_common(),
        "negative_bbox_shape_top": Counter(f"{row.bbox_h}x{row.bbox_w}" for row in neg).most_common(12),
        "positive_size_hist": Counter(row.size for row in pos).most_common(),
        "positive_boundary_colors": Counter(row.boundary_colors for row in pos).most_common(),
        "negative_boundary_colors_top": Counter(row.boundary_colors for row in neg).most_common(12),
        "positive_ray4_all": sum(row.ray4_all_cells for row in pos),
        "positive_adj_any_all": sum(row.adj_any_all_cells for row in pos),
        "negative_boundary2_closed": sum((not row.touches_grid_border) and row.boundary_colors == "[2]" for row in neg),
    }
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "audit_ready",
        "task_id": TASK_ID,
        "hypothesis": "task251 closed zero components may be rectangle-like enough for low-cost closed-form mask lowering.",
        "summary": summary,
        "decision": "Use this audit to decide whether rectangle/bbox mask lowering is plausible before emitting ONNX.",
        "leakage_risk": "low: structure audit only; no submission artifact.",
        "overfitting_risk": "medium: component shape regularities may overfit arc-gen; any lowering still needs LB calibration.",
        "outputs": {"component_shape_audit": "component_shape_audit.csv"},
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`task251` のclosed zero-component ruleを低cost化できるか、正例componentの形状を監査する。

## 結果

- positive components: {summary["positive_components"]}
- negative components: {summary["negative_components"]}
- positive rectangles: {summary["positive_rectangle"]}/{summary["positive_components"]}
- positive bbox shapes: {summary["positive_bbox_shape_hist"][:10]}
- positive boundary colors: {summary["positive_boundary_colors"]}

## 判断

矩形性やbbox shapeの偏りが強ければ、flood-fill unrollではなくrectangle/bbox mask loweringへ進む。弱ければcomponent connectivityを別の安い方法で表現する必要がある。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
