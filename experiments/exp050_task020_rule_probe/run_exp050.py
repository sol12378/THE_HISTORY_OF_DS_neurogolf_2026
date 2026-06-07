from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task


EXP_ID = "exp050_task020_rule_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TARGET_TASK = 20


@dataclass
class ExampleSummary:
    example_idx: int
    split_guess: str
    input_shape: str
    output_shape: str
    input_colors: str
    output_colors: str
    changed_count: int
    added_count: int
    removed_count: int
    changed_bbox: str
    input_nonzero_bbox: str
    output_nonzero_bbox: str
    dominant_changed_from_to: str
    component_count_input: int
    component_count_output: int
    rule_hints: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def bbox_s(mask: np.ndarray) -> str:
    b = bbox(mask)
    return "" if b is None else f"{b[0]}:{b[2]},{b[1]}:{b[3]}"


def components(grid: np.ndarray, nonzero_only: bool = True) -> list[dict[str, Any]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, Any]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            if nonzero_only and color == 0:
                seen[r, c] = True
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = rr + dr, cc + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w or seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            rows = [x[0] for x in cells]
            cols = [x[1] for x in cells]
            comps.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": (min(rows), min(cols), max(rows) + 1, max(cols) + 1),
                    "height": max(rows) - min(rows) + 1,
                    "width": max(cols) - min(cols) + 1,
                }
            )
    return comps


def infer_hints(x: np.ndarray, y: np.ndarray) -> list[str]:
    hints: list[str] = []
    if x.shape == y.shape:
        hints.append("same_shape")
    changed = x != y
    if np.any(changed):
        if np.all(y[~changed] == x[~changed]):
            hints.append("sparse_edit")
        changed_to = set(int(v) for v in y[changed].ravel())
        changed_from = set(int(v) for v in x[changed].ravel())
        if len(changed_to) == 1:
            hints.append(f"single_target_color_{next(iter(changed_to))}")
        if changed_from == {0}:
            hints.append("only_fill_background")
        if 0 in changed_to:
            hints.append("some_deletion_to_background")
    xb = bbox(x != 0)
    yb = bbox(y != 0)
    if xb == yb:
        hints.append("nonzero_bbox_preserved")
    return hints


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TARGET_TASK)
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    rows: list[ExampleSummary] = []
    comp_rows: list[dict[str, Any]] = []

    for idx, ex in enumerate(examples):
        split = "train" if idx < train_n else ("test" if idx < train_n + test_n else "arc-gen")
        x = arr(ex["input"])
        y = arr(ex["output"])
        changed = x != y
        added = (x == 0) & (y != 0)
        removed = (x != 0) & (y == 0)
        pairs = Counter((int(a), int(b)) for a, b in zip(x[changed].ravel(), y[changed].ravel()))
        xcomps = components(x)
        ycomps = components(y)
        for label, comps in [("input", xcomps), ("output", ycomps)]:
            for ci, comp in enumerate(comps):
                comp_rows.append({"example_idx": idx, "split": split, "io": label, "component_idx": ci, **comp})
        rows.append(
            ExampleSummary(
                example_idx=idx,
                split_guess=split,
                input_shape=str(tuple(x.shape)),
                output_shape=str(tuple(y.shape)),
                input_colors=str(sorted(int(v) for v in np.unique(x))),
                output_colors=str(sorted(int(v) for v in np.unique(y))),
                changed_count=int(changed.sum()),
                added_count=int(added.sum()),
                removed_count=int(removed.sum()),
                changed_bbox=bbox_s(changed),
                input_nonzero_bbox=bbox_s(x != 0),
                output_nonzero_bbox=bbox_s(y != 0),
                dominant_changed_from_to=str(pairs.most_common(5)),
                component_count_input=len(xcomps),
                component_count_output=len(ycomps),
                rule_hints=";".join(infer_hints(x, y)),
            )
        )

    hint_counts = Counter()
    for row in rows:
        hint_counts.update(row.rule_hints.split(";"))
    shape_pairs = Counter((row.input_shape, row.output_shape) for row in rows)
    changed_pairs = Counter(row.dominant_changed_from_to for row in rows)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_probe_complete",
        "target_task": TARGET_TASK,
        "example_count": len(rows),
        "hint_counts": dict(hint_counts),
        "shape_pairs": {str(k): v for k, v in shape_pairs.items()},
        "top_changed_pair_patterns": dict(changed_pairs.most_common(10)),
        "candidate_rule_direction": "same-shape sparse object/background edit; inspect component_summary for fill/deletion geometry",
        "next": "Build focused rule candidates from component geometry instead of signature lookup.",
        "leakage_risk": "low: descriptive analysis only.",
        "overfitting_risk": "medium: all arc-gen included for diagnosis; future rule needs holdout.",
    }
    with (EXP_DIR / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (EXP_DIR / "example_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ExampleSummary.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])
    with (EXP_DIR / "component_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comp_rows)

    notes = f"""# {EXP_ID}

## 目的

task020のsignature lookup teacherを明示ruleへ圧縮するため、入出力差分とcomponent geometryを調べる。

## 結果

- examples: {len(rows)}
- hint counts: {dict(hint_counts)}
- shape pairs: {dict(shape_pairs)}
- candidate direction: same-shape sparse object/background edit

## 解釈

この段階ではONNXは出さない。`example_summary.csv` と `component_summary.csv` から、changed-cell maskを説明する幾何ruleを作る。

## 次

component bbox/size/colorに基づく候補ruleを列挙し、train + holdout arc-genで評価する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
