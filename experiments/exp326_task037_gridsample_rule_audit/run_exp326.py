from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp326_task037_gridsample_rule_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 37


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def nearest_color_in_direction(x: np.ndarray, r: int, c: int, dr: int, dc: int) -> tuple[int | None, int | None]:
    rr, cc = r + dr, c + dc
    dist = 1
    while 0 <= rr < x.shape[0] and 0 <= cc < x.shape[1]:
        if x[rr, cc] != 0:
            return int(x[rr, cc]), dist
        rr += dr
        cc += dc
        dist += 1
    return None, None


def opposite_ray_diag_same_color(x: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]]]:
    out = x.copy()
    changes: list[dict[str, int]] = []
    for r, c in np.argwhere(x == 0):
        votes: list[tuple[int, int, int]] = []
        for dr, dc in ((1, 1), (1, -1)):
            a, da = nearest_color_in_direction(x, int(r), int(c), dr, dc)
            b, db = nearest_color_in_direction(x, int(r), int(c), -dr, -dc)
            if a is not None and a == b and da is not None and db is not None:
                votes.append((a, da, db))
        colors = {v[0] for v in votes}
        if len(colors) == 1 and votes:
            color = votes[0][0]
            out[int(r), int(c)] = color
            changes.append(
                {
                    "r": int(r),
                    "c": int(c),
                    "color": color,
                    "min_endpoint_distance": min(min(da, db) for _, da, db in votes),
                    "max_endpoint_distance": max(max(da, db) for _, da, db in votes),
                    "vote_count": len(votes),
                }
            )
    return out, changes


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = examples_for(task, -1)
    split_counts = {"train": [0, len(task["train"])], "test": [0, len(task["test"])], "arc-gen": [0, len(task["arc-gen"])]}
    changed_counts: list[int] = []
    generated_new_color = 0
    endpoint_distances = Counter()
    vote_counts = Counter()
    output_new_color_examples = 0
    mismatch_examples: list[int] = []
    for idx, ex in enumerate(examples):
        split = "train" if idx < len(task["train"]) else ("test" if idx < len(task["train"]) + len(task["test"]) else "arc-gen")
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred, changes = opposite_ray_diag_same_color(x)
        if np.array_equal(pred, y):
            split_counts[split][0] += 1
        else:
            mismatch_examples.append(idx)
        changed_counts.append(int(np.count_nonzero(x != y)))
        input_colors = set(int(v) for v in np.unique(x) if int(v) != 0)
        output_colors = set(int(v) for v in np.unique(y) if int(v) != 0)
        if not output_colors.issubset(input_colors):
            output_new_color_examples += 1
        for change in changes:
            if change["color"] not in input_colors:
                generated_new_color += 1
            endpoint_distances[change["max_endpoint_distance"]] += 1
            vote_counts[change["vote_count"]] += 1

    total_pass = sum(v[0] for v in split_counts.values())
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-11",
        "status": "rule_audit_complete",
        "task_id": TASK_ID,
        "hypothesis": "task037 is a better GridSample/shift target than task251 because the fill color is copied from existing diagonal endpoints, not created as a new color.",
        "validation": {
            "status": f"{total_pass}_pass_{len(examples) - total_pass}_fail",
            "splits": {k: {"pass": v[0], "total": v[1]} for k, v in split_counts.items()},
            "mismatch_examples": mismatch_examples[:20],
        },
        "data_profile": {
            "example_count": len(examples),
            "shape": [10, 10],
            "changed_cells_min": min(changed_counts),
            "changed_cells_mean": float(np.mean(changed_counts)),
            "changed_cells_max": max(changed_counts),
            "output_new_color_examples": output_new_color_examples,
            "generated_new_color_cells": generated_new_color,
            "max_endpoint_distance_hist": dict(sorted(endpoint_distances.items())),
            "vote_count_hist": dict(sorted(vote_counts.items())),
        },
        "decision": (
            "task037 remains a plausible GridSample/shift lowering target: the rule is full-arc valid, "
            "outputs introduce no new nonzero colors, and each changed cell copies an existing diagonal endpoint color. "
            "Next implementation should avoid full-grid Conv visibility and instead test a bounded diagonal shift stack "
            "or GridSample-generated diagonal samples with cheap equality masks."
        ),
        "submission_decision": "no_submit: audit only; no ONNX replacement generated.",
        "leakage_risk": "low: revalidates an explanatory rule on local task examples only.",
        "overfitting_risk": "low-to-medium: task-specific rule is full-arc validated, but a future ONNX candidate still needs cost and Kaggle calibration.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp325 採点待ち中の独立作業。task037 が task251 より GridSample/shift lowering に向くか、rule と data profile を再監査する。

## 結果

- rule: `opposite_ray_diag_same_color`
- validation: `{result['validation']['status']}`
- changed cells mean: `{result['data_profile']['changed_cells_mean']}`
- output new color examples: `{result['data_profile']['output_new_color_examples']}`
- generated new color cells: `{result['data_profile']['generated_new_color_cells']}`
- max endpoint distance hist: `{result['data_profile']['max_endpoint_distance_hist']}`

## 判断

{result['decision']}

## Submission

`{result['submission_decision']}`

## Risk

- leakage risk: {result['leakage_risk']}
- overfitting risk: {result['overfitting_risk']}
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
