from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, deque
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp223_small_output_mask_shape_inventory"
EXP_DIR = ROOT / "experiments" / EXP_ID
CANDIDATE_CSV = ROOT / "experiments" / "exp087_small_output_crop_candidate_scan" / "small_output_candidates.csv"


def load_candidate_task_ids() -> list[int]:
    out: list[int] = []
    with CANDIDATE_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["is_cropish"] == "True" and int(row["max_output_area"]) <= 16:
                out.append(int(row["task_id"]))
    return sorted(set(out))


def bbox_shape(x: np.ndarray) -> tuple[int, int]:
    nz = np.argwhere(x != 0)
    if nz.size == 0:
        return x.shape
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0) + 1
    return int(r1 - r0), int(c1 - c0)


def bbox_crop(x: np.ndarray) -> np.ndarray:
    nz = np.argwhere(x != 0)
    if nz.size == 0:
        return x
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0) + 1
    return x[int(r0) : int(r1), int(c0) : int(c1)]


def components(x: np.ndarray) -> list[dict[str, Any]]:
    seen = np.zeros(x.shape, dtype=bool)
    comps: list[dict[str, Any]] = []
    for r, c in np.argwhere(x != 0):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        color = int(x[r, c])
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            cells.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < x.shape[0] and 0 <= nc < x.shape[1] and not seen[nr, nc] and int(x[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        arr = np.asarray(cells)
        h = int(arr[:, 0].max() - arr[:, 0].min() + 1)
        w = int(arr[:, 1].max() - arr[:, 1].min() + 1)
        comps.append({"color": color, "size": len(cells), "shape": (h, w)})
    return comps


def binary_signature(y: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    vals = sorted({int(v) for v in y.ravel()})
    if len(vals) <= 1:
        mask = tuple(0 for _ in y.ravel())
    else:
        bg = max(vals, key=lambda v: int(np.sum(y == v)))
        mask = tuple(int(int(v) != bg) for v in y.ravel())
    return tuple(y.shape), mask


def shape_rule_hits(examples: list[dict[str, Any]]) -> dict[str, int]:
    hits = Counter()
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        out_shape = tuple(y.shape)
        comps = components(x)
        comp_shapes = [tuple(c["shape"]) for c in comps]
        if out_shape == tuple(x.shape):
            hits["input_shape"] += 1
        if out_shape == bbox_shape(x):
            hits["bbox_shape"] += 1
        if out_shape in comp_shapes:
            hits["one_component_shape"] += 1
        if comps and out_shape == max(comp_shapes, key=lambda s: s[0] * s[1]):
            hits["largest_component_bbox_shape"] += 1
        if out_shape == (len({int(v) for v in x.ravel() if int(v) != 0}), 1):
            hits["num_colors_by_1"] += 1
        if out_shape == (1, len({int(v) for v in x.ravel() if int(v) != 0})):
            hits["1_by_num_colors"] += 1
    return {k: int(v) for k, v in hits.items()}


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()
    rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for task_id in load_candidate_task_ids():
        task = load_task(task_id)
        examples = task["train"] + task["test"] + task["arc-gen"]
        out_shapes = Counter(tuple(grid_to_array(ex["output"]).shape) for ex in examples)
        sigs = Counter(binary_signature(grid_to_array(ex["output"])) for ex in examples)
        rules = shape_rule_hits(examples)
        best_shape_rule = max(rules.items(), key=lambda kv: (kv[1], kv[0])) if rules else ("", 0)
        output_color_counts = Counter(len({int(v) for v in grid_to_array(ex["output"]).ravel()}) for ex in examples)
        row = {
            "task_id": task_id,
            "baseline_cost": base[task_id].cost,
            "baseline_points": base[task_id].points,
            "example_count": len(examples),
            "output_shape_count": len(out_shapes),
            "output_shapes": ";".join(f"{h}x{w}:{n}" for (h, w), n in sorted(out_shapes.items())),
            "binary_signature_count": len(sigs),
            "most_common_binary_signature_count": sigs.most_common(1)[0][1],
            "output_color_count_hist": dict(sorted(output_color_counts.items())),
            "shape_rule_hits": rules,
            "best_shape_rule": best_shape_rule[0],
            "best_shape_rule_pass_count": best_shape_rule[1],
            "best_shape_rule_fail_count": len(examples) - best_shape_rule[1],
        }
        result_rows.append(row)
        rows.append(
            {
                "task_id": task_id,
                "baseline_cost": base[task_id].cost,
                "example_count": len(examples),
                "output_shape_count": len(out_shapes),
                "binary_signature_count": len(sigs),
                "most_common_binary_signature_count": sigs.most_common(1)[0][1],
                "best_shape_rule": best_shape_rule[0],
                "best_shape_rule_pass_count": best_shape_rule[1],
                "best_shape_rule_fail_count": len(examples) - best_shape_rule[1],
                "output_shapes": row["output_shapes"],
            }
        )

    rows.sort(key=lambda r: (r["best_shape_rule_fail_count"], -r["baseline_cost"], r["task_id"]))
    with (EXP_DIR / "small_output_mask_shape_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "baseline_cost",
                "example_count",
                "output_shape_count",
                "binary_signature_count",
                "most_common_binary_signature_count",
                "best_shape_rule",
                "best_shape_rule_pass_count",
                "best_shape_rule_fail_count",
                "output_shapes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    shape_full_hits = [r for r in result_rows if r["best_shape_rule_fail_count"] == 0]
    stable_binary = [r for r in result_rows if r["binary_signature_count"] <= 3 and r["baseline_cost"] > 1000]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "inventory_ready",
        "task_count": len(result_rows),
        "shape_full_hits": sorted(shape_full_hits, key=lambda r: (-r["baseline_cost"], r["task_id"]))[:20],
        "stable_binary_high_cost": sorted(stable_binary, key=lambda r: (-r["baseline_cost"], r["task_id"]))[:20],
        "top_rows": rows[:20],
        "decision": "shape_full_hitsはshape-branch supplier候補。stable_binary_high_costはmask/template supplier候補として個別監査する。",
        "submission_decision": "no_submit: inventory only",
        "leakage_risk": "low: input/output structural inventory only.",
        "overfitting_risk": "low: no candidate selected yet.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

小出力cropish候補について、出力shapeが入力bbox/component/color-countなどの簡単なshape特徴で説明できるか、またbinary mask/templateとして安定しているかを棚卸しする。

## 結果

- task_count: `{len(result_rows)}`
- shape_full_hits(top): `{[(r['task_id'], r['baseline_cost'], r['best_shape_rule'], r['output_shapes']) for r in sorted(shape_full_hits, key=lambda r: (-r['baseline_cost'], r['task_id']))[:10]]}`
- stable_binary_high_cost(top): `{[(r['task_id'], r['baseline_cost'], r['binary_signature_count'], r['output_shapes']) for r in sorted(stable_binary, key=lambda r: (-r['baseline_cost'], r['task_id']))[:10]]}`

## 判断

shape_full_hitsはshape-branch supplier候補、stable_binary_high_costはmask/template supplier候補として個別監査する。提出なし。

## リスク

- leakage risk: low。
- overfitting risk: low。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
