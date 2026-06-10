from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, grid_to_array, load_task


EXP_DIR = ROOT / "experiments" / "exp162_task025_rule_diagnostic"
TASK_ID = 25


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def color_counts(arr: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    n = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            n += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return n


def relation_signature(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    changed = x != y
    same_shape = x.shape == y.shape
    row: dict[str, Any] = {
        "same_shape": same_shape,
        "input_shape": str(tuple(x.shape)),
        "output_shape": str(tuple(y.shape)),
        "input_colors": " ".join(f"{k}:{v}" for k, v in sorted(color_counts(x).items())),
        "output_colors": " ".join(f"{k}:{v}" for k, v in sorted(color_counts(y).items())),
    }
    if not same_shape:
        return row
    changed_count = int(changed.sum())
    row.update(
        {
            "changed_count": changed_count,
            "changed_bbox": str(bbox(changed)),
            "changed_components": component_count(changed),
            "changed_from_to": " ".join(f"{a}->{b}:{n}" for (a, b), n in Counter(zip(x[changed].tolist(), y[changed].tolist())).most_common()),
            "input_nonzero_bbox": str(bbox(x != 0)),
            "output_nonzero_bbox": str(bbox(y != 0)),
            "input_nonzero_count": int((x != 0).sum()),
            "output_nonzero_count": int((y != 0).sum()),
            "output_equals_input_plus_sparse": changed_count <= 20,
            "all_changes_on_zero": bool(np.all(x[changed] == 0)) if changed_count else True,
            "all_changes_to_existing_colors": bool(set(y[changed].tolist()).issubset(set(x.flatten().tolist()))) if changed_count else True,
        }
    )
    if changed_count:
        coords = np.argwhere(changed)
        row["changed_coords_norm"] = " ".join(f"{int(r)}:{int(c)}:{int(x[r,c])}->{int(y[r,c])}" for r, c in coords[:30])
    return row


def try_rules(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rule_rows = []
    arrays = [(grid_to_array(ex["input"]), grid_to_array(ex["output"])) for ex in examples]
    same_shape = [(x, y) for x, y in arrays if x.shape == y.shape]
    if len(same_shape) != len(arrays):
        return [{"rule": "same_shape_required", "passed": 0, "total": len(arrays), "note": "shape mismatch exists"}]

    # Diagnostic baselines: identity, copy fixed color transforms, fill zero cells whose row/col has nonzero.
    candidates = {
        "identity": lambda x: x.copy(),
        "fill_zero_with_most_common_nonzero": fill_zero_with_most_common_nonzero,
        "row_col_intersection_fill": row_col_intersection_fill,
        "bbox_hole_fill": bbox_hole_fill,
        "mirror_horizontal": lambda x: np.fliplr(x),
        "mirror_vertical": lambda x: np.flipud(x),
        "rot180": lambda x: np.rot90(x, 2),
    }
    for name, fn in candidates.items():
        passed = 0
        first_fail = ""
        for idx, (x, y) in enumerate(arrays):
            try:
                pred = fn(x)
            except Exception as exc:
                pred = None
                first_fail = f"exception {idx}: {exc}"
            if pred is not None and pred.shape == y.shape and np.array_equal(pred, y):
                passed += 1
            elif not first_fail:
                first_fail = f"mismatch {idx}"
        rule_rows.append({"rule": name, "passed": passed, "total": len(arrays), "note": first_fail})
    return rule_rows


def fill_zero_with_most_common_nonzero(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    nz = x[x != 0]
    if nz.size == 0:
        return y
    color = Counter(nz.tolist()).most_common(1)[0][0]
    y[x == 0] = color
    return y


def row_col_intersection_fill(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    nz = x != 0
    rows = nz.any(axis=1)
    cols = nz.any(axis=0)
    targets = (x == 0) & rows[:, None] & cols[None, :]
    nzvals = x[nz]
    color = Counter(nzvals.tolist()).most_common(1)[0][0] if nzvals.size else 0
    y[targets] = color
    return y


def bbox_hole_fill(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    b = bbox(x != 0)
    if b is None:
        return y
    r0, c0, r1, c1 = b
    nz = x[x != 0]
    color = Counter(nz.tolist()).most_common(1)[0][0] if nz.size else 0
    region = (x[r0 : r1 + 1, c0 : c1 + 1] == 0)
    y[r0 : r1 + 1, c0 : c1 + 1][region] = color
    return y


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    splits = [("train", ex) for ex in task["train"]] + [("test", ex) for ex in task["test"]] + [("arc-gen", ex) for ex in task["arc-gen"]]
    examples = [ex for _, ex in splits]

    rows = []
    for idx, (split, ex) in enumerate(splits):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        rows.append({"idx": idx, "split": split, **relation_signature(x, y)})

    out_csv = EXP_DIR / "task025_example_diagnostic.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rule_rows = try_rules(examples)
    rule_csv = EXP_DIR / "task025_rule_probe.csv"
    with rule_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rule_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rule_rows)

    summary = {
        "shape_counts": dict(Counter(f"{row['input_shape']} -> {row['output_shape']}" for row in rows)),
        "changed_count_counts": dict(Counter(row.get("changed_count", "") for row in rows)),
        "changed_from_to_top": dict(Counter(row.get("changed_from_to", "") for row in rows).most_common(12)),
        "changed_component_counts": dict(Counter(row.get("changed_components", "") for row in rows)),
        "all_changes_on_zero_count": sum(1 for row in rows if row.get("all_changes_on_zero") is True),
        "all_changes_to_existing_colors_count": sum(1 for row in rows if row.get("all_changes_to_existing_colors") is True),
    }
    best_rule = max(rule_rows, key=lambda r: int(r["passed"]))
    result: dict[str, Any] = {
        "exp_id": "exp162_task025_rule_diagnostic",
        "date": "2026-06-10",
        "status": "diagnostic_complete",
        "task_id": TASK_ID,
        "example_count": len(examples),
        "summary": summary,
        "rule_probe": rule_rows,
        "best_rule_probe": best_rule,
        "outputs": {"example_diagnostic": str(out_csv.relative_to(ROOT)), "rule_probe": str(rule_csv.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "decision": decide(summary, best_rule),
        "leakage_risk": "low: diagnostic over provided train/test/arc-gen examples only.",
        "overfitting_risk": "medium: rule inference from local examples may not generalize to public/private hidden variants.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def decide(summary: dict[str, Any], best_rule: dict[str, Any]) -> str:
    if int(best_rule["passed"]) == int(best_rule["total"]):
        return f"lower_full_pass_rule:{best_rule['rule']}"
    if summary["all_changes_on_zero_count"] == sum(summary["changed_count_counts"].values()):
        return "changes_are_zero_fill_like; mine target-cell predicate and fill-color rule next"
    return "no_simple_rule_hit; inspect diagnostic csv and mine task-specific predicate"


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp162_task025_rule_diagnostic",
        "",
        "## 目的",
        "",
        "public-zeroだが既存source差し替え不可だったtask025について、rule repairの入口として入出力差分と単純ruleを診断する。",
        "",
        "## 結果",
        "",
        f"- example_count: `{result['example_count']}`",
        f"- best_rule_probe: `{result['best_rule_probe']}`",
        f"- decision: {result['decision']}",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(result["summary"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## リスク",
        "",
        str(result["leakage_risk"]),
        str(result["overfitting_risk"]),
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
