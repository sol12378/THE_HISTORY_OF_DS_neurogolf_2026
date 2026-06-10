from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_task


EXP_DIR = ROOT / "experiments" / "exp164_task025_line_projection_rule"
TASK_ID = 25


def detect_full_lines(x: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    row_lines = [int(r) for r in range(x.shape[0]) if np.all(x[r, :] == color)]
    col_lines = [int(c) for c in range(x.shape[1]) if np.all(x[:, c] == color)]
    return row_lines, col_lines


def apply_line_projection_rule(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    colors = sorted(int(c) for c in np.unique(x) if int(c) != 0)
    for color in colors:
        row_lines, col_lines = detect_full_lines(x, color)
        if row_lines:
            for r in row_lines:
                y[r, :] = color
            for r, c in np.argwhere(x == color):
                r = int(r)
                c = int(c)
                if r in row_lines:
                    continue
                line_r = min(row_lines, key=lambda rr: (abs(r - rr), rr))
                dst_r = line_r - 1 if r < line_r else line_r + 1
                if 0 <= dst_r < x.shape[0]:
                    y[dst_r, c] = color
            continue
        if col_lines:
            for c in col_lines:
                y[:, c] = color
            for r, c in np.argwhere(x == color):
                r = int(r)
                c = int(c)
                if c in col_lines:
                    continue
                line_c = min(col_lines, key=lambda cc: (abs(c - cc), cc))
                dst_c = line_c - 1 if c < line_c else line_c + 1
                if 0 <= dst_c < x.shape[1]:
                    y[r, dst_c] = color
    return y


def row_to_text(arr: np.ndarray) -> str:
    return "\n".join("".join(str(int(v)) if int(v) else "." for v in row) for row in arr)


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    splits = [("train", ex) for ex in task["train"]] + [("test", ex) for ex in task["test"]] + [("arc-gen", ex) for ex in task["arc-gen"]]
    rows: list[dict[str, Any]] = []
    first_fail: dict[str, Any] | None = None
    pass_count = 0
    for idx, (split, ex) in enumerate(splits):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        pred = apply_line_projection_rule(x)
        ok = bool(np.array_equal(pred, y))
        if ok:
            pass_count += 1
        diff = int((pred != y).sum())
        row = {"idx": idx, "split": split, "ok": ok, "diff": diff, "shape": str(tuple(x.shape))}
        rows.append(row)
        if first_fail is None and not ok:
            first_fail = {
                **row,
                "input": row_to_text(x),
                "expected": row_to_text(y),
                "pred": row_to_text(pred),
            }

    out_csv = EXP_DIR / "task025_line_projection_eval.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    result = {
        "exp_id": "exp164_task025_line_projection_rule",
        "date": "2026-06-10",
        "status": "rule_probe_complete",
        "task_id": TASK_ID,
        "pass_count": pass_count,
        "total": total,
        "first_fail": first_fail,
        "outputs": {"eval_csv": str(out_csv.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "decision": "rule_found_lower_next" if pass_count == total else "partial_rule_refine_projection_or_line_detection",
        "leakage_risk": "low: input-only geometric rule over provided train/test/arc-gen examples.",
        "overfitting_risk": "medium: full-line assumption may fail if hidden variants use broken or partial guide lines.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp164_task025_line_projection_rule",
        "",
        "## 目的",
        "",
        "task025について、完全な縦/横ラインをguideとしてstray cellをライン隣接セルへ射影するinput-only ruleを検証する。",
        "",
        "## 結果",
        "",
        f"- pass_count: `{result['pass_count']}/{result['total']}`",
        f"- decision: {result['decision']}",
        "",
        "## リスク",
        "",
        result["leakage_risk"],
        result["overfitting_risk"],
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
