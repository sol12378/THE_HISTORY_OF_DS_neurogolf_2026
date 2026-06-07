from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp116_task185_window_selector_rule"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185
CURRENT_LOCAL_ESTIMATE = 6282.93615685804
CAMPAIGN_INDEX = 18


def grid_color(arr: np.ndarray) -> int:
    vals = [int(v) for v in arr.ravel() if int(v) != 0]
    return Counter(vals).most_common(1)[0][0]


def grid_lines(arr: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    h, w = arr.shape
    rows = [r for r in range(h) if int((arr[r, :] == bg).sum()) > w * 0.8]
    cols = [c for c in range(w) if int((arr[:, c] == bg).sum()) > h * 0.8]
    return rows, cols


def windows(lines: list[int]) -> list[tuple[int, int, int, int]]:
    out = []
    for i in range(len(lines) - 3):
        win = tuple(lines[i : i + 4])
        delta = [win[j + 1] - win[j] for j in range(3)]
        if len(set(delta)) == 1 and delta[0] in (3, 4, 5):
            out.append(win)
    return out


def select_window_pair(arr: np.ndarray, bg: int, rows: list[int], cols: list[int]) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]] | None:
    row_windows = windows(rows)
    col_windows = windows(cols)
    best: tuple[int, tuple[int, ...], tuple[int, ...], int] | None = None
    # The correct lattice is a dense 4x4-ish set of special intersections. Pick
    # the row/col 4-window pair with the most non-bg/non-zero intersections.
    for rr in row_windows:
        for cc in col_windows:
            vals = [int(arr[r, c]) for r in rr for c in cc]
            score = sum(1 for v in vals if v not in (0, bg))
            filled_rows = sum(any(int(arr[r, c]) not in (0, bg) for c in cc) for r in rr)
            filled_cols = sum(any(int(arr[r, c]) not in (0, bg) for r in rr) for c in cc)
            tie = filled_rows + filled_cols
            candidate = (score, rr, cc, tie)
            if best is None or (score, tie, -rr[0], -cc[0]) > (best[0], best[3], -best[1][0], -best[2][0]):
                best = candidate
    if best is None:
        return None
    score, rr, cc, tie = best
    return rr, cc, {"special_score": score, "tie_score": tie, "row_window_count": len(row_windows), "col_window_count": len(col_windows)}


def predict(arr: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    bg = grid_color(arr)
    rows, cols = grid_lines(arr, bg)
    selected = select_window_pair(arr, bg, rows, cols)
    if selected is None:
        return None, {"bg": bg, "line_rows": rows, "line_cols": cols, "reason": "no_window"}
    rr, cc, meta = selected
    mat = np.zeros((4, 4), dtype=np.int64)
    for i, r in enumerate(rr):
        for j, c in enumerate(cc):
            v = int(arr[r, c])
            mat[i, j] = 0 if v in (0, bg) else v
    out = np.zeros((3, 3), dtype=np.int64)
    for r in range(3):
        for c in range(3):
            block = mat[r : r + 2, c : c + 2]
            vals = set(int(v) for v in block.ravel())
            if len(vals) == 1 and 0 not in vals:
                out[r, c] = int(block[0, 0])
    meta.update({"bg": bg, "line_rows": rows, "line_cols": cols, "rr": rr, "cc": cc})
    return out, meta


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    rows: list[dict[str, Any]] = []
    fail_examples: list[dict[str, Any]] = []
    pass_count = 0
    for idx, ex in enumerate(examples):
        split = "train" if idx < len(task["train"]) else ("test" if idx < len(task["train"]) + len(task["test"]) else "arc-gen")
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        pred, meta = predict(x)
        ok = pred is not None and np.array_equal(pred, y)
        pass_count += int(ok)
        if not ok and len(fail_examples) < 20:
            fail_examples.append({"idx": idx, "expected": y.tolist(), "pred": None if pred is None else pred.tolist(), "meta": meta})
        rows.append(
            {
                "idx": idx,
                "split": split,
                "status": "pass" if ok else "fail",
                "shape": f"{x.shape[0]}x{x.shape[1]}",
                "rr": "" if "rr" not in meta else ",".join(map(str, meta["rr"])),
                "cc": "" if "cc" not in meta else ",".join(map(str, meta["cc"])),
                "special_score": meta.get("special_score", ""),
                "row_window_count": meta.get("row_window_count", ""),
                "col_window_count": meta.get("col_window_count", ""),
            }
        )

    with (EXP_DIR / "selector_eval.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["idx", "split", "status", "shape", "rr", "cc", "special_score", "row_window_count", "col_window_count"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    base = load_base_tasks()[TASK_ID]
    fail_count = len(examples) - pass_count
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if fail_count == 0 else "partial",
        "campaign_index": CAMPAIGN_INDEX,
        "task_id": TASK_ID,
        "hypothesis": "Task185 lattice positions can be selected by scoring 4-consecutive grid-line windows, avoiding a raw position table.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "example_count": len(examples),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "fail_examples": fail_examples,
        "base_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "local_estimate_delta": 0.0,
        "new_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "decision": "If full pass, next ONNX probe should implement low-cost window scoring or approximate it with shape/spacing branches.",
        "submission_decision": "no_submit: Python selector rule only",
        "leakage_risk": "low: input-only geometry selector.",
        "overfitting_risk": "medium-low if implemented as window scoring; high if converted to raw position table.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## Hypothesis

task185のlattice位置はraw coordinate tableではなく、grid line上の4連続windowをscoreして選べる。

## Result

- pass: `{pass_count}/{len(examples)}`
- fail: `{fail_count}`
- local delta: `0.000000`

## Interpretation

full passなら、次はONNXでwindow scoringを低cost化する。partialなら失敗例からtie-breakerを追加する。

## Risk

- leakage risk: low。
- overfitting risk: medium-low。window scoringなら許容、raw position table化は避ける。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
