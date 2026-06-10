from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
from collections import Counter
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp205_task185_arbitrary_dilated_selector_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185
CURRENT_PUBLIC_BEST_LB = 6005.93


def grid_color(arr: np.ndarray) -> int:
    vals = [int(v) for v in arr.ravel() if int(v) != 0]
    return Counter(vals).most_common(1)[0][0]


def grid_lines(arr: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    h, w = arr.shape
    rows = [r for r in range(h) if int((arr[r, :] == bg).sum()) > w * 0.8]
    cols = [c for c in range(w) if int((arr[:, c] == bg).sum()) > h * 0.8]
    return rows, cols


def valid_windows(lines: list[int]) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for i in range(len(lines) - 3):
        win = tuple(lines[i : i + 4])
        delta = [win[j + 1] - win[j] for j in range(3)]
        if len(set(delta)) == 1 and delta[0] in (3, 4, 5):
            out.append(win)
    return out


def arbitrary_windows(size: int = 30) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for spacing in (3, 4, 5):
        for start in range(size - 3 * spacing):
            out.append(tuple(start + spacing * k for k in range(4)))
    return out


def special_score(arr: np.ndarray, bg: int, axis: int, win: tuple[int, ...]) -> int:
    if axis == 0:
        vals = arr[np.asarray(win), :]
    else:
        vals = arr[:, np.asarray(win)]
    return int(((vals != 0) & (vals != bg)).sum())


def choose_axis(arr: np.ndarray, bg: int, windows: list[tuple[int, ...]], axis: int) -> tuple[int, ...]:
    return max(windows, key=lambda win: (special_score(arr, bg, axis, win), -win[0]))


def render_output(arr: np.ndarray, bg: int, rr: tuple[int, ...], cc: tuple[int, ...]) -> np.ndarray:
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
    return out


def main() -> None:
    t0 = time.perf_counter()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    base = load_base_tasks()[TASK_ID]
    valid_pass = 0
    arbitrary_pass = 0
    arbitrary_match_valid = 0
    rows: list[dict[str, Any]] = []
    fails: list[dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        bg = grid_color(x)
        line_rows, line_cols = grid_lines(x, bg)
        row_valid = valid_windows(line_rows)
        col_valid = valid_windows(line_cols)
        row_all_windows = arbitrary_windows(x.shape[0])
        col_all_windows = arbitrary_windows(x.shape[1])
        rr_valid = choose_axis(x, bg, row_valid, 0)
        cc_valid = choose_axis(x, bg, col_valid, 1)
        rr_any = choose_axis(x, bg, row_all_windows, 0)
        cc_any = choose_axis(x, bg, col_all_windows, 1)
        pred_valid = render_output(x, bg, rr_valid, cc_valid)
        pred_any = render_output(x, bg, rr_any, cc_any)
        ok_valid = np.array_equal(pred_valid, y)
        ok_any = np.array_equal(pred_any, y)
        match = rr_valid == rr_any and cc_valid == cc_any
        valid_pass += int(ok_valid)
        arbitrary_pass += int(ok_any)
        arbitrary_match_valid += int(match)
        if not ok_any and len(fails) < 12:
            fails.append(
                {
                    "idx": idx,
                    "shape": f"{x.shape[0]}x{x.shape[1]}",
                    "expected": y.tolist(),
                    "pred_any": pred_any.tolist(),
                    "rr_valid": rr_valid,
                    "cc_valid": cc_valid,
                    "rr_any": rr_any,
                    "cc_any": cc_any,
                    "scores": {
                        "rr_valid": special_score(x, bg, 0, rr_valid),
                        "rr_any": special_score(x, bg, 0, rr_any),
                        "cc_valid": special_score(x, bg, 1, cc_valid),
                        "cc_any": special_score(x, bg, 1, cc_any),
                    },
                }
            )
        rows.append(
            {
                "idx": idx,
                "shape": f"{x.shape[0]}x{x.shape[1]}",
                "ok_valid": ok_valid,
                "ok_any": ok_any,
                "match": match,
                "rr_valid": ",".join(map(str, rr_valid)),
                "cc_valid": ",".join(map(str, cc_valid)),
                "rr_any": ",".join(map(str, rr_any)),
                "cc_any": ",".join(map(str, cc_any)),
                "rr_valid_score": special_score(x, bg, 0, rr_valid),
                "rr_any_score": special_score(x, bg, 0, rr_any),
                "cc_valid_score": special_score(x, bg, 1, cc_valid),
                "cc_any_score": special_score(x, bg, 1, cc_any),
            }
        )

    with (EXP_DIR / "arbitrary_selector_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "audit_complete",
        "task_id": TASK_ID,
        "hypothesis": "exp204 mismatch is caused by scoring arbitrary dilated starts rather than detected grid-line windows.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "current_public_best_lb": CURRENT_PUBLIC_BEST_LB,
        "example_count": len(examples),
        "valid_selector_pass": valid_pass,
        "arbitrary_selector_pass": arbitrary_pass,
        "arbitrary_match_valid": arbitrary_match_valid,
        "fail_examples": fails,
        "decision": "If arbitrary_selector_pass is low, dynamic ONNX must mask candidate starts by detected grid lines or use compact valid-window templates.",
        "submission_decision": "no_submit: selector audit only",
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "leakage_risk": "low: input-only audit.",
        "overfitting_risk": "medium-low: diagnoses selector mechanics, no candidate emitted.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp204のmismatchが、detected grid-line windowsではなく任意のdilated startをscoreしていることに由来するか確認する。

## 結果

- valid_selector_pass: `{valid_pass}/{len(examples)}`
- arbitrary_selector_pass: `{arbitrary_pass}/{len(examples)}`
- arbitrary_match_valid: `{arbitrary_match_valid}/{len(examples)}`

## 判断

arbitrary selectorが低ければ、ONNX candidateにはgrid-line window maskまたはcompact valid-window templateが必要。

## リスク

- leakage risk: low。
- overfitting risk: medium-low。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
