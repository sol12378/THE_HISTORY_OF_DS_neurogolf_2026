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


EXP_ID = "exp088_task185_grid_2x2_compress_rule"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185


def grid_color(arr: np.ndarray) -> int:
    vals = [int(v) for v in arr.ravel() if int(v) != 0]
    return Counter(vals).most_common(1)[0][0]


def grid_lines(arr: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    h, w = arr.shape
    rows = [r for r in range(h) if int((arr[r, :] == bg).sum()) > w * 0.8]
    cols = [c for c in range(w) if int((arr[:, c] == bg).sum()) > h * 0.8]
    return rows, cols


def predict(arr: np.ndarray) -> np.ndarray | None:
    bg = grid_color(arr)
    rows, cols = grid_lines(arr, bg)
    special = [(r, c, int(arr[r, c])) for r in rows for c in cols if int(arr[r, c]) not in (0, bg)]
    if not special:
        return None
    rr = sorted({r for r, _, _ in special})
    cc = sorted({c for _, c, _ in special})
    if len(rr) != 4 or len(cc) != 4:
        return None
    mat = np.zeros((4, 4), dtype=np.int64)
    r_index = {r: i for i, r in enumerate(rr)}
    c_index = {c: i for i, c in enumerate(cc)}
    for r, c, v in special:
        mat[r_index[r], c_index[c]] = v
    out = np.zeros((3, 3), dtype=np.int64)
    for r in range(3):
        for c in range(3):
            block = mat[r : r + 2, c : c + 2]
            vals = set(int(v) for v in block.ravel())
            if len(vals) == 1 and 0 not in vals:
                out[r, c] = int(block[0, 0])
    return out


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    rows: list[dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    fail_examples: list[dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        pred = predict(x)
        ok = pred is not None and np.array_equal(pred, y)
        if ok:
            pass_count += 1
        else:
            fail_count += 1
            if len(fail_examples) < 10:
                fail_examples.append(
                    {
                        "idx": idx,
                        "expected": y.tolist(),
                        "pred": None if pred is None else pred.tolist(),
                    }
                )
        rows.append(
            {
                "idx": idx,
                "split": "train" if idx < len(task["train"]) else ("test" if idx < len(task["train"]) + len(task["test"]) else "arc-gen"),
                "status": "pass" if ok else "fail",
                "input_shape": f"{x.shape[0]}x{x.shape[1]}",
                "output_shape": f"{y.shape[0]}x{y.shape[1]}",
            }
        )

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "split", "status", "input_shape", "output_shape"])
        writer.writeheader()
        writer.writerows(rows)

    base = load_base_tasks()[TASK_ID]
    status = "rule_found" if fail_count == 0 else "partial"
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": status,
        "hypothesis": "task185 is solved by extracting the colored 4x4 lattice matrix and emitting the color of each homogeneous nonzero 2x2 block as a 3x3 output.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "examples": len(examples),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "fail_examples": fail_examples,
        "decision": "If full pass, next lower to ONNX using grid-line index extraction or a cheaper task-specific existing artifact surgery route; target cost<=600 because output is fixed 3x3.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: Python rule only",
        "leakage_risk": "low: rule uses lattice geometry and homogeneous 2x2 blocks, not per-example lookup.",
        "overfitting_risk": "medium-low: task-specific but validated on all arc-gen; ONNX lowering must avoid hidden-specific absolute periods.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`exp087` のP0 cropish最上位 `task185` について、固定3x3出力の説明可能ruleを確認する。

## 仮説

入力は格子線上に4x4の色付きmatrixを持つ。隣接2x2 blockが非zero同色で完全に埋まる場合だけ、その色を3x3出力の対応位置に置く。

## 結果

- pass: `{pass_count}/{len(examples)}`
- fail: `{fail_count}`
- baseline cost: `{base.cost}`

## Decision

full passなら、次はONNX loweringへ進む。出力は固定3x3なので、`exp086` の3x3 `Slice+Pad` proxy `381` の近傍を狙える。ただし格子線/4x4抽出と2x2同色判定がselector costを追加するため、まずcost proxyを測る。

## Risk

- leakage risk: low。geometry ruleであり、raw lookupではない。
- overfitting risk: medium-low。all arc-genで確認済みだが、ONNX化では絶対座標/周期に過依存しない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
