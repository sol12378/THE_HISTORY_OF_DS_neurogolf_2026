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


EXP_ID = "exp091_task048_bridge_connectivity_rule"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 48


def components(arr: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = arr.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(arr[r, c]) != color or (r, c) in seen:
                continue
            q = [(r, c)]
            seen.add((r, c))
            comp: list[tuple[int, int]] = []
            while q:
                a, b = q.pop()
                comp.append((a, b))
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = a + dr, b + dc
                    if 0 <= nr < h and 0 <= nc < w and int(arr[nr, nc]) == color and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def touches(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    bs = set(b)
    return any((r + dr, c + dc) in bs for r, c in a for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)])


def predict(arr: np.ndarray) -> np.ndarray:
    comps2 = components(arr, 2)
    comps8 = components(arr, 8)
    bridged = False
    if len(comps2) >= 2:
        for comp8 in comps8:
            if sum(1 for comp2 in comps2 if touches(comp8, comp2)) >= 2:
                bridged = True
                break
    return np.asarray([[8 if bridged else 0]], dtype=np.int64)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    rows: list[dict[str, Any]] = []
    pass_count = 0
    fails: list[dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        pred = predict(x)
        ok = np.array_equal(pred, y)
        pass_count += int(ok)
        if not ok and len(fails) < 10:
            fails.append({"idx": idx, "expected": y.tolist(), "pred": pred.tolist()})
        rows.append(
            {
                "idx": idx,
                "split": "train" if idx < len(task["train"]) else ("test" if idx < len(task["train"]) + len(task["test"]) else "arc-gen"),
                "status": "pass" if ok else "fail",
                "input_shape": f"{x.shape[0]}x{x.shape[1]}",
                "out": int(y[0, 0]),
                "color_counts": dict(Counter(int(v) for v in x.ravel())),
            }
        )
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "split", "status", "input_shape", "out", "color_counts"])
        writer.writeheader()
        writer.writerows(rows)
    base = load_base_tasks()[TASK_ID]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "rule_found" if pass_count == len(examples) else "partial",
        "hypothesis": "task048 outputs 8 iff a 4-connected component of color 8 touches both color-2 components; otherwise it outputs 0.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "examples": len(examples),
        "pass_count": pass_count,
        "fail_count": len(examples) - pass_count,
        "fail_examples": fails,
        "decision": "If full pass, probe small 8x8 connectivity lowering. Because output is 1x1 and max input is 8x8, this is a stronger <=600 candidate than task185.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: Python rule only",
        "leakage_risk": "low: explicit connectivity rule over colors 2 and 8.",
        "overfitting_risk": "medium-low: task-specific colors but all arc-gen pass.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`exp087` の1x1 P0候補 `task048` の説明可能ruleを確認する。

## 結果

- pass: `{pass_count}/{len(examples)}`
- baseline cost: `{base.cost}`

## Rule

色8の4-connected componentが2つの色2 componentの両方に接していれば出力は `[[8]]`。そうでなければ `[[0]]`。

## Decision

full passなら、次は8x8小領域connectivity loweringのcost probeへ進む。出力1x1なので、task185より600級候補として強い。

## Risk

- leakage risk: low。明示的な連結性rule。
- overfitting risk: medium-low。色2/8固定だがall arc-gen pass。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
