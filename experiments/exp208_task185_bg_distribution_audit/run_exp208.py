from __future__ import annotations

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


EXP_ID = "exp208_task185_bg_distribution_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185


def grid_color(arr: np.ndarray) -> int:
    vals = [int(v) for v in arr.ravel() if int(v) != 0]
    return Counter(vals).most_common(1)[0][0]


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    base = load_base_tasks()[TASK_ID]
    bg_hist: Counter[int] = Counter()
    shape_bg_hist: Counter[str] = Counter()
    examples_by_bg: dict[str, list[int]] = {}
    for idx, ex in enumerate(examples):
        arr = grid_to_array(ex["input"])
        bg = grid_color(arr)
        bg_hist[bg] += 1
        shape_bg_hist[f"{arr.shape[0]}x{arr.shape[1]}_bg{bg}"] += 1
        examples_by_bg.setdefault(str(bg), []).append(idx)
    result: dict[str, Any] = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "audit_complete",
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "example_count": len(examples),
        "bg_hist": dict(sorted(bg_hist.items())),
        "shape_bg_hist": dict(sorted(shape_bg_hist.items())),
        "examples_by_bg_sample": {k: v[:12] for k, v in sorted(examples_by_bg.items())},
        "decision": "If one bg dominates all examples, a fixed bg suppressor can debug candidate; otherwise dynamic bg detection is required.",
        "submission_decision": "no_submit: audit only",
        "leakage_risk": "low: input distribution audit.",
        "overfitting_risk": "medium-low: fixed bg would overfit if bg varies.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task185 selector scoreで背景色を除外する必要があるため、背景色分布を確認する。

## 結果

- bg_hist: `{dict(sorted(bg_hist.items()))}`
- shape_bg_hist: `{dict(sorted(shape_bg_hist.items()))}`

## 判断

背景色が固定なら固定maskでcandidate debugを続ける。変動するならdynamic bg detectionが必要。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
