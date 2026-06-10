from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from datetime import date
from typing import Any, Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp211_onecell_cropish_rule_sweep"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_IDS = [355, 346, 291, 48]


def counts(x: np.ndarray, include_zero: bool = False) -> Counter[int]:
    return Counter(int(v) for v in x.ravel() if include_zero or int(v) != 0)


def choose_count(x: np.ndarray, reverse: bool, include_zero: bool = False) -> int:
    c = counts(x, include_zero)
    if not c:
        return 0
    return min(c, key=lambda k: ((-c[k] if reverse else c[k]), k))


def bbox_crop(x: np.ndarray) -> np.ndarray:
    nz = np.argwhere(x != 0)
    if nz.size == 0:
        return x
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0) + 1
    return x[int(r0) : int(r1), int(c0) : int(c1)]


def funcs() -> list[tuple[str, Callable[[np.ndarray], int]]]:
    return [
        ("mode_nz", lambda x: choose_count(x, True)),
        ("least_nz", lambda x: choose_count(x, False)),
        ("mode_all", lambda x: choose_count(x, True, True)),
        ("center", lambda x: int(x[x.shape[0] // 2, x.shape[1] // 2])),
        ("tl", lambda x: int(x[0, 0])),
        ("tr", lambda x: int(x[0, -1])),
        ("bl", lambda x: int(x[-1, 0])),
        ("br", lambda x: int(x[-1, -1])),
        ("bbox_center", lambda x: int((b := bbox_crop(x))[b.shape[0] // 2, b.shape[1] // 2])),
        ("bbox_mode", lambda x: choose_count(bbox_crop(x), True)),
        ("bbox_least", lambda x: choose_count(bbox_crop(x), False)),
    ]


def output_color(ex: dict[str, Any]) -> int:
    y = grid_to_array(ex["output"])
    return int(y[0, 0])


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()
    rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        task = load_task(task_id)
        examples = task["train"] + task["test"] + task["arc-gen"]
        pass_counts = Counter()
        output_hist = Counter(output_color(ex) for ex in examples)
        for ex in examples:
            x = grid_to_array(ex["input"])
            y = output_color(ex)
            for name, fn in funcs():
                pass_counts[name] += int(int(fn(x)) == y)
        ranked = sorted(
            [{"rule": name, "pass_count": int(pass_counts[name]), "fail_count": len(examples) - int(pass_counts[name])} for name, _ in funcs()],
            key=lambda r: (-r["pass_count"], r["rule"]),
        )
        best = ranked[0]
        result_rows.append(
            {
                "task_id": task_id,
                "baseline_cost": base[task_id].cost,
                "baseline_points": base[task_id].points,
                "example_count": len(examples),
                "output_hist": dict(sorted(output_hist.items())),
                "best_rule": best["rule"],
                "best_pass_count": best["pass_count"],
                "best_fail_count": best["fail_count"],
                "ranked_rules": ranked,
            }
        )
        for rank, item in enumerate(ranked):
            rows.append({"task_id": task_id, "rank": rank, **item})
    with (EXP_DIR / "onecell_rule_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "rank", "rule", "pass_count", "fail_count"])
        writer.writeheader()
        writer.writerows(rows)
    full_hits = [r for r in result_rows if r["best_fail_count"] == 0]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if full_hits else "no_full_hit",
        "task_ids": TASK_IDS,
        "rows": result_rows,
        "full_hits": full_hits,
        "decision": "If full hit exists, lower with tiny ONNX; otherwise simple one-cell aggregate lane is not enough for these tasks.",
        "submission_decision": "no_submit: rule sweep only",
        "leakage_risk": "low: input-only aggregate rules.",
        "overfitting_risk": "medium-low unless branch tables are added.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

1x1 cropish候補にtask355の単純集約rule minerを横展開し、短いruleの当たりを探す。

## 結果

- full_hits: `{[r['task_id'] for r in full_hits]}`
- best: `{[(r['task_id'], r['best_rule'], r['best_pass_count'], r['example_count']) for r in result_rows]}`

## 判断

full hitがあればtiny ONNX化。なければこの単純aggregate laneは打ち切り、別rule familyへ移る。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
