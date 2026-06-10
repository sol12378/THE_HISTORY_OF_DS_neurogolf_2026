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


EXP_ID = "exp212_task346_least_color_failure_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 346


def counts_nz(x: np.ndarray) -> Counter[int]:
    return Counter(int(v) for v in x.ravel() if int(v) != 0)


def least_nz(x: np.ndarray) -> int:
    c = counts_nz(x)
    return min(c, key=lambda k: (c[k], k)) if c else 0


def ranked_counts(x: np.ndarray) -> list[tuple[int, int]]:
    c = counts_nz(x)
    return sorted(c.items(), key=lambda kv: (kv[1], kv[0]))


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    base = load_base_tasks()[TASK_ID]
    pass_count = 0
    fails: list[dict[str, Any]] = []
    tie_hist = Counter()
    output_rank_hist = Counter()
    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = int(grid_to_array(ex["output"])[0, 0])
        ranked = ranked_counts(x)
        pred = ranked[0][0]
        ok = pred == y
        pass_count += int(ok)
        min_count = ranked[0][1]
        tied = [c for c, n in ranked if n == min_count]
        tie_hist[len(tied)] += 1
        output_rank = next((i for i, (c, _) in enumerate(ranked) if c == y), -1)
        output_rank_hist[output_rank] += 1
        if not ok:
            fails.append(
                {
                    "idx": idx,
                    "shape": f"{x.shape[0]}x{x.shape[1]}",
                    "expected": y,
                    "least_pred": pred,
                    "ranked_counts": ranked,
                    "tied_min_colors": tied,
                    "output_rank": output_rank,
                    "output_count": dict(counts_nz(x)).get(y, 0),
                    "corners": [int(x[0, 0]), int(x[0, -1]), int(x[-1, 0]), int(x[-1, -1])],
                    "center": int(x[x.shape[0] // 2, x.shape[1] // 2]),
                }
            )
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "audit_complete",
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "example_count": len(examples),
        "least_nz_pass": pass_count,
        "fail_count": len(fails),
        "fails": fails,
        "tie_hist": dict(sorted(tie_hist.items())),
        "output_rank_hist": dict(sorted(output_rank_hist.items())),
        "decision": "If failures are all count ties, add a non-lookup tie-breaker; otherwise pivot.",
        "submission_decision": "no_submit: audit only",
        "leakage_risk": "low: input count audit.",
        "overfitting_risk": "medium: adding failure-specific branches can overfit.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task346で`least_nz`が`263/267`まで通るため、残り4failがtie-breakで解けるか確認する。

## 結果

- least_nz_pass: `{pass_count}/{len(examples)}`
- fail_count: `{len(fails)}`
- output_rank_hist: `{dict(sorted(output_rank_hist.items()))}`

## 判断

failがcount tie由来ならnon-lookup tie-breakerを追加する。そうでなければ1x1 aggregate laneからpivotする。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
