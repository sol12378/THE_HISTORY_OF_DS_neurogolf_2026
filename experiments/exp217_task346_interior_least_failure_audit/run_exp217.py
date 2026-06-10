from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter, deque
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_task  # noqa: E402


EXP_ID = "exp217_task346_interior_least_failure_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 346


def counts(x: np.ndarray, include_zero: bool = False) -> Counter[int]:
    return Counter(int(v) for v in x.ravel() if include_zero or int(v) != 0)


def ordered_least(x: np.ndarray, include_zero: bool = False) -> list[tuple[int, int]]:
    c = counts(x, include_zero)
    return sorted(((int(k), int(v)) for k, v in c.items()), key=lambda kv: (kv[1], kv[0]))


def largest_component(x: np.ndarray, color: int) -> int:
    mask = x == color
    seen = np.zeros(mask.shape, dtype=bool)
    best = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        size = 0
        while q:
            rr, cc = q.popleft()
            size += 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        best = max(best, size)
    return int(best)


def interior(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 2 or x.shape[1] <= 2:
        return np.array([], dtype=x.dtype)
    return x[1:-1, 1:-1]


def edge(x: np.ndarray) -> np.ndarray:
    if min(x.shape) == 1:
        return x.ravel()
    return np.concatenate([x[0, :], x[-1, :], x[1:-1, 0], x[1:-1, -1]])


def corners(x: np.ndarray) -> list[int]:
    return [int(x[0, 0]), int(x[0, -1]), int(x[-1, 0]), int(x[-1, -1])]


def pred_interior_least(x: np.ndarray) -> int:
    ranked = ordered_least(interior(x), include_zero=False)
    return int(ranked[0][0]) if ranked else 0


def color_features(x: np.ndarray, color: int) -> dict[str, Any]:
    inner = interior(x)
    ed = edge(x)
    return {
        "color": int(color),
        "all_count": int(np.sum(x == color)),
        "interior_count": int(np.sum(inner == color)),
        "edge_count": int(np.sum(ed == color)),
        "corner_count": int(sum(v == color for v in corners(x))),
        "largest_component": largest_component(x, color),
    }


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]

    failures: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = int(grid_to_array(ex["output"])[0, 0])
        pred = pred_interior_least(x)
        ranked_inner = ordered_least(interior(x), include_zero=False)
        ranked_all = ordered_least(x, include_zero=False)
        colors = sorted({c for c, _ in ranked_inner[:4]} | {c for c, _ in ranked_all[:4]} | {y, pred})
        row = {
            "idx": idx,
            "expected": y,
            "pred": pred,
            "ok": pred == y,
            "shape": list(x.shape),
            "corners": corners(x),
            "inner_rank": ranked_inner[:6],
            "all_rank": ranked_all[:6],
            "edge_rank": ordered_least(edge(x), include_zero=False)[:6],
            "color_features": [color_features(x, c) for c in colors],
        }
        if pred == y:
            pass_rows.append(row)
        else:
            failures.append(row)

    # Candidate simple branch predicates that avoid full-grid component growth.
    feature_names = ["all_count", "interior_count", "edge_count", "corner_count"]
    branch_tests: list[dict[str, Any]] = []
    for feat in feature_names:
        for op in ["<=", ">=", "=="]:
            values = sorted({cf[feat] for row in failures + pass_rows for cf in row["color_features"]})
            for threshold in values:
                tp = fp = fn = 0
                for row in failures + pass_rows:
                    pred_cf = next(cf for cf in row["color_features"] if cf["color"] == row["pred"])
                    value = pred_cf[feat]
                    hit = value <= threshold if op == "<=" else value >= threshold if op == ">=" else value == threshold
                    if hit and not row["ok"]:
                        tp += 1
                    elif hit and row["ok"]:
                        fp += 1
                    elif (not hit) and not row["ok"]:
                        fn += 1
                if tp:
                    branch_tests.append({"feature": feat, "op": op, "threshold": int(threshold), "tp": tp, "fp": fp, "fn": fn})
    branch_tests.sort(key=lambda r: (-r["tp"], r["fp"], r["fn"], r["feature"], r["op"], r["threshold"]))

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "base_rule": "interior_least_nz",
        "pass_count": len(pass_rows),
        "fail_count": len(failures),
        "failures": failures,
        "best_simple_branch_tests": branch_tests[:20],
        "decision": "2 failのみ。count/edge/cornerだけでTP2 FP0の補正があればcost probeへ、なければcomponent proxy依存で保留。",
        "submission_decision": "no_submit: diagnostic only",
        "leakage_risk": "low: input-only diagnostic.",
        "overfitting_risk": "medium: 2例補正はarc-gen全体に対する分岐でもhidden generalization注意。",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task346の `interior_least_nz` が失敗する2例を監査し、component growthなしの安い補正で267/267へ到達できるかを判断する。

## 結果

- pass/fail: `{len(pass_rows)}/{len(failures)}`
- failures: `{[(r['idx'], r['expected'], r['pred'], r['inner_rank'][:3], r['edge_rank'][:3], r['corners']) for r in failures]}`
- best_simple_branch_tests: `{branch_tests[:5]}`

## 判断

TP2 FP0の単純count系分岐があれば次はONNX cost probe。なければexp214同様component条件が必要で、exp215のcost wallから一旦保留。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
