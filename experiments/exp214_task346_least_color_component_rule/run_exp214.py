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

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp214_task346_least_color_component_rule"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 346


def comps(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        cells = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        out.append(cells)
    return out


def largest_component_size(x: np.ndarray, color: int) -> int:
    cs = comps(x == color)
    return max((len(c) for c in cs), default=0)


def predict(x: np.ndarray) -> int:
    counts = Counter(int(v) for v in x.ravel() if int(v) != 0)
    ranked = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    if not ranked:
        return 0
    rank0 = ranked[0][0]
    if len(ranked) > 1 and largest_component_size(x, rank0) >= 8:
        return ranked[1][0]
    return rank0


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    base = load_base_tasks()[TASK_ID]
    pass_count = 0
    fails: list[dict[str, Any]] = []
    branch_hist = Counter()
    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = int(grid_to_array(ex["output"])[0, 0])
        pred = predict(x)
        ok = pred == y
        pass_count += int(ok)
        counts = Counter(int(v) for v in x.ravel() if int(v) != 0)
        ranked = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
        rank0 = ranked[0][0]
        branch = "rank1" if len(ranked) > 1 and largest_component_size(x, rank0) >= 8 else "rank0"
        branch_hist[branch] += 1
        if not ok and len(fails) < 20:
            fails.append({"idx": idx, "pred": pred, "expected": y, "ranked": ranked, "rank0_largest": largest_component_size(x, rank0)})
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if pass_count == len(examples) else "partial",
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "example_count": len(examples),
        "pass_count": pass_count,
        "fail_count": len(examples) - pass_count,
        "branch_hist": dict(sorted(branch_hist.items())),
        "fails": fails,
        "rule": "choose least nonzero color by count; if that color's largest 4-connected component has size >= 8, choose second-least color.",
        "decision": "Rule is full-pass but ONNX lowering needs color counts and connected-component largest-size proxy; likely not immediate tiny lowering.",
        "submission_decision": "no_submit: Python rule only",
        "leakage_risk": "low: input-only structural rule.",
        "overfitting_risk": "medium: threshold 8 inferred from four switch cases; all-arc pass but hidden robustness uncertain.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task346の1x1 outputについて、least-color近似の4failをcomponent特徴で補正したruleをfull validationする。

## 結果

- pass: `{pass_count}/{len(examples)}`
- branch_hist: `{dict(sorted(branch_hist.items()))}`

## 判断

Python ruleはfull-pass。ただしONNX化には色countとlargest connected component proxyが必要で、即tiny loweringではない。

## リスク

- leakage risk: low。
- overfitting risk: medium。threshold 8 は4 switch例から得たためhiddenで要注意。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
