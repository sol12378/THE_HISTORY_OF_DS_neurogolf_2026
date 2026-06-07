from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp_b027_task251_closed_zero_component_rule"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 251


@dataclass(frozen=True)
class RuleEval:
    candidate: str
    status: str
    train_pass: int
    train_examples: int
    test_pass: int
    test_examples: int
    arc_pass: int
    arc_examples: int
    total_pass: int
    total_examples: int
    avg_changed_cells: float
    max_changed_cells: int
    fail_examples: str
    lowering_plan: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def closed_zero_component_fill(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    h, w = x.shape
    for comp in components(x == 0):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        neigh: set[int] = set()
        for r, c in comp:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and x[rr, cc] != 0:
                    neigh.add(int(x[rr, cc]))
        if neigh == {2}:
            for r, c in comp:
                out[r, c] = 1
    return out


def nearest_nonzero(x: np.ndarray, r: int, c: int, dr: int, dc: int) -> int | None:
    h, w = x.shape
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        if x[r, c] != 0:
            return int(x[r, c])
        r += dr
        c += dc
    return None


def row_col_boundary_fill(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    h, w = x.shape
    for r in range(h):
        for c in range(w):
            if x[r, c] != 0:
                continue
            if (
                nearest_nonzero(x, r, c, 0, -1) == 2
                and nearest_nonzero(x, r, c, 0, 1) == 2
                and nearest_nonzero(x, r, c, -1, 0) == 2
                and nearest_nonzero(x, r, c, 1, 0) == 2
            ):
                out[r, c] = 1
    return out


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def evaluate(name: str, task: dict[str, Any], pred_fn, lowering_plan: str) -> RuleEval:
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    split_total: Counter[str] = Counter()
    split_pass: Counter[str] = Counter()
    fail_examples: list[int] = []
    changed_cells: list[int] = []
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        split_total[split] += 1
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred = pred_fn(x)
        changed_cells.append(int(np.count_nonzero(x != pred)))
        if np.array_equal(pred, y):
            split_pass[split] += 1
        else:
            fail_examples.append(idx)
    total_pass = int(sum(split_pass.values()))
    status = "full_pass" if total_pass == len(examples) else ("train_test_pass" if split_pass["train"] == split_total["train"] and split_pass["test"] == split_total["test"] else "partial")
    return RuleEval(
        candidate=name,
        status=status,
        train_pass=split_pass["train"],
        train_examples=split_total["train"],
        test_pass=split_pass["test"],
        test_examples=split_total["test"],
        arc_pass=split_pass["arc-gen"],
        arc_examples=split_total["arc-gen"],
        total_pass=total_pass,
        total_examples=len(examples),
        avg_changed_cells=float(np.mean(changed_cells)),
        max_changed_cells=max(changed_cells),
        fail_examples=json.dumps(fail_examples[:30], ensure_ascii=False),
        lowering_plan=lowering_plan,
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    rows = [
        evaluate(
            "closed_zero_component_neighbor2_to_1",
            task,
            closed_zero_component_fill,
            "Do not lower by naive flood-fill unroll. Candidate lowering should compute outside-zero reachability from border using closed-form masks, row/column reductions, or small iterative kernels with strict cost gate.",
        ),
        evaluate(
            "row_col_nearest2_boundary_to_1",
            task,
            row_col_boundary_fill,
            "Cheap row/column reduction approximation; rejected because it misses components that require true connectivity.",
        ),
    ]
    rows.sort(key=lambda r: (r.status != "full_pass", -(r.total_pass / r.total_examples), r.candidate))
    full = [row for row in rows if row.status == "full_pass"]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if full else "no_full_rule",
        "task_id": TASK_ID,
        "rule_summary": "Fill every 0-component that does not touch the grid border and whose 4-neighbor boundary colors are exactly {2}; write color 1.",
        "evaluated_candidates": len(rows),
        "full_pass_hits": [asdict(row) for row in full],
        "all_rows": [asdict(row) for row in rows],
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: reference rule only; ONNX lowering pending",
        "next_lowering": "Build a cost-gated closed-component mask. Avoid full-grid flood-fill unroll; use border reachability or rectangle-specific masks if possible.",
        "leakage_risk": "low-to-medium: rule is simple and explanatory; verified on all arc-gen examples but not yet hidden LB.",
        "overfitting_risk": "medium: task-specific component rule needs ONNX validation and LB delta before adoption.",
    }
    with (EXP_DIR / "rule_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RuleEval.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`exp070` で有望に見えた task251 を、説明可能な sparse/region fill rule に圧縮できるか確認する。

## 結果

- full pass rule: `{full[0].candidate if full else 'none'}`
- validation: `{full[0].total_pass if full else 0}/{rows[0].total_examples}`
- rule: 0-component がgrid borderへ接しておらず、4-neighbor境界色が `{2}` だけなら、そのcomponentを color `1` にする。
- cheap row/column approximation: `{[row for row in rows if row.candidate == 'row_col_nearest2_boundary_to_1'][0].total_pass}/{rows[0].total_examples}`

## 判断

task251は説明可能ruleとして解けた。ただし、素朴なflood-fill unrollは過去実験で高cost化しやすい。次はclosed component maskを安くONNX化する lowering 専用実験に進む。

## Risk

- leakage risk: {result["leakage_risk"]}
- overfitting risk: {result["overfitting_risk"]}
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
