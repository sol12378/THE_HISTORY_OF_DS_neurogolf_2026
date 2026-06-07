from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp_b031_task085_horizontal_bar_alternate_erase_rule"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 85


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


def horizontal_bar_alternate_erase(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for color in [int(v) for v in np.unique(x) if int(v) != 0]:
        for comp in components(x == color):
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            r0, r1 = min(rs), max(rs) + 1
            c0, c1 = min(cs), max(cs) + 1
            if r1 - r0 != 3:
                continue
            if len(comp) != 3 * (c1 - c0):
                continue
            mid = r0 + 1
            for c in range(c0, c1):
                if (c - c0) % 2 == 1:
                    out[mid, c] = 0
    return out


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def evaluate(task: dict[str, Any]) -> RuleEval:
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    split_total = {"train": 0, "test": 0, "arc-gen": 0}
    split_pass = {"train": 0, "test": 0, "arc-gen": 0}
    changed_cells: list[int] = []
    fails: list[int] = []
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        split_total[split] += 1
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred = horizontal_bar_alternate_erase(x)
        changed_cells.append(int(np.count_nonzero(x != pred)))
        if np.array_equal(pred, y):
            split_pass[split] += 1
        else:
            fails.append(idx)
    total_pass = sum(split_pass.values())
    status = "full_pass" if total_pass == len(examples) else "partial"
    return RuleEval(
        candidate="horizontal_3row_bar_middle_alternate_erase",
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
        fail_examples=json.dumps(fails[:30], ensure_ascii=False),
        lowering_plan="Detect solid same-color 3-row horizontal components, then zero middle-row cells with odd offset from component left edge. ONNX lowering needs run-relative parity; fixed checkerboard parity is insufficient.",
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    row = evaluate(task)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if row.status == "full_pass" else "no_full_rule",
        "task_id": TASK_ID,
        "rule_summary": "For every solid same-color horizontal rectangle with height 3, zero every second cell of its middle row starting at left+1.",
        "rule_eval": asdict(row),
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: reference rule only; ONNX lowering pending",
        "next_lowering": "Profile existing artifact and try run-left parity lowering or graph surgery; avoid fixed global checkerboard parity.",
        "leakage_risk": "low-to-medium: explanatory rule verified on all arc-gen examples; no lookup table.",
        "overfitting_risk": "medium: task-specific rule needs ONNX validation and LB calibration before adoption.",
    }
    with (EXP_DIR / "rule_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RuleEval.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerow(asdict(row))
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`exp071` で object_erase_or_mask_clean と判定された task085 を説明可能ruleへ圧縮する。

## 結果

- validation: `{row.total_pass}/{row.total_examples}`
- train: `{row.train_pass}/{row.train_examples}`
- test: `{row.test_pass}/{row.test_examples}`
- arc-gen: `{row.arc_pass}/{row.arc_examples}`
- avg changed cells: `{row.avg_changed_cells:.4f}`

## Rule

同一色で構成されたheight 3のsolid horizontal rectangleを見つけ、その中央行だけ、component左端から奇数offsetのセルを0へ消す。

## Lowering Note

global checkerboard parityではなく、component left edgeからのrelative parityが必要。次は既存artifact profileかrun-left parity loweringを試す。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
