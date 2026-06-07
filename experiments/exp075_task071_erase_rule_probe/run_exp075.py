from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp075_task071_erase_rule_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 71


@dataclass(frozen=True)
class CandidateEval:
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
    mean_extra: float
    mean_missing: float
    fail_examples: str


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


def bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs) + 1, max(cs) + 1


def erase_solid_horizontal_h3_alt(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for color in [int(v) for v in np.unique(x) if int(v) != 0]:
        for comp in components(x == color):
            r0, c0, r1, c1 = bbox(comp)
            if r1 - r0 != 3:
                continue
            if len(comp) != 3 * (c1 - c0):
                continue
            for c in range(c0, c1):
                if (c - c0) % 2 == 1:
                    out[r0 + 1, c] = 0
    return out


def erase_any_solid_h3_alt(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for color in [int(v) for v in np.unique(x) if int(v) != 0]:
        for comp in components(x == color):
            r0, c0, r1, c1 = bbox(comp)
            if r1 - r0 != 3:
                continue
            if len(comp) == (r1 - r0) * (c1 - c0):
                for c in range(c0, c1):
                    if (c - c0) % 2 == 1:
                        out[r0 + 1, c] = 0
    return out


def erase_all_changed_color_components(x: np.ndarray, learned_colors: set[int]) -> np.ndarray:
    out = x.copy()
    for color in learned_colors:
        out[x == color] = 0
    return out


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def evaluate(
    task: dict[str, Any],
    name: str,
    predictor: Callable[[np.ndarray], np.ndarray],
) -> CandidateEval:
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    split_total = {"train": 0, "test": 0, "arc-gen": 0}
    split_pass = {"train": 0, "test": 0, "arc-gen": 0}
    extras: list[int] = []
    missing: list[int] = []
    fails: list[dict[str, int | str]] = []
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        split_total[split] += 1
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred = predictor(x)
        if np.array_equal(pred, y):
            split_pass[split] += 1
            extras.append(0)
            missing.append(0)
            continue
        extra = int(np.count_nonzero((pred != x) & (y == x)))
        miss = int(np.count_nonzero((pred == x) & (y != x)))
        extras.append(extra)
        missing.append(miss)
        if len(fails) < 30:
            fails.append({"idx": idx, "split": split, "extra": extra, "missing": miss})
    total_pass = sum(split_pass.values())
    return CandidateEval(
        candidate=name,
        status="full_pass" if total_pass == len(examples) else "partial",
        train_pass=split_pass["train"],
        train_examples=split_total["train"],
        test_pass=split_pass["test"],
        test_examples=split_total["test"],
        arc_pass=split_pass["arc-gen"],
        arc_examples=split_total["arc-gen"],
        total_pass=total_pass,
        total_examples=len(examples),
        mean_extra=float(np.mean(extras)),
        mean_missing=float(np.mean(missing)),
        fail_examples=json.dumps(fails, ensure_ascii=False),
    )


def audit_changed_cells(task: dict[str, Any]) -> dict[str, Any]:
    examples = examples_for(task, -1)
    changed_to_zero = 0
    changed_from_zero = 0
    color_hist = Counter()
    out_hist = Counter()
    changed_counts = Counter()
    comp_touched = Counter()
    for ex in examples:
        x = arr(ex["input"])
        y = arr(ex["output"])
        diff = x != y
        coords = [(int(r), int(c)) for r, c in np.argwhere(diff)]
        changed_counts[len(coords)] += 1
        if coords and all(int(y[r, c]) == 0 for r, c in coords):
            changed_to_zero += 1
        if coords and all(int(x[r, c]) == 0 for r, c in coords):
            changed_from_zero += 1
        color_hist.update(int(x[r, c]) for r, c in coords)
        out_hist.update(int(y[r, c]) for r, c in coords)
        touched = 0
        for color in [int(v) for v in np.unique(x) if int(v) != 0]:
            for comp in components(x == color):
                if any(diff[r, c] for r, c in comp):
                    touched += 1
        comp_touched[touched] += 1
    return {
        "examples": len(examples),
        "changed_to_zero_examples": changed_to_zero,
        "changed_from_zero_examples": changed_from_zero,
        "changed_count_hist": dict(changed_counts.most_common()),
        "input_changed_color_hist": dict(color_hist.most_common()),
        "output_changed_color_hist": dict(out_hist.most_common()),
        "touched_component_count_hist": dict(comp_touched.most_common()),
    }


def learned_train_changed_colors(task: dict[str, Any]) -> set[int]:
    colors: set[int] = set()
    for ex in task["train"]:
        x = arr(ex["input"])
        y = arr(ex["output"])
        for r, c in np.argwhere(x != y):
            if int(y[r, c]) == 0:
                colors.add(int(x[r, c]))
    return colors


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    train_colors = learned_train_changed_colors(task)
    rows = [
        evaluate(task, "task085_horizontal_3row_bar_middle_alternate_erase", erase_solid_horizontal_h3_alt),
        evaluate(task, "any_solid_h3_middle_alternate_erase", erase_any_solid_h3_alt),
        evaluate(
            task,
            "train_changed_colors_erase_components",
            lambda x: erase_all_changed_color_components(x, train_colors),
        ),
    ]
    audit = audit_changed_cells(task)
    best = max(rows, key=lambda r: r.total_pass)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "rule_found" if best.status == "full_pass" else "no_simple_erase_rule",
        "hypothesis": "task071がtask085型のsolid-bar eraseとして低cost rule化できるかを検証する。",
        "audit": audit,
        "train_changed_colors": sorted(train_colors),
        "candidate_evals": [asdict(r) for r in rows],
        "best_candidate": asdict(best),
        "decision": "task071はtask085型の単純eraseではない。changed-to-zeroではなくrecolor/copyを含むmixed_component_editとして扱う。",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: no full-pass replacement candidate",
        "next_action": "task071はrecolor/copyまたはcomponent branch-treeへ回し、score直結の次候補は既知ruleのtask085 loweringかtask251 closed-component loweringに戻す。",
        "leakage_risk": "low: train-derived raw color eraseを候補に含めたが提出物は作らず、全arc-genで否定する診断のみ。",
        "overfitting_risk": "low: no candidate accepted.",
    }
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CandidateEval.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`task071` が `task085` と同じ horizontal-bar erase 型なら、既知ruleを横展開して低cost置換候補にできる。

## 結果

- best: `{best.candidate}` = `{best.total_pass}/{best.total_examples}`
- train: `{best.train_pass}/{best.train_examples}`
- test: `{best.test_pass}/{best.test_examples}`
- arc-gen: `{best.arc_pass}/{best.arc_examples}`
- changed_to_zero_examples: `{audit["changed_to_zero_examples"]}/{audit["examples"]}`
- changed_from_zero_examples: `{audit["changed_from_zero_examples"]}/{audit["examples"]}`

## 解釈

`task071` は `task085` 型の単純eraseではない。変更セルは mixed component edit で、0への消去だけでは説明できない。
このため、`task071` をすぐ ONNX lowering するのは避け、recolor/copy compiler または小さな component branch-tree の探索へ回す。

## Risk

- leakage risk: low。診断のみで、lookup提出物は生成していない。
- overfitting risk: low。full-pass候補なし。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
