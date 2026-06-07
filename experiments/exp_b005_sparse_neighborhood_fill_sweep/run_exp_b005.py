from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from typing import Callable

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp_b005_sparse_neighborhood_fill_sweep"
EXP_DIR = ROOT / "experiments" / EXP_ID
TAXONOMY_PATH = ROOT / "experiments" / "exp_b004_teacher_gain_p0_structural_taxonomy" / "task_taxonomy.csv"


@dataclass(frozen=True)
class CandidateResult:
    task_id: int
    grammar: str
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
    fail_reasons: str
    lowering_plan: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def neighbor_count(x: np.ndarray, r: int, c: int, mode: str, color: int | None) -> int:
    count = 0
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)] if mode.startswith("n4") else [
        (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)
    ]
    for dr, dc in deltas:
        rr, cc = r + dr, c + dc
        if 0 <= rr < x.shape[0] and 0 <= cc < x.shape[1]:
            if color is None:
                count += int(x[rr, cc] != 0)
            else:
                count += int(x[rr, cc] == color)
    return count


def line_count(x: np.ndarray, r: int, c: int, color: int | None) -> int:
    vals = []
    vals.extend(list(x[r, :]))
    vals.extend(list(x[:, c]))
    if color is None:
        return sum(1 for v in vals if int(v) != 0)
    return sum(1 for v in vals if int(v) == color)


def changed_target_colors(examples: list[dict[str, object]]) -> set[int]:
    colors: set[int] = set()
    for ex in examples:
        x = arr(ex["input"])  # type: ignore[arg-type]
        y = arr(ex["output"])  # type: ignore[arg-type]
        if x.shape != y.shape:
            return set()
        changed = x != y
        if not np.any(changed):
            continue
        if not np.all(x[changed] == 0):
            return set()
        colors.update(int(v) for v in np.unique(y[changed]))
    return colors


def apply_candidate(x: np.ndarray, target_color: int, feature: str, threshold: int) -> tuple[np.ndarray, str]:
    out = x.copy()
    changed = 0
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            if x[r, c] != 0:
                continue
            if feature == "n4_nonzero":
                val = neighbor_count(x, r, c, "n4", None)
            elif feature == "n8_nonzero":
                val = neighbor_count(x, r, c, "n8", None)
            elif feature == "n4_target":
                val = neighbor_count(x, r, c, "n4", target_color)
            elif feature == "n8_target":
                val = neighbor_count(x, r, c, "n8", target_color)
            elif feature == "rowcol_nonzero":
                val = line_count(x, r, c, None)
            elif feature == "rowcol_target":
                val = line_count(x, r, c, target_color)
            else:
                return x.copy(), "bad_feature"
            if val >= threshold:
                out[r, c] = target_color
                changed += 1
    if changed == 0:
        return out, "no_change"
    return out, "ok"


def load_targets() -> list[int]:
    with TAXONOMY_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    targets = []
    for row in rows:
        grammar = row["suggested_grammar"]
        if "sparse background fill" in grammar or "sparse color-role fill" in grammar:
            targets.append(int(row["task_id"]))
    return targets


def train_fit_candidates(task: dict[str, object]) -> list[tuple[int, str, int]]:
    train_examples = list(task["train"])  # type: ignore[index]
    colors = changed_target_colors(train_examples)
    candidates: list[tuple[int, str, int]] = []
    for color in colors:
        for feature in ["n4_nonzero", "n8_nonzero", "n4_target", "n8_target", "rowcol_nonzero", "rowcol_target"]:
            for threshold in range(1, 9):
                ok = True
                for ex in train_examples:
                    x = arr(ex["input"])
                    y = arr(ex["output"])
                    pred, _ = apply_candidate(x, color, feature, threshold)
                    if not np.array_equal(pred, y):
                        ok = False
                        break
                if ok:
                    candidates.append((color, feature, threshold))
    return candidates


def evaluate_candidate(task_id: int, task: dict[str, object], candidate: tuple[int, str, int]) -> CandidateResult:
    color, feature, threshold = candidate
    examples = examples_for(task, -1)
    train_n = len(task["train"])  # type: ignore[index]
    test_n = len(task["test"])  # type: ignore[index]
    split_total = Counter()
    split_pass = Counter()
    fail_reasons = Counter()
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        split_total[split] += 1
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred, reason = apply_candidate(x, color, feature, threshold)
        if np.array_equal(pred, y):
            split_pass[split] += 1
        else:
            fail_reasons[reason] += 1
    total_pass = sum(split_pass.values())
    status = "full_pass" if total_pass == len(examples) else ("train_test_pass" if split_pass["train"] == split_total["train"] and split_pass["test"] == split_total["test"] else "partial")
    return CandidateResult(
        task_id=task_id,
        grammar="sparse_local_neighborhood_fill",
        candidate=f"fill0_to_{color}_if_{feature}>={threshold}",
        status=status,
        train_pass=split_pass["train"],
        train_examples=split_total["train"],
        test_pass=split_pass["test"],
        test_examples=split_total["test"],
        arc_pass=split_pass["arc-gen"],
        arc_examples=split_total["arc-gen"],
        total_pass=total_pass,
        total_examples=len(examples),
        fail_reasons=json.dumps(dict(fail_reasons.most_common(8)), ensure_ascii=False),
        lowering_plan="small Conv kernel or row/column ReduceSum mask plus one Where; target cost goal <=250",
    )


def notes_text(result: dict[str, object]) -> str:
    return f"""# {EXP_ID}

## 目的

cost<=250へ近づけるため、P0 sparse fill taskに対してlocal-neighborhood predicateをtrain fitし、全例検証する。

## 結果

- target tasks: {result["target_task_count"]}
- train-fit candidates: {result["train_fit_candidate_count"]}
- full pass hits: {result["full_pass_hit_count"]}
- train/test pass hits: {result["train_test_pass_hit_count"]}
- best partial: {result["best_partial"]}

## 解釈

このgrammarはcost<=250に落としやすいが、現時点でfull passがなければ、近傍数だけでは不足。object role、bbox-local coordinate、symmetry orbitを組み合わせる必要がある。
"""


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    target_ids = load_targets()
    rows: list[CandidateResult] = []
    train_fit_count = 0
    no_candidate_tasks: list[int] = []
    for task_id in target_ids:
        task = load_task(task_id)
        candidates = train_fit_candidates(task)
        train_fit_count += len(candidates)
        if not candidates:
            no_candidate_tasks.append(task_id)
        for candidate in candidates:
            rows.append(evaluate_candidate(task_id, task, candidate))
    rows = sorted(rows, key=lambda r: (r.status != "full_pass", r.status != "train_test_pass", -(r.total_pass / r.total_examples), r.task_id, r.candidate))
    full = [r for r in rows if r.status == "full_pass"]
    train_test = [r for r in rows if r.status == "train_test_pass"]
    best = rows[0] if rows else None
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "full_pass_found" if full else ("train_test_hit" if train_test else "no_full_hit"),
        "target_task_count": len(target_ids),
        "target_task_ids": target_ids,
        "no_candidate_tasks": no_candidate_tasks,
        "train_fit_candidate_count": train_fit_count,
        "evaluated_candidate_count": len(rows),
        "full_pass_hit_count": len(full),
        "train_test_pass_hit_count": len(train_test),
        "full_pass_hits": [asdict(r) for r in full[:20]],
        "train_test_hits": [asdict(r) for r in train_test[:20]],
        "best_partial": asdict(best) if best else None,
        "decision": "if no full hit, combine neighborhood predicates with object-role/bbox-local grammar",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: no ONNX candidate generated" if not full else "no_submit_yet: lower full pass first",
        "leakage_risk": "low: candidates are train-fit predicates, not arc-gen labels.",
        "overfitting_risk": "medium: train-fit predicates must pass all arc-gen and Kaggle delta before acceptance.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CandidateResult.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    (EXP_DIR / "notes.md").write_text(notes_text(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
