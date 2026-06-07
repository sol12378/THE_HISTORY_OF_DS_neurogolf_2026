from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp_b017_l2_changed_cell_feature_profile.run_exp_b017 import (  # noqa: E402
    FAST_TOP_K,
    cell_features,
    load_targets,
)
from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp_b018_high_precision_island_or_tree_miner"
EXP_DIR = ROOT / "experiments" / EXP_ID
MAX_BRANCHES = 80
MIN_PRECISION = 0.999


@dataclass(frozen=True)
class CandidateResult:
    task_id: int
    compiler_lane: str
    strict_cost: int
    gain_to_250: float
    status: str
    branch_count: int
    train_positive_cells: int
    train_negative_cells: int
    train_covered_positive: int
    train_false_positive: int
    train_pass: int
    train_examples: int
    test_pass: int
    test_examples: int
    arc_pass: int
    arc_examples: int
    total_pass: int
    total_examples: int
    branches: str
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


def train_cell_rows(task: dict[str, Any], task_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ex_idx, ex in enumerate(task["train"]):
        x = arr(ex["input"])
        y = arr(ex["output"])
        if x.shape != y.shape:
            continue
        changed = (x != y) & (x == 0)
        out_colors = [int(v) for v in np.unique(y[changed])] if np.any(changed) else []
        candidate_colors = out_colors or [int(v) for v in np.unique(y) if int(v) != 0]
        for r, c in np.argwhere(x == 0):
            for out_color in candidate_colors:
                label = int(changed[int(r), int(c)] and int(y[int(r), int(c)]) == out_color)
                feats = cell_features(x, int(r), int(c), out_color)
                rows.append({"ex_idx": ex_idx, "r": int(r), "c": int(c), "color": out_color, "label": label, "features": feats})
    return rows


def mine_branches(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pos_ids = {i for i, row in enumerate(rows) if row["label"] == 1}
    neg_ids = {i for i, row in enumerate(rows) if row["label"] == 0}
    value_hits: dict[tuple[str, str], set[int]] = {}
    for i, row in enumerate(rows):
        for feat, val in row["features"].items():
            value_hits.setdefault((feat, val), set()).add(i)
    candidates = []
    for key, ids in value_hits.items():
        pc = len(ids & pos_ids)
        nc = len(ids & neg_ids)
        if pc == 0:
            continue
        precision = pc / max(1, pc + nc)
        if precision >= MIN_PRECISION:
            candidates.append((key, pc, nc, precision))
    covered: set[int] = set()
    branches: list[tuple[str, str]] = []
    for key, _pc, _nc, _precision in sorted(candidates, key=lambda x: (-len(value_hits[x[0]] & (pos_ids - covered)), x[2], x[0])):
        gain = len(value_hits[key] & (pos_ids - covered))
        if gain <= 0:
            continue
        branches.append(key)
        covered |= value_hits[key] & pos_ids
        if len(branches) >= MAX_BRANCHES or covered == pos_ids:
            break
    return branches


def apply_tree(x: np.ndarray, branches: list[tuple[str, str]], candidate_colors: list[int]) -> tuple[np.ndarray, str]:
    out = x.copy()
    for r, c in np.argwhere(x == 0):
        votes = []
        for color in candidate_colors:
            feats = cell_features(x, int(r), int(c), color)
            if any(feats.get(feat, "") == val for feat, val in branches):
                votes.append(color)
        if len(votes) == 1:
            out[int(r), int(c)] = votes[0]
        elif len(votes) > 1:
            return x.copy(), "color_conflict"
    return out, "ok"


def eval_tree(task_id: int, meta: dict[str, str], task: dict[str, Any], branches: list[tuple[str, str]], train_rows: list[dict[str, Any]]) -> CandidateResult:
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    split_total = Counter()
    split_pass = Counter()
    fail_reasons = Counter()
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        split_total[split] += 1
        x = arr(ex["input"])
        y = arr(ex["output"])
        if x.shape != y.shape:
            fail_reasons["shape_mismatch"] += 1
            continue
        changed = (x != y) & (x == 0)
        candidate_colors = [int(v) for v in np.unique(y[changed])] if np.any(changed) else [int(v) for v in np.unique(y) if int(v) != 0]
        pred, reason = apply_tree(x, branches, candidate_colors)
        if np.array_equal(pred, y):
            split_pass[split] += 1
        else:
            fail_reasons[reason] += 1
    pos = [r for r in train_rows if r["label"] == 1]
    neg = [r for r in train_rows if r["label"] == 0]
    covered_pos = sum(1 for row in pos if any(row["features"].get(f, "") == v for f, v in branches))
    false_pos = sum(1 for row in neg if any(row["features"].get(f, "") == v for f, v in branches))
    total_pass = sum(split_pass.values())
    status = "full_pass" if total_pass == len(examples) else ("train_test_pass" if split_pass["train"] == split_total["train"] and split_pass["test"] == split_total["test"] else ("train_fit" if split_pass["train"] == split_total["train"] else "partial"))
    return CandidateResult(
        task_id=task_id,
        compiler_lane=meta["compiler_lane"],
        strict_cost=int(float(meta["strict_cost"])),
        gain_to_250=float(meta["gain_to_250"]),
        status=status,
        branch_count=len(branches),
        train_positive_cells=len(pos),
        train_negative_cells=len(neg),
        train_covered_positive=covered_pos,
        train_false_positive=false_pos,
        train_pass=split_pass["train"],
        train_examples=split_total["train"],
        test_pass=split_pass["test"],
        test_examples=split_total["test"],
        arc_pass=split_pass["arc-gen"],
        arc_examples=split_total["arc-gen"],
        total_pass=total_pass,
        total_examples=len(examples),
        branches=json.dumps(branches, ensure_ascii=False),
        fail_reasons=json.dumps(dict(fail_reasons.most_common(8)), ensure_ascii=False),
        lowering_plan="OR of ray/bbox/neighborhood feature-value masks; lower only if branch_count remains small",
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    rows: list[CandidateResult] = []
    for task_id, meta in targets.items():
        task = load_task(task_id)
        train_rows = train_cell_rows(task, task_id)
        branches = mine_branches(train_rows)
        if branches:
            rows.append(eval_tree(task_id, meta, task, branches, train_rows))
    rows = sorted(rows, key=lambda r: (r.status != "full_pass", r.status != "train_test_pass", r.status != "train_fit", -r.train_covered_positive, r.train_false_positive, r.branch_count, r.task_id))
    full = [r for r in rows if r.status == "full_pass"]
    train_test = [r for r in rows if r.status == "train_test_pass"]
    train_fit = [r for r in rows if r.status == "train_fit"]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "full_pass_found" if full else ("train_test_hit" if train_test else ("train_fit_hit" if train_fit else "no_full_hit")),
        "target_task_count": len(targets),
        "fast_top_k": FAST_TOP_K,
        "target_task_ids": sorted(targets),
        "max_branches": MAX_BRANCHES,
        "min_precision": MIN_PRECISION,
        "evaluated_candidate_count": len(rows),
        "full_pass_hit_count": len(full),
        "train_test_pass_hit_count": len(train_test),
        "train_fit_hit_count": len(train_fit),
        "full_pass_hits": [asdict(r) for r in full[:20]],
        "train_test_hits": [asdict(r) for r in train_test[:20]],
        "train_fit_hits": [asdict(r) for r in train_fit[:20]],
        "candidate_rows": [asdict(r) for r in rows],
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: tree discovery only; lower full pass first",
        "decision": "if train-fit exists, reduce branches and validate; otherwise add pairwise feature conjunctions",
        "leakage_risk": "medium: branch values are train-derived; no arc-gen used for mining.",
        "overfitting_risk": "high unless full arc-gen and Kaggle delta pass.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CandidateResult.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    notes = f"""# {EXP_ID}

## 目的

exp_b017で見つかったhigh precision feature islandをOR treeとして足し、train full fitする説明可能branch ruleを探索する。

## 結果

- target tasks: {len(targets)}
- evaluated candidates: {len(rows)}
- full pass hits: {len(full)}
- train/test hits: {len(train_test)}
- train-fit hits: {len(train_fit)}
- rows: {[asdict(r) for r in rows[:3]]}

## 判断

train-fitがあればbranch削減とfull loweringへ進む。なければpairwise feature conjunctionが必要。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
