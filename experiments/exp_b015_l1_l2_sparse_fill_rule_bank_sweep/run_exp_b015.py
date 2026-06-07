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

from experiments.exp_b010_component_object_role_sparse_fill_miner.run_exp_b010 import (  # noqa: E402
    RuleFn,
    arr,
    changed_count,
    rule_bank,
    split_name,
)
from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp_b015_l1_l2_sparse_fill_rule_bank_sweep"
EXP_DIR = ROOT / "experiments" / EXP_ID
TAXONOMY_PATH = ROOT / "experiments" / "exp054_signature_lookup_family_taxonomy" / "task_taxonomy.csv"
FAST_TOP_K = 20


@dataclass(frozen=True)
class CandidateResult:
    task_id: int
    compiler_lane: str
    strict_cost: int
    gain_to_600: float
    gain_to_250: float
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
    pass_rate: float
    avg_pred_changed_cells: float
    fail_reasons: str
    lowering_plan: str


def load_targets() -> dict[int, dict[str, str]]:
    with TAXONOMY_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lane_rows = [
        row
        for row in rows
        if row["compiler_lane"] in {"L1_static_sparse_background_fill", "L2_local_predicate_sparse_fill"}
    ]
    lane_rows = sorted(lane_rows, key=lambda row: (-float(row["gain_to_250"]), int(row["task_id"])))
    targets = {}
    for row in lane_rows[:FAST_TOP_K]:
        targets[int(row["task_id"])] = row
    return targets


def evaluate(task_id: int, meta: dict[str, str], task: dict[str, Any], name: str, rule: RuleFn, lowering_plan: str) -> CandidateResult:
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    split_total = Counter()
    split_pass = Counter()
    fail_reasons = Counter()
    changed_cells = []
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        split_total[split] += 1
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred, reason = rule(x)
        if np.array_equal(pred, y):
            split_pass[split] += 1
        else:
            fail_reasons[reason] += 1
        changed_cells.append(changed_count(x, pred))
    total_pass = sum(split_pass.values())
    status = "full_pass" if total_pass == len(examples) else ("train_test_pass" if split_pass["train"] == split_total["train"] and split_pass["test"] == split_total["test"] else ("train_fit" if split_pass["train"] == split_total["train"] else "partial"))
    return CandidateResult(
        task_id=task_id,
        compiler_lane=meta["compiler_lane"],
        strict_cost=int(float(meta["strict_cost"])),
        gain_to_600=float(meta["gain_to_600"]),
        gain_to_250=float(meta["gain_to_250"]),
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
        pass_rate=total_pass / max(1, len(examples)),
        avg_pred_changed_cells=float(np.mean([v for v in changed_cells if v >= 0])) if changed_cells else 0.0,
        fail_reasons=json.dumps(dict(fail_reasons.most_common(8)), ensure_ascii=False),
        lowering_plan=lowering_plan,
    )


def notes_text(result: dict[str, Any]) -> str:
    return f"""# {EXP_ID}

## 目的

exp_b010でtask037 full passを出したcomponent/object-role sparse fill rule bankを、signature lookup taxonomyのL1/L2 lane全体へ横展開する。

## 結果

- target tasks: {result["target_task_count"]}
- rule count: {result["rule_count"]}
- evaluated candidates: {result["evaluated_candidate_count"]}
- full pass hits: {result["full_pass_hit_count"]}
- train/test hits: {result["train_test_pass_hit_count"]}
- train-fit hits: {result["train_fit_hit_count"]}
- best candidates: {result["best_candidates"][:5]}

## 判断

full passがあればloweringへ進む。full passが少ない場合でも、partial上位をdecision tree合成の素材として使う。
"""


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    rules = rule_bank()
    rows: list[CandidateResult] = []
    for task_id, meta in sorted(targets.items()):
        task = load_task(task_id)
        for name, rule, lowering_plan in rules:
            rows.append(evaluate(task_id, meta, task, name, rule, lowering_plan))
    rows = sorted(
        rows,
        key=lambda r: (
            r.status != "full_pass",
            r.status != "train_test_pass",
            r.status != "train_fit",
            -r.pass_rate,
            -r.gain_to_250,
            r.avg_pred_changed_cells,
            r.task_id,
            r.candidate,
        ),
    )
    full = [r for r in rows if r.status == "full_pass"]
    train_test = [r for r in rows if r.status == "train_test_pass"]
    train_fit = [r for r in rows if r.status == "train_fit"]
    best_by_task = {}
    for row in rows:
        best_by_task.setdefault(row.task_id, row)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "full_pass_found" if full else ("train_test_hit" if train_test else ("train_fit_hit" if train_fit else "no_full_hit")),
        "target_task_count": len(targets),
        "fast_top_k": FAST_TOP_K,
        "target_task_ids": sorted(targets),
        "lane_counts": dict(Counter(meta["compiler_lane"] for meta in targets.values())),
        "rule_count": len(rules),
        "evaluated_candidate_count": len(rows),
        "full_pass_hit_count": len(full),
        "train_test_pass_hit_count": len(train_test),
        "train_fit_hit_count": len(train_fit),
        "full_pass_hits": [asdict(r) for r in full[:50]],
        "train_test_hits": [asdict(r) for r in train_test[:50]],
        "train_fit_hits": [asdict(r) for r in train_fit[:50]],
        "best_candidates": [asdict(r) for r in list(best_by_task.values())[:50]],
        "candidate_status_counts": dict(Counter(r.status for r in rows)),
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: rule discovery only; lower full pass first",
        "decision": "lower full pass hits; use high-partial rows for task-specific decision tree synthesis",
        "leakage_risk": "low-to-medium: explanatory rules; arc-gen used only for evaluation.",
        "overfitting_risk": "medium: train-derived selection must be validated by full arc-gen and Kaggle delta before adoption.",
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
