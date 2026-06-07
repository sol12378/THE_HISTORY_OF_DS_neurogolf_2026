from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from itertools import combinations
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp057_l1_object_role_feature_miner"
EXP_DIR = ROOT / "experiments" / EXP_ID
TAXONOMY = ROOT / "experiments" / "exp054_signature_lookup_family_taxonomy" / "task_taxonomy.csv"


@dataclass(frozen=True)
class RuleEval:
    task_id: int
    rule_name: str
    status: str
    total_pass: int
    total_examples: int
    train_pass: int
    train_examples: int
    test_pass: int
    test_examples: int
    arc_pass: int
    arc_examples: int
    predicted_change_median: float
    fail_reasons: str
    rule_spec: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def load_l1_tasks() -> list[int]:
    return [20]


def rel_features(x: np.ndarray, r: int, c: int, color: int) -> dict[str, bool]:
    h, w = x.shape
    b_all = bbox(x != 0)
    b_col = bbox(x == color)
    feats: dict[str, bool] = {}
    feats["is_zero"] = x[r, c] == 0
    feats["row_has_color"] = bool(np.any(x[r, :] == color))
    feats["col_has_color"] = bool(np.any(x[:, c] == color))
    feats["row_has_any"] = bool(np.any(x[r, :] != 0))
    feats["col_has_any"] = bool(np.any(x[:, c] != 0))
    feats["row_count_color_1"] = int(np.sum(x[r, :] == color)) == 1
    feats["row_count_color_ge2"] = int(np.sum(x[r, :] == color)) >= 2
    feats["col_count_color_1"] = int(np.sum(x[:, c] == color)) == 1
    feats["col_count_color_ge2"] = int(np.sum(x[:, c] == color)) >= 2
    feats["on_main_diag"] = r == c
    feats["on_anti_diag"] = r + c == h - 1
    feats["border"] = r in {0, h - 1} or c in {0, w - 1}
    feats["interior"] = not feats["border"]

    for name, b in [("allbbox", b_all), ("colorbbox", b_col)]:
        if b is None:
            for key in ["inside", "border", "corner", "center_row", "center_col"]:
                feats[f"{name}_{key}"] = False
            continue
        r0, c0, r1, c1 = b
        inside = r0 <= r < r1 and c0 <= c < c1
        feats[f"{name}_inside"] = inside
        feats[f"{name}_border"] = inside and (r in {r0, r1 - 1} or c in {c0, c1 - 1})
        feats[f"{name}_corner"] = inside and r in {r0, r1 - 1} and c in {c0, c1 - 1}
        feats[f"{name}_center_row"] = inside and (2 * (r - r0) == (r1 - r0 - 1))
        feats[f"{name}_center_col"] = inside and (2 * (c - c0) == (c1 - c0 - 1))
        feats[f"{name}_same_rel_row_as_any"] = inside and bool(np.any((x[r, c0:c1] == color)))
        feats[f"{name}_same_rel_col_as_any"] = inside and bool(np.any((x[r0:r1, c] == color)))

    # Local neighborhood roles.
    for radius in [1, 2]:
        r0, r1 = max(0, r - radius), min(h, r + radius + 1)
        c0, c1 = max(0, c - radius), min(w, c + radius + 1)
        win = x[r0:r1, c0:c1]
        cnt = int(np.sum(win == color))
        anycnt = int(np.sum(win != 0))
        feats[f"n{radius}_has_color"] = cnt > 0
        feats[f"n{radius}_color_ge2"] = cnt >= 2
        feats[f"n{radius}_color_ge3"] = cnt >= 3
        feats[f"n{radius}_any_ge2"] = anycnt >= 2
        feats[f"n{radius}_any_ge3"] = anycnt >= 3
    return feats


def truth_masks(examples: list[dict[str, Any]]) -> list[tuple[np.ndarray, dict[int, np.ndarray]]]:
    out = []
    for ex in examples:
        x = arr(ex["input"])
        y = arr(ex["output"])
        changed = x != y
        by_color: dict[int, np.ndarray] = {}
        if x.shape == y.shape:
            for color in [int(v) for v in np.unique(y[changed]) if int(v) != 0]:
                by_color[color] = changed & (y == color) & (x == 0)
        out.append((x, by_color))
    return out


def eval_feature_set_on_truth(
    truth: list[tuple[np.ndarray, dict[int, np.ndarray]]], feature_set: tuple[str, ...]
) -> tuple[int, int, int]:
    missing_total = 0
    extra_total = 0
    predicted_total = 0
    for x, by_color in truth:
        claim_count = np.zeros(x.shape, dtype=np.int64)
        for color in by_color:
            for r, c in np.argwhere(x == 0):
                feats = rel_features(x, int(r), int(c), color)
                if all(feats.get(f, False) for f in feature_set):
                    claim_count[int(r), int(c)] += 1
        truth_mask = np.zeros(x.shape, dtype=bool)
        for mask in by_color.values():
            truth_mask |= mask
        pred_mask = claim_count == 1
        missing_total += int(np.logical_and(truth_mask, ~pred_mask).sum())
        extra_total += int(np.logical_and(pred_mask, ~truth_mask).sum())
        predicted_total += int(pred_mask.sum())
    return missing_total, extra_total, predicted_total


def candidate_feature_sets(train_truth: list[tuple[np.ndarray, dict[int, np.ndarray]]]) -> list[tuple[str, ...]]:
    positive_sets: list[set[str]] = []
    negative_sets: list[set[str]] = []
    for x, by_color in train_truth:
        for color, mask in by_color.items():
            positives = np.argwhere(mask)
            for r, c in positives:
                feats = rel_features(x, int(r), int(c), color)
                positive_sets.append({k for k, v in feats.items() if v})
            for r, c in np.argwhere(x == 0):
                if mask[int(r), int(c)]:
                    continue
                feats = rel_features(x, int(r), int(c), color)
                negative_sets.append({k for k, v in feats.items() if v})
    if not positive_sets:
        return []
    common = set.intersection(*positive_sets)
    # A feature combination is useful only if it excludes at least one train negative.
    singles = [(f,) for f in sorted(common)]
    pairs = [tuple(sorted(p)) for p in combinations(sorted(common), 2)]
    triples = [tuple(sorted(p)) for p in combinations(sorted(common), 3)]
    candidates = singles + pairs + triples
    scored = []
    for cand in candidates:
        pos_ok = sum(1 for s in positive_sets if all(f in s for f in cand))
        neg_hit = sum(1 for s in negative_sets if all(f in s for f in cand))
        if pos_ok == len(positive_sets):
            missing, extra, predicted = eval_feature_set_on_truth(train_truth, cand)
            scored.append((missing, extra, abs(predicted - len(positive_sets)), neg_hit, len(cand), cand))
    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
    return [cand for *_, cand in scored[:10]]


def apply_feature_rule(x: np.ndarray, feature_set: tuple[str, ...]) -> np.ndarray:
    out = x.copy()
    claim_count = np.zeros(x.shape, dtype=np.int64)
    claim_color = np.zeros(x.shape, dtype=np.int64)
    for color in [int(v) for v in np.unique(x) if int(v) != 0]:
        for r, c in np.argwhere(x == 0):
            feats = rel_features(x, int(r), int(c), color)
            if all(feats.get(f, False) for f in feature_set):
                claim_count[int(r), int(c)] += 1
                claim_color[int(r), int(c)] = color
    out[claim_count == 1] = claim_color[claim_count == 1]
    return out


def evaluate(task_id: int, feature_set: tuple[str, ...]) -> RuleEval:
    task = load_task(task_id)
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    counts = Counter()
    passes = Counter()
    fail_reasons = Counter()
    pred_changes = []
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred = apply_feature_rule(x, feature_set)
        counts[split] += 1
        pred_changes.append(int(np.sum(pred != x)))
        if np.array_equal(pred, y):
            passes[split] += 1
        else:
            if pred.shape != y.shape:
                fail_reasons["shape"] += 1
            else:
                missing = int(np.logical_and(pred == x, y != x).sum())
                extra = int(np.logical_and(pred != x, y == x).sum())
                wrong = int(np.logical_and(pred != y, pred != x, y != x).sum())
                fail_reasons[f"missing={missing};extra={extra};wrong={wrong}"] += 1
    total_pass = sum(passes.values())
    total_examples = len(examples)
    status = "full_pass" if total_pass == total_examples else ("train_test_pass" if passes["train"] == counts["train"] and passes["test"] == counts["test"] else "partial")
    return RuleEval(
        task_id=task_id,
        rule_name="and_features",
        status=status,
        total_pass=total_pass,
        total_examples=total_examples,
        train_pass=passes["train"],
        train_examples=counts["train"],
        test_pass=passes["test"],
        test_examples=counts["test"],
        arc_pass=passes["arc-gen"],
        arc_examples=counts["arc-gen"],
        predicted_change_median=float(np.median(pred_changes)) if pred_changes else 0.0,
        fail_reasons=json.dumps(dict(fail_reasons.most_common(5)), ensure_ascii=False),
        rule_spec=json.dumps({"all_features": feature_set}, ensure_ascii=False),
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_l1_tasks()
    rows: list[RuleEval] = []
    feature_count_by_task: dict[int, int] = {}
    for task_id in tasks:
        task = load_task(task_id)
        train_examples = examples_for(task, len(task["train"]))
        candidates = candidate_feature_sets(truth_masks(train_examples))
        feature_count_by_task[task_id] = len(candidates)
        for feature_set in candidates:
            row = evaluate(task_id, feature_set)
            if row.train_pass == row.train_examples or row.total_pass > 0:
                rows.append(row)

    rows.sort(
        key=lambda r: (
            r.status == "full_pass",
            r.status == "train_test_pass",
            r.total_pass / max(1, r.total_examples),
            r.train_pass / max(1, r.train_examples),
            -len(json.loads(r.rule_spec)["all_features"]),
        ),
        reverse=True,
    )
    full_hits = [r for r in rows if r.status == "full_pass"]
    train_test_hits = [r for r in rows if r.status in {"full_pass", "train_test_pass"}]
    by_task_best = {}
    for row in rows:
        by_task_best.setdefault(row.task_id, asdict(row))

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if full_hits else "no_full_hit",
        "source": str(TAXONOMY.relative_to(ROOT)),
        "scanned_task_count": len(tasks),
        "tasks": tasks,
        "candidate_count_by_task": feature_count_by_task,
        "evaluated_candidate_count": len(rows),
        "full_pass_hit_count": len(full_hits),
        "train_test_pass_hit_count": len(train_test_hits),
        "full_pass_hits": [asdict(r) for r in full_hits[:20]],
        "train_test_hits": [asdict(r) for r in train_test_hits[:20]],
        "best_by_task": by_task_best,
        "best_overall": asdict(rows[0]) if rows else {},
        "decision": "If a full-pass single-task rule appears, lower it and submit as a calibration delta; otherwise expand beyond conjunctive boolean object-role predicates.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: no ONNX candidate generated yet",
        "leakage_risk": "medium: feature candidates are induced from train positives; full arc-gen and Kaggle delta are required before trust.",
        "overfitting_risk": "medium-high: conjunctive train-fit rules can overfit; use only full-pass candidates for lowering.",
        "outputs": {
            "rule_eval": "rule_eval.csv",
            "notes": "notes.md",
        },
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "rule_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RuleEval.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    notes = f"""# {EXP_ID}

## 目的

L1 sparse background fill taskに対して、object-role / bbox-local boolean featureのAND ruleを探索する。

## 結果

- scanned tasks: {len(tasks)}
- evaluated candidates: {len(rows)}
- full pass hits: {len(full_hits)}
- train/test pass hits: {len(train_test_hits)}
- best overall: {asdict(rows[0]) if rows else {}}

## 解釈

full-pass single-task ruleが出た場合のみ、低cost ONNXへloweringしてKaggle single-task delta提出へ進む。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
