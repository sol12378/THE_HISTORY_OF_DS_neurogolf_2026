from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from itertools import combinations
from typing import Any, Callable

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp_b016_l2_local_predicate_decision_tree_miner"
EXP_DIR = ROOT / "experiments" / EXP_ID
TAXONOMY_PATH = ROOT / "experiments" / "exp054_signature_lookup_family_taxonomy" / "task_taxonomy.csv"
FAST_TOP_K = 5


PredicateFn = Callable[[np.ndarray, int, int, int], bool]


@dataclass(frozen=True)
class Predicate:
    name: str
    fn: PredicateFn
    lowering_plan: str


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


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def load_targets() -> dict[int, dict[str, str]]:
    with TAXONOMY_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lane_rows = [
        row
        for row in rows
        if row["compiler_lane"] in {"L1_static_sparse_background_fill", "L2_local_predicate_sparse_fill"}
    ]
    lane_rows = sorted(lane_rows, key=lambda row: (-float(row["gain_to_250"]), int(row["task_id"])))
    return {int(row["task_id"]): row for row in lane_rows[:FAST_TOP_K]}


def colors(x: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(x) if int(v) != 0]


def count_offsets(x: np.ndarray, r: int, c: int, color: int, offsets: list[tuple[int, int]]) -> int:
    n = 0
    for dr, dc in offsets:
        rr, cc = r + dr, c + dc
        if 0 <= rr < x.shape[0] and 0 <= cc < x.shape[1] and int(x[rr, cc]) == color:
            n += 1
    return n


ADJ4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG4 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL8 = ADJ4 + DIAG4
RADIUS2 = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if not (dr == 0 and dc == 0)]


def sees_color(x: np.ndarray, r: int, c: int, color: int, dr: int, dc: int) -> bool:
    rr, cc = r + dr, c + dc
    while 0 <= rr < x.shape[0] and 0 <= cc < x.shape[1]:
        if x[rr, cc] != 0:
            return int(x[rr, cc]) == color
        rr += dr
        cc += dc
    return False


def color_bbox(x: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(x == color)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def predicate_bank() -> list[Predicate]:
    preds: list[Predicate] = []
    for k in range(1, 5):
        preds.append(Predicate(f"adj4_count_eq_{k}", lambda x, r, c, color, k=k: count_offsets(x, r, c, color, ADJ4) == k, "3x3 Conv count over cardinal neighbors"))
        preds.append(Predicate(f"diag4_count_eq_{k}", lambda x, r, c, color, k=k: count_offsets(x, r, c, color, DIAG4) == k, "3x3 Conv count over diagonal neighbors"))
        preds.append(Predicate(f"all8_count_eq_{k}", lambda x, r, c, color, k=k: count_offsets(x, r, c, color, ALL8) == k, "3x3 Conv count over all neighbors"))
    for k in range(2, 9):
        preds.append(Predicate(f"all8_count_ge_{k}", lambda x, r, c, color, k=k: count_offsets(x, r, c, color, ALL8) >= k, "3x3 Conv threshold"))
    for k in range(2, 13):
        preds.append(Predicate(f"radius2_count_ge_{k}", lambda x, r, c, color, k=k: count_offsets(x, r, c, color, RADIUS2) >= k, "5x5 Conv threshold"))
    preds.extend(
        [
            Predicate("opposite_horizontal_seen", lambda x, r, c, color: sees_color(x, r, c, color, 0, -1) and sees_color(x, r, c, color, 0, 1), "row directional reductions"),
            Predicate("opposite_vertical_seen", lambda x, r, c, color: sees_color(x, r, c, color, -1, 0) and sees_color(x, r, c, color, 1, 0), "column directional reductions"),
            Predicate("opposite_diag_main_seen", lambda x, r, c, color: sees_color(x, r, c, color, -1, -1) and sees_color(x, r, c, color, 1, 1), "diagonal directional reductions"),
            Predicate("opposite_diag_anti_seen", lambda x, r, c, color: sees_color(x, r, c, color, -1, 1) and sees_color(x, r, c, color, 1, -1), "diagonal directional reductions"),
            Predicate("inside_color_bbox", lambda x, r, c, color: (lambda b: b is not None and b[0] <= r < b[2] and b[1] <= c < b[3])(color_bbox(x, color)), "per-color bbox mask"),
            Predicate("on_color_bbox_border", lambda x, r, c, color: (lambda b: b is not None and b[0] <= r < b[2] and b[1] <= c < b[3] and (r in (b[0], b[2] - 1) or c in (b[1], b[3] - 1)))(color_bbox(x, color)), "per-color bbox border mask"),
            Predicate("inside_color_bbox_interior", lambda x, r, c, color: (lambda b: b is not None and b[0] < r < b[2] - 1 and b[1] < c < b[3] - 1)(color_bbox(x, color)), "per-color bbox interior mask"),
        ]
    )
    return preds


def apply_predicates(x: np.ndarray, preds: tuple[Predicate, ...], mode: str) -> tuple[np.ndarray, str]:
    out = x.copy()
    for r, c in np.argwhere(x == 0):
        votes = []
        for color in colors(x):
            vals = [pred.fn(x, int(r), int(c), color) for pred in preds]
            ok = all(vals) if mode == "and" else any(vals)
            if ok:
                votes.append(color)
        if len(votes) == 1:
            out[int(r), int(c)] = votes[0]
        elif len(votes) > 1:
            return x.copy(), "color_conflict"
    return out, "ok"


def changed_count(x: np.ndarray, y: np.ndarray) -> int:
    return int(np.count_nonzero(x != y)) if x.shape == y.shape else -1


def evaluate(
    task_id: int,
    meta: dict[str, str],
    task: dict[str, Any],
    preds: tuple[Predicate, ...],
    mode: str,
) -> CandidateResult:
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
        pred, reason = apply_predicates(x, preds, mode)
        if np.array_equal(pred, y):
            split_pass[split] += 1
        else:
            fail_reasons[reason] += 1
        changed_cells.append(changed_count(x, pred))
    total_pass = sum(split_pass.values())
    status = "full_pass" if total_pass == len(examples) else ("train_test_pass" if split_pass["train"] == split_total["train"] and split_pass["test"] == split_total["test"] else ("train_fit" if split_pass["train"] == split_total["train"] else "partial"))
    name = f"{mode}:" + "&".join(pred.name for pred in preds)
    lowering_plan = " + ".join(sorted({pred.lowering_plan for pred in preds}))
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


def train_pass_count(task: dict[str, Any], preds: tuple[Predicate, ...], mode: str) -> tuple[int, int]:
    passed = 0
    total = 0
    for ex in task["train"]:
        total += 1
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred, _ = apply_predicates(x, preds, mode)
        if np.array_equal(pred, y):
            passed += 1
    return passed, total


def candidate_predicate_sets(preds: list[Predicate]) -> list[tuple[str, tuple[Predicate, ...]]]:
    out: list[tuple[str, tuple[Predicate, ...]]] = []
    for pred in preds:
        out.append(("and", (pred,)))
    return out


def notes_text(result: dict[str, Any]) -> str:
    return f"""# {EXP_ID}

## 目的

L1/L2 sparse fillを、zero cellに対する3x3/5x5近傍・ray・bbox predicateの小さいdecision ruleとして合成する。

## 結果

- target tasks: {result["target_task_count"]}
- predicate count: {result["predicate_count"]}
- candidate predicate sets: {result["predicate_set_count"]}
- evaluated candidates: {result["evaluated_candidate_count"]}
- full pass hits: {result["full_pass_hit_count"]}
- train/test hits: {result["train_test_pass_hit_count"]}
- train-fit hits: {result["train_fit_hit_count"]}

## 判断

full passがあればsmall Conv/mask loweringへ進む。なければ、色roleの決定やnegative samplingを加えたtask-specific treeへ進む。
"""


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    preds = predicate_bank()
    pred_sets = candidate_predicate_sets(preds)
    rows: list[CandidateResult] = []
    screened_candidate_count = 0
    train_partial_counter = Counter()
    for task_id, meta in sorted(targets.items()):
        task = load_task(task_id)
        for mode, pred_tuple in pred_sets:
            screened_candidate_count += 1
            train_pass, train_total = train_pass_count(task, pred_tuple, mode)
            train_partial_counter[f"{train_pass}/{train_total}"] += 1
            if train_pass == train_total:
                rows.append(evaluate(task_id, meta, task, pred_tuple, mode))
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
        "predicate_count": len(preds),
        "predicate_set_count": len(pred_sets),
        "screened_candidate_count": screened_candidate_count,
        "evaluated_candidate_count": len(rows),
        "train_screen_counts": dict(train_partial_counter),
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
        "decision": "lower full pass hits; otherwise add color-role target selection and learned small decision trees",
        "leakage_risk": "low-to-medium: predicates are explanatory; arc-gen used only for evaluation.",
        "overfitting_risk": "medium: train-derived predicate selection needs full arc-gen and Kaggle delta before adoption.",
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
