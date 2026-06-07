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


EXP_ID = "exp_b002_p0_explainable_rule_sweep"
EXP_DIR = ROOT / "experiments" / EXP_ID
BACKLOG_PATH = ROOT / "experiments" / "exp_b001_rule_replacement_backlog" / "rule_replacement_backlog.csv"
TOP_N = 40


@dataclass(frozen=True)
class RuleResult:
    task_id: int
    rank: int
    family: str
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
    fail_reasons: str
    lowering_plan: str
    reject_risk: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def d4_orbits(n: int) -> list[set[tuple[int, int]]]:
    c = (n - 1) / 2.0
    out: set[frozenset[tuple[int, int]]] = set()
    for r in range(n):
        for col in range(n):
            dr = r - c
            dc = col - c
            pts_float = {
                (c + dr, c + dc),
                (c + dr, c - dc),
                (c - dr, c + dc),
                (c - dr, c - dc),
                (c + dc, c + dr),
                (c + dc, c - dr),
                (c - dc, c + dr),
                (c - dc, c - dr),
            }
            pts = set()
            for rr, cc in pts_float:
                if abs(rr - round(rr)) > 1e-9 or abs(cc - round(cc)) > 1e-9:
                    continue
                ir = int(round(rr))
                ic = int(round(cc))
                if 0 <= ir < n and 0 <= ic < n:
                    pts.add((ir, ic))
            out.add(frozenset(pts))
    return [set(x) for x in sorted(out, key=lambda s: (len(s), sorted(s)))]


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def color_positions(x: np.ndarray, color: int, b: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, _, _ = b
    return {(int(r - r0), int(c - c0)) for r, c in np.argwhere(x == color)}


def rule_square_d4_orbit_completion(x: np.ndarray) -> tuple[np.ndarray, str]:
    b = bbox(x != 0)
    if b is None:
        return x.copy(), "empty"
    r0, c0, r1, c1 = b
    h = r1 - r0
    w = c1 - c0
    if h != w or h < 2 or h > 10:
        return x.copy(), f"bbox_not_square_{h}x{w}"
    candidates: list[tuple[int, set[tuple[int, int]], set[tuple[int, int]]]] = []
    for color in [int(v) for v in np.unique(x) if int(v) != 0]:
        pos = color_positions(x, color, b)
        for orbit in d4_orbits(h):
            if not pos <= orbit:
                continue
            missing = orbit - pos
            if not missing:
                continue
            if all(x[r0 + rr, c0 + cc] == 0 for rr, cc in missing):
                candidates.append((color, orbit, missing))
    if len(candidates) != 1:
        return x.copy(), f"candidate_count={len(candidates)}"
    color, _, missing = candidates[0]
    out = x.copy()
    for rr, cc in missing:
        out[r0 + rr, c0 + cc] = color
    return out, "ok"


def rule_color_bbox_rectangle_closure(x: np.ndarray) -> tuple[np.ndarray, str]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in [int(v) for v in np.unique(x) if int(v) != 0]:
        b = bbox(x == color)
        if b is None:
            continue
        r0, c0, r1, c1 = b
        region = x[r0:r1, c0:c1]
        missing = region != color
        if not np.any(missing):
            continue
        if np.all(region[missing] == 0):
            candidates.append((color, b))
    if len(candidates) != 1:
        return x.copy(), f"candidate_count={len(candidates)}"
    color, (r0, c0, r1, c1) = candidates[0]
    out = x.copy()
    out[r0:r1, c0:c1][out[r0:r1, c0:c1] == 0] = color
    return out, "ok"


def rule_row_col_segment_fill(x: np.ndarray) -> tuple[np.ndarray, str]:
    out = x.copy()
    changed = 0
    for color in [int(v) for v in np.unique(x) if int(v) != 0]:
        for r in range(x.shape[0]):
            cols = np.where(x[r] == color)[0]
            if len(cols) >= 2:
                c0, c1 = int(cols.min()), int(cols.max())
                segment = out[r, c0 : c1 + 1]
                mask = segment == 0
                changed += int(mask.sum())
                segment[mask] = color
        for c in range(x.shape[1]):
            rows = np.where(x[:, c] == color)[0]
            if len(rows) >= 2:
                r0, r1 = int(rows.min()), int(rows.max())
                segment = out[r0 : r1 + 1, c]
                mask = segment == 0
                changed += int(mask.sum())
                segment[mask] = color
    if changed == 0:
        return x.copy(), "no_change"
    return out, "ok"


RULES: list[tuple[str, Callable[[np.ndarray], tuple[np.ndarray, str]], str, str]] = [
    (
        "square_d4_orbit_completion",
        rule_square_d4_orbit_completion,
        "constant D4 orbit masks plus small Equal/Where or tiny ScatterND",
        "low if bbox size is static/small; reject if dynamic bbox requires full-grid GatherND",
    ),
    (
        "color_bbox_rectangle_closure",
        rule_color_bbox_rectangle_closure,
        "color Equal mask, Reduce row/column bbox if cheap, then static rectangle mask",
        "medium: bbox detection must avoid NonZero/Compress/full-grid GatherND",
    ),
    (
        "row_col_segment_fill",
        rule_row_col_segment_fill,
        "row/column ReduceSum masks and one bounded Where",
        "medium-low: reject if implemented as long iterative propagation",
    ),
]


def read_backlog() -> list[dict[str, str]]:
    with BACKLOG_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["priority"] == "P0"]
    return sorted(rows, key=lambda r: int(r["rank"]))[:TOP_N]


def evaluate_rule(task_id: int, rank: int, family: str, rule_name: str, fn: Callable[[np.ndarray], tuple[np.ndarray, str]], lowering: str, risk: str) -> RuleResult:
    task = load_task(task_id)
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
        pred, reason = fn(x)
        if np.array_equal(pred, y):
            split_pass[split] += 1
        else:
            fail_reasons[reason] += 1
    total_pass = sum(split_pass.values())
    status = "full_pass" if total_pass == len(examples) else ("train_test_pass" if split_pass["train"] == split_total["train"] and split_pass["test"] == split_total["test"] else "partial")
    return RuleResult(
        task_id=task_id,
        rank=rank,
        family=family,
        rule_name=rule_name,
        status=status,
        total_pass=total_pass,
        total_examples=len(examples),
        train_pass=split_pass["train"],
        train_examples=split_total["train"],
        test_pass=split_pass["test"],
        test_examples=split_total["test"],
        arc_pass=split_pass["arc-gen"],
        arc_examples=split_total["arc-gen"],
        fail_reasons=json.dumps(dict(fail_reasons.most_common(8)), ensure_ascii=False),
        lowering_plan=lowering,
        reject_risk=risk,
    )


def notes_text(result: dict[str, object]) -> str:
    return f"""# {EXP_ID}

## 目的

exp_b routeの最初のrule miningとして、P0上位{TOP_N} taskに対して、説明可能かつ安いlowering候補になりうるrule familyを横断探索する。

## 仮説

signature lookup teacherの一部は、D4 orbit completion、rectangle closure、row/column segment fillのような小さな幾何ruleで置換できる。

## 結果

- scanned tasks: {result["scanned_task_count"]}
- evaluated rows: {result["evaluated_rule_count"]}
- full pass hits: {result["full_pass_hit_count"]}
- train/test pass hits: {result["train_test_pass_hit_count"]}
- best partial: {result["best_partial"]}

## 解釈

full pass hitがあれば次にONNX loweringへ進む。full passがなければ、失敗理由を見てrule familyを追加する。local/LB較正のため、採用はsmall delta submission単位に限定する。

## Risk

- leakage risk: 低。arc-gen label lookupやteacher output tableは使っていない。
- overfitting risk: 中。P0 taskに対する固定rule family sweepなので、hit後はfamily holdoutとKaggle small submissionが必要。
"""


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    backlog = read_backlog()
    rows: list[RuleResult] = []
    for item in backlog:
        task_id = int(item["task_id"])
        rank = int(item["rank"])
        family = item["family"]
        for rule_name, fn, lowering, risk in RULES:
            rows.append(evaluate_rule(task_id, rank, family, rule_name, fn, lowering, risk))
    rows = sorted(rows, key=lambda r: (r.status != "full_pass", r.status != "train_test_pass", -(r.total_pass / r.total_examples), r.rank, r.rule_name))
    full = [r for r in rows if r.status == "full_pass"]
    train_test = [r for r in rows if r.status == "train_test_pass"]
    best = rows[0] if rows else None
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "full_pass_found" if full else ("train_test_hit" if train_test else "no_full_hit"),
        "top_n": TOP_N,
        "scanned_task_count": len(backlog),
        "evaluated_rule_count": len(rows),
        "full_pass_hit_count": len(full),
        "train_test_pass_hit_count": len(train_test),
        "full_pass_hits": [asdict(r) for r in full[:20]],
        "train_test_hits": [asdict(r) for r in train_test[:20]],
        "best_partial": asdict(best) if best else None,
        "decision": "lower full_pass hits to ONNX; otherwise add new explainable sparse/object rule families before submission",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: rule sweep only; no full-pass ONNX candidate yet" if not full else "no_submit_yet: lower full-pass candidates first",
        "leakage_risk": "low: no teacher label table or arc-gen memorization.",
        "overfitting_risk": "medium: all arc-gen used for diagnostics; accepted candidates need family holdout and Kaggle delta submission.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "rule_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RuleResult.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    (EXP_DIR / "notes.md").write_text(notes_text(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
