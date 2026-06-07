from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from itertools import product
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp_b007_bbox_local_role_miner"
EXP_DIR = ROOT / "experiments" / EXP_ID
TAXONOMY_PATH = ROOT / "experiments" / "exp_b004_teacher_gain_p0_structural_taxonomy" / "task_taxonomy.csv"


@dataclass(frozen=True)
class CandidateResult:
    task_id: int
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
    estimated_lowering_cost_class: str
    fail_reasons: str
    lowering_plan: str


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


def load_targets() -> list[int]:
    with TAXONOMY_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        grammar = row["suggested_grammar"]
        if "sparse background fill" in grammar or "sparse color-role fill" in grammar:
            out.append(int(row["task_id"]))
    return out


def changed_info(ex: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, set[tuple[int, int]], int | None]:
    x = arr(ex["input"])
    y = arr(ex["output"])
    if x.shape != y.shape:
        return x, y, set(), None
    changed = x != y
    if not np.any(changed) or not np.all(x[changed] == 0):
        return x, y, set(), None
    colors = [int(v) for v in np.unique(y[changed])]
    if len(colors) != 1:
        return x, y, set(), None
    return x, y, {(int(r), int(c)) for r, c in np.argwhere(changed)}, colors[0]


def rel_to_bbox(cells: set[tuple[int, int]], b: tuple[int, int, int, int]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = b
    return frozenset((r - r0, c - c0) for r, c in cells)


def rel_to_center(cells: set[tuple[int, int]], b: tuple[int, int, int, int]) -> frozenset[tuple[int, int]]:
    r0, c0, r1, c1 = b
    cr = (r0 + r1 - 1) / 2.0
    cc = (c0 + c1 - 1) / 2.0
    out = []
    for r, c in cells:
        dr = r - cr
        dc = c - cc
        if abs(dr - round(dr)) > 1e-9 or abs(dc - round(dc)) > 1e-9:
            return frozenset()
        out.append((int(round(dr)), int(round(dc))))
    return frozenset(out)


def infer_static_rel_candidates(train_examples: list[dict[str, Any]]) -> list[tuple[str, int, frozenset[tuple[int, int]]]]:
    by_color_bbox: dict[int, list[frozenset[tuple[int, int]]]] = {}
    by_color_center: dict[int, list[frozenset[tuple[int, int]]]] = {}
    for ex in train_examples:
        x, _, cells, color = changed_info(ex)
        if color is None or not cells:
            return []
        b = bbox(x != 0)
        if b is None:
            return []
        by_color_bbox.setdefault(color, []).append(rel_to_bbox(cells, b))
        center_rel = rel_to_center(cells, b)
        if center_rel:
            by_color_center.setdefault(color, []).append(center_rel)
    out: list[tuple[str, int, frozenset[tuple[int, int]]]] = []
    for color, rels in by_color_bbox.items():
        if rels and all(r == rels[0] for r in rels):
            out.append(("bbox_rel", color, rels[0]))
    for color, rels in by_color_center.items():
        if rels and all(r == rels[0] for r in rels):
            out.append(("center_rel", color, rels[0]))
    return out


def color_role_candidates(train_examples: list[dict[str, Any]]) -> list[tuple[str, str]]:
    roles = ["min_nonzero", "max_nonzero", "most_common_nonzero", "least_common_nonzero"]
    possible: list[tuple[str, str]] = []
    for role in roles:
        ok = True
        for ex in train_examples:
            x, _, _, color = changed_info(ex)
            if color is None:
                ok = False
                break
            nonzero = [int(v) for v in x.ravel() if int(v) != 0]
            if not nonzero:
                ok = False
                break
            cnt = Counter(nonzero)
            if role == "min_nonzero":
                pred = min(cnt)
            elif role == "max_nonzero":
                pred = max(cnt)
            elif role == "most_common_nonzero":
                pred = cnt.most_common(1)[0][0]
            else:
                pred = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0][0]
            if pred != color:
                ok = False
                break
        if ok:
            possible.append(("target_color_role", role))
    return possible


def resolve_color(x: np.ndarray, color_spec: str) -> int | None:
    if color_spec.startswith("const_"):
        return int(color_spec.split("_", 1)[1])
    nonzero = [int(v) for v in x.ravel() if int(v) != 0]
    if not nonzero:
        return None
    cnt = Counter(nonzero)
    role = color_spec
    if role == "min_nonzero":
        return min(cnt)
    if role == "max_nonzero":
        return max(cnt)
    if role == "most_common_nonzero":
        return cnt.most_common(1)[0][0]
    if role == "least_common_nonzero":
        return sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0][0]
    return None


def apply_candidate(x: np.ndarray, mode: str, color_spec: str, rels: frozenset[tuple[int, int]]) -> tuple[np.ndarray, str]:
    b = bbox(x != 0)
    if b is None:
        return x.copy(), "empty"
    r0, c0, r1, c1 = b
    color = resolve_color(x, color_spec)
    if color is None:
        return x.copy(), "no_color"
    out = x.copy()
    cells = []
    if mode == "bbox_rel":
        for dr, dc in rels:
            cells.append((r0 + dr, c0 + dc))
    elif mode == "center_rel":
        cr = (r0 + r1 - 1) / 2.0
        cc = (c0 + c1 - 1) / 2.0
        for dr, dc in rels:
            rr = cr + dr
            cc2 = cc + dc
            if abs(rr - round(rr)) > 1e-9 or abs(cc2 - round(cc2)) > 1e-9:
                return x.copy(), "non_integer_center"
            cells.append((int(round(rr)), int(round(cc2))))
    else:
        return x.copy(), "bad_mode"
    changed = 0
    for r, c in cells:
        if not (0 <= r < x.shape[0] and 0 <= c < x.shape[1]):
            return x.copy(), "cell_oob"
        if out[r, c] != 0:
            return x.copy(), "target_not_background"
        out[r, c] = color
        changed += 1
    if changed == 0:
        return x.copy(), "no_change"
    return out, "ok"


def train_fit_candidates(task: dict[str, Any]) -> list[tuple[str, str, frozenset[tuple[int, int]]]]:
    train_examples = list(task["train"])
    static = infer_static_rel_candidates(train_examples)
    roles = color_role_candidates(train_examples)
    out = []
    for mode, color, rels in static:
        color_specs = [f"const_{color}"] + [role for _, role in roles]
        for color_spec in sorted(set(color_specs)):
            ok = True
            for ex in train_examples:
                x = arr(ex["input"])
                y = arr(ex["output"])
                pred, _ = apply_candidate(x, mode, color_spec, rels)
                if not np.array_equal(pred, y):
                    ok = False
                    break
            if ok:
                out.append((mode, color_spec, rels))
    return out


def evaluate(task_id: int, task: dict[str, Any], cand: tuple[str, str, frozenset[tuple[int, int]]]) -> CandidateResult:
    mode, color_spec, rels = cand
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
        pred, reason = apply_candidate(x, mode, color_spec, rels)
        if np.array_equal(pred, y):
            split_pass[split] += 1
        else:
            fail_reasons[reason] += 1
    total_pass = sum(split_pass.values())
    status = "full_pass" if total_pass == len(examples) else ("train_test_pass" if split_pass["train"] == split_total["train"] and split_pass["test"] == split_total["test"] else "partial")
    est_class = "target_cost<=250_plausible" if len(rels) <= 8 and color_spec.startswith("const_") else "target_cost<=250_possible_with_role_logic"
    return CandidateResult(
        task_id=task_id,
        candidate=f"{mode}:{color_spec}:{sorted(rels)}",
        status=status,
        train_pass=split_pass["train"],
        train_examples=split_total["train"],
        test_pass=split_pass["test"],
        test_examples=split_total["test"],
        arc_pass=split_pass["arc-gen"],
        arc_examples=split_total["arc-gen"],
        total_pass=total_pass,
        total_examples=len(examples),
        estimated_lowering_cost_class=est_class,
        fail_reasons=json.dumps(dict(fail_reasons.most_common(8)), ensure_ascii=False),
        lowering_plan="bbox-local tiny coordinate fill; implement as small coordinate mask/ScatterND only if full pass",
    )


def notes_text(result: dict[str, Any]) -> str:
    return f"""# {EXP_ID}

## 目的

cost<=250を意識し、P0 sparse background fill taskに対してbbox-local coordinateとtarget color roleを組み合わせた説明可能ruleを探索する。

## 結果

- target tasks: {result["target_task_count"]}
- train-fit candidates: {result["train_fit_candidate_count"]}
- full pass hits: {result["full_pass_hit_count"]}
- train/test pass hits: {result["train_test_pass_hit_count"]}
- best partial: {result["best_partial"]}

## 判断

full pass hitがあればtiny coordinate loweringへ進む。なければ、static relative positionsだけでは不足なので、shape-conditioned / orbit-conditioned relative positionsを追加する。
"""


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    target_ids = load_targets()
    rows: list[CandidateResult] = []
    no_candidate_tasks: list[int] = []
    for task_id in target_ids:
        task = load_task(task_id)
        candidates = train_fit_candidates(task)
        if not candidates:
            no_candidate_tasks.append(task_id)
        for cand in candidates:
            rows.append(evaluate(task_id, task, cand))
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
        "train_fit_candidate_count": len(rows),
        "evaluated_candidate_count": len(rows),
        "full_pass_hit_count": len(full),
        "train_test_pass_hit_count": len(train_test),
        "full_pass_hits": [asdict(r) for r in full[:20]],
        "train_test_hits": [asdict(r) for r in train_test[:20]],
        "best_partial": asdict(best) if best else None,
        "decision": "lower full-pass tiny rules; otherwise add shape/orbit-conditioned relative coordinate grammar",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: no ONNX candidate generated" if not full else "no_submit_yet: lower full pass first",
        "leakage_risk": "low: train-fit rules use coordinates/roles, not arc-gen labels.",
        "overfitting_risk": "medium: train-fit candidates must pass all arc-gen and Kaggle delta before acceptance.",
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
