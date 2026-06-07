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


EXP_ID = "exp_b010_component_object_role_sparse_fill_miner"
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
    avg_changed_cells: float
    fail_reasons: str
    lowering_plan: str


RuleFn = Callable[[np.ndarray], tuple[np.ndarray, str]]


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def load_targets() -> list[int]:
    with TAXONOMY_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        int(row["task_id"])
        for row in rows
        if "sparse background fill" in row["suggested_grammar"] or "sparse color-role fill" in row["suggested_grammar"]
    ]


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def colors(x: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(x) if int(v) != 0]


def changed_count(x: np.ndarray, y: np.ndarray) -> int:
    return int(np.count_nonzero(x != y)) if x.shape == y.shape else -1


def fill_row_span_per_color(x: np.ndarray, require_enclosed: bool) -> tuple[np.ndarray, str]:
    out = x.copy()
    for color in colors(x):
        mask = x == color
        for r in range(x.shape[0]):
            cs = np.flatnonzero(mask[r])
            if len(cs) < 2:
                continue
            c0, c1 = int(cs.min()), int(cs.max())
            for c in range(c0 + 1, c1):
                if out[r, c] == 0:
                    if require_enclosed and not (x[r, c0] == color and x[r, c1] == color):
                        continue
                    out[r, c] = color
    return out, "ok"


def fill_col_span_per_color(x: np.ndarray, require_enclosed: bool) -> tuple[np.ndarray, str]:
    out = x.copy()
    for color in colors(x):
        mask = x == color
        for c in range(x.shape[1]):
            rs = np.flatnonzero(mask[:, c])
            if len(rs) < 2:
                continue
            r0, r1 = int(rs.min()), int(rs.max())
            for r in range(r0 + 1, r1):
                if out[r, c] == 0:
                    if require_enclosed and not (x[r0, c] == color and x[r1, c] == color):
                        continue
                    out[r, c] = color
    return out, "ok"


def nearest_color_in_direction(x: np.ndarray, r: int, c: int, dr: int, dc: int) -> int | None:
    rr, cc = r + dr, c + dc
    while 0 <= rr < x.shape[0] and 0 <= cc < x.shape[1]:
        if x[rr, cc] != 0:
            return int(x[rr, cc])
        rr += dr
        cc += dc
    return None


def fill_opposite_ray_same_color(x: np.ndarray, dirs: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, str]:
    out = x.copy()
    for r, c in np.argwhere(x == 0):
        votes = []
        for dr, dc in dirs:
            a = nearest_color_in_direction(x, int(r), int(c), dr, dc)
            b = nearest_color_in_direction(x, int(r), int(c), -dr, -dc)
            if a is not None and a == b:
                votes.append(a)
        if len(set(votes)) == 1 and votes:
            out[int(r), int(c)] = votes[0]
    return out, "ok"


def fill_color_bbox_interior(x: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    out = x.copy()
    for color in colors(x):
        b = bbox(x == color)
        if b is None:
            continue
        r0, c0, r1, c1 = b
        for r in range(r0, r1):
            for c in range(c0, c1):
                if out[r, c] != 0:
                    continue
                if mode == "all":
                    out[r, c] = color
                elif mode == "border_only" and (r in (r0, r1 - 1) or c in (c0, c1 - 1)):
                    out[r, c] = color
                elif mode == "interior_only" and (r0 < r < r1 - 1 and c0 < c < c1 - 1):
                    out[r, c] = color
    return out, "ok"


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
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


def fill_component_bbox_interior(x: np.ndarray, max_area: int) -> tuple[np.ndarray, str]:
    out = x.copy()
    for color in colors(x):
        for comp in components(x == color):
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs) + 1
            c0, c1 = min(cs), max(cs) + 1
            if (r1 - r0) * (c1 - c0) > max_area:
                continue
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if out[r, c] == 0:
                        out[r, c] = color
    return out, "ok"


def fill_zero_holes_with_border_color(x: np.ndarray, same_color_border: bool) -> tuple[np.ndarray, str]:
    out = x.copy()
    zero = x == 0
    for comp in components(zero):
        touches_border = any(r == 0 or c == 0 or r == x.shape[0] - 1 or c == x.shape[1] - 1 for r, c in comp)
        if touches_border:
            continue
        neigh = Counter()
        for r, c in comp:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < x.shape[0] and 0 <= cc < x.shape[1] and x[rr, cc] != 0:
                    neigh[int(x[rr, cc])] += 1
        if not neigh:
            continue
        if same_color_border and len(neigh) != 1:
            continue
        color = neigh.most_common(1)[0][0]
        for r, c in comp:
            out[r, c] = color
    return out, "ok"


def rule_bank() -> list[tuple[str, RuleFn, str]]:
    hv_dirs = ((0, 1), (1, 0))
    diag_dirs = ((1, 1), (1, -1))
    all_dirs = ((0, 1), (1, 0), (1, 1), (1, -1))
    return [
        ("row_span_per_color", lambda x: fill_row_span_per_color(x, False), "row reduction plus equality mask; no full-grid dynamic indices"),
        ("col_span_per_color", lambda x: fill_col_span_per_color(x, False), "column reduction plus equality mask; no full-grid dynamic indices"),
        ("row_col_span_union", lambda x: merge_rules(x, [lambda z: fill_row_span_per_color(z, False), lambda z: fill_col_span_per_color(z, False)]), "row/column reductions plus small masks"),
        ("opposite_ray_hv_same_color", lambda x: fill_opposite_ray_same_color(x, hv_dirs), "two directional reductions over rows/cols"),
        ("opposite_ray_diag_same_color", lambda x: fill_opposite_ray_same_color(x, diag_dirs), "diagonal reductions or small directional scans"),
        ("opposite_ray_all_same_color", lambda x: fill_opposite_ray_same_color(x, all_dirs), "row/column/diagonal reductions"),
        ("color_bbox_all", lambda x: fill_color_bbox_interior(x, "all"), "per-color bbox mask; reject if bbox mask is large"),
        ("color_bbox_border", lambda x: fill_color_bbox_interior(x, "border_only"), "per-color bbox border mask"),
        ("color_bbox_interior", lambda x: fill_color_bbox_interior(x, "interior_only"), "per-color bbox interior mask"),
        ("component_bbox_area16", lambda x: fill_component_bbox_interior(x, 16), "small component bbox masks only"),
        ("component_bbox_area25", lambda x: fill_component_bbox_interior(x, 25), "small component bbox masks only"),
        ("zero_holes_unanimous_border_color", lambda x: fill_zero_holes_with_border_color(x, True), "component hole mask plus neighboring color vote"),
        ("zero_holes_majority_border_color", lambda x: fill_zero_holes_with_border_color(x, False), "component hole mask plus neighboring color vote"),
    ]


def merge_rules(x: np.ndarray, rules: list[RuleFn]) -> tuple[np.ndarray, str]:
    out = x.copy()
    for rule in rules:
        pred, reason = rule(x)
        if reason != "ok":
            return x.copy(), reason
        changed = (out == 0) & (pred != 0)
        conflict = (out != 0) & (pred != 0) & (out != pred)
        if np.any(conflict):
            return x.copy(), "conflict"
        out[changed] = pred[changed]
    return out, "ok"


def train_fit(task: dict[str, Any], rule: RuleFn) -> bool:
    for ex in task["train"]:
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred, _ = rule(x)
        if not np.array_equal(pred, y):
            return False
    return True


def evaluate(task_id: int, task: dict[str, Any], name: str, rule: RuleFn, lowering_plan: str) -> CandidateResult:
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
    status = "full_pass" if total_pass == len(examples) else ("train_test_pass" if split_pass["train"] == split_total["train"] and split_pass["test"] == split_total["test"] else "partial")
    return CandidateResult(
        task_id=task_id,
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
        avg_changed_cells=float(np.mean([v for v in changed_cells if v >= 0])) if changed_cells else 0.0,
        fail_reasons=json.dumps(dict(fail_reasons.most_common(8)), ensure_ascii=False),
        lowering_plan=lowering_plan,
    )


def notes_text(result: dict[str, Any]) -> str:
    return f"""# {EXP_ID}

## 目的

coordinate-firstでなく、object/component/color-role firstのsparse fill ruleを探索する。候補はrow/column reduction、directional reduction、small bbox mask、hole/component maskへloweringできるものに限定する。

## 結果

- target tasks: {result["target_task_count"]}
- evaluated candidates: {result["evaluated_candidate_count"]}
- full pass hits: {result["full_pass_hit_count"]}
- train/test pass hits: {result["train_test_pass_hit_count"]}
- best partial: {result["best_partial"]}

## 判断

full passがあればtiny mask/reduction loweringへ進む。なければ、単一ruleでは不足なので、train-fitではなく部分一致からtask-specific decision treeを合成する。
"""


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    target_ids = load_targets()
    rules = rule_bank()
    rows: list[CandidateResult] = []
    no_candidate_tasks: list[int] = []
    for task_id in target_ids:
        task = load_task(task_id)
        task_rows = []
        for name, rule, lowering_plan in rules:
            # Keep the candidate set honest: only exact train-fit rules are eligible for full evaluation.
            if train_fit(task, rule):
                task_rows.append(evaluate(task_id, task, name, rule, lowering_plan))
        if task_rows:
            rows.extend(task_rows)
        else:
            no_candidate_tasks.append(task_id)
    rows = sorted(rows, key=lambda r: (r.status != "full_pass", r.status != "train_test_pass", -(r.total_pass / r.total_examples), r.avg_changed_cells, r.task_id, r.candidate))
    full = [r for r in rows if r.status == "full_pass"]
    train_test = [r for r in rows if r.status == "train_test_pass"]
    best = rows[0] if rows else None
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "full_pass_found" if full else ("train_test_hit" if train_test else "no_full_hit"),
        "target_task_count": len(target_ids),
        "target_task_ids": target_ids,
        "rule_count": len(rules),
        "no_candidate_tasks": no_candidate_tasks,
        "evaluated_candidate_count": len(rows),
        "full_pass_hit_count": len(full),
        "train_test_pass_hit_count": len(train_test),
        "full_pass_hits": [asdict(r) for r in full[:20]],
        "train_test_hits": [asdict(r) for r in train_test[:20]],
        "best_partial": asdict(best) if best else None,
        "decision": "lower full pass if any; otherwise synthesize task-specific decision trees from partial object-role predicates",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: no ONNX candidate generated" if not full else "no_submit_yet: lower full pass first",
        "leakage_risk": "low-to-medium: rules are explanatory and train-fit gated; no arc-gen labels used for synthesis.",
        "overfitting_risk": "medium: full arc-gen and Kaggle delta required before adoption.",
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
