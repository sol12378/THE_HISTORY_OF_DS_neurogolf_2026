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


EXP_ID = "exp_b009_bbox_affine_formula_miner"
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
    formula_count: int
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
    return [
        int(row["task_id"])
        for row in rows
        if "sparse background fill" in row["suggested_grammar"] or "sparse color-role fill" in row["suggested_grammar"]
    ]


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


def coord_values(n: int) -> dict[str, int]:
    vals = {
        "0": 0,
        "1": 1,
        "2": 2,
        "n-3": n - 3,
        "n-2": n - 2,
        "n-1": n - 1,
        "n": n,
        "n+1": n + 1,
        "2n-1": 2 * n - 1,
        "2n": 2 * n,
        "2n+1": 2 * n + 1,
    }
    if n % 2 == 1:
        vals["mid"] = n // 2
        vals["mid-1"] = n // 2 - 1
        vals["mid+1"] = n // 2 + 1
    return vals


def possible_formulas_for_cell(rel: tuple[int, int], h: int, w: int) -> set[tuple[str, str]]:
    rr, cc = rel
    r_forms = [name for name, val in coord_values(h).items() if val == rr]
    c_forms = [name for name, val in coord_values(w).items() if val == cc]
    return set((rf, cf) for rf in r_forms for cf in c_forms)


def eval_formula(form: tuple[str, str], h: int, w: int) -> tuple[int, int] | None:
    rf, cf = form
    rv = coord_values(h).get(rf)
    cv = coord_values(w).get(cf)
    if rv is None or cv is None:
        return None
    return rv, cv


def infer_color_roles(train_examples: list[dict[str, Any]]) -> list[str]:
    roles = ["const", "min_nonzero", "max_nonzero", "most_common_nonzero", "least_common_nonzero"]
    out = []
    const_color = None
    for role in roles:
        ok = True
        for ex in train_examples:
            x, _, _, color = changed_info(ex)
            if color is None:
                ok = False
                break
            cnt = Counter(int(v) for v in x.ravel() if int(v) != 0)
            if role == "const":
                const_color = color if const_color is None else const_color
                pred = const_color
            elif role == "min_nonzero":
                pred = min(cnt) if cnt else None
            elif role == "max_nonzero":
                pred = max(cnt) if cnt else None
            elif role == "most_common_nonzero":
                pred = cnt.most_common(1)[0][0] if cnt else None
            else:
                pred = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0][0] if cnt else None
            if pred != color:
                ok = False
                break
        if ok:
            out.append(f"const_{const_color}" if role == "const" else role)
    return out


def resolve_color(x: np.ndarray, role: str) -> int | None:
    if role.startswith("const_"):
        return int(role.split("_", 1)[1])
    cnt = Counter(int(v) for v in x.ravel() if int(v) != 0)
    if not cnt:
        return None
    if role == "min_nonzero":
        return min(cnt)
    if role == "max_nonzero":
        return max(cnt)
    if role == "most_common_nonzero":
        return cnt.most_common(1)[0][0]
    if role == "least_common_nonzero":
        return sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0][0]
    return None


def infer_formula_sets(task: dict[str, Any]) -> list[frozenset[tuple[str, str]]]:
    examples = list(task["train"])
    per_example_possible: list[list[set[tuple[str, str]]]] = []
    counts = []
    for ex in examples:
        x, _, cells, color = changed_info(ex)
        if color is None or not cells:
            return []
        b = bbox(x != 0)
        if b is None:
            return []
        r0, c0, r1, c1 = b
        h, w = r1 - r0, c1 - c0
        rels = [(r - r0, c - c0) for r, c in sorted(cells)]
        counts.append(len(rels))
        cell_possible = []
        for rel in rels:
            poss = possible_formulas_for_cell(rel, h, w)
            if not poss:
                return []
            cell_possible.append(poss)
        per_example_possible.append(cell_possible)
    if len(set(counts)) != 1 or not counts:
        return []
    k = counts[0]
    candidates: list[frozenset[tuple[str, str]]] = []
    all_forms = sorted(set().union(*[set().union(*pe) for pe in per_example_possible]))
    for combo in combinations(all_forms, k):
        combo_set = frozenset(combo)
        ok = True
        for pe in per_example_possible:
            # Every actual changed cell must be covered by exactly one formula candidate for that example.
            if not all(any(form in poss for form in combo_set) for poss in pe):
                ok = False
                break
        if ok:
            candidates.append(combo_set)
    return candidates[:200]


def apply_candidate(x: np.ndarray, color_role: str, formulas: frozenset[tuple[str, str]]) -> tuple[np.ndarray, str]:
    b = bbox(x != 0)
    if b is None:
        return x.copy(), "empty"
    r0, c0, r1, c1 = b
    h, w = r1 - r0, c1 - c0
    color = resolve_color(x, color_role)
    if color is None:
        return x.copy(), "no_color"
    out = x.copy()
    cells = set()
    for form in formulas:
        rel = eval_formula(form, h, w)
        if rel is None:
            return x.copy(), "formula_undefined"
        rr, cc = rel
        r = r0 + rr
        c = c0 + cc
        if not (0 <= r < x.shape[0] and 0 <= c < x.shape[1]):
            return x.copy(), "cell_oob"
        cells.add((r, c))
    for r, c in cells:
        if out[r, c] != 0:
            return x.copy(), "target_not_background"
        out[r, c] = color
    return out, "ok"


def evaluate(task_id: int, task: dict[str, Any], color_role: str, formulas: frozenset[tuple[str, str]]) -> CandidateResult:
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
        pred, reason = apply_candidate(x, color_role, formulas)
        if np.array_equal(pred, y):
            split_pass[split] += 1
        else:
            fail_reasons[reason] += 1
    total_pass = sum(split_pass.values())
    status = "full_pass" if total_pass == len(examples) else ("train_test_pass" if split_pass["train"] == split_total["train"] and split_pass["test"] == split_total["test"] else "partial")
    return CandidateResult(
        task_id=task_id,
        candidate=f"{color_role}:{sorted(formulas)}",
        status=status,
        train_pass=split_pass["train"],
        train_examples=split_total["train"],
        test_pass=split_pass["test"],
        test_examples=split_total["test"],
        arc_pass=split_pass["arc-gen"],
        arc_examples=split_total["arc-gen"],
        total_pass=total_pass,
        total_examples=len(examples),
        formula_count=len(formulas),
        fail_reasons=json.dumps(dict(fail_reasons.most_common(8)), ensure_ascii=False),
        lowering_plan="affine bbox coordinate generator plus tiny fill mask; target cost<=250 if formula_count small",
    )


def notes_text(result: dict[str, Any]) -> str:
    return f"""# {EXP_ID}

## 目的

bbox height/widthから変更セル座標を生成するaffine-like formulaを探索する。shape-conditioned lookupより説明可能で、cost<=250 loweringに近い。

## 結果

- target tasks: {result["target_task_count"]}
- evaluated candidates: {result["evaluated_candidate_count"]}
- full pass hits: {result["full_pass_hit_count"]}
- train/test pass hits: {result["train_test_pass_hit_count"]}
- best partial: {result["best_partial"]}

## 判断

full passがあればaffine coordinate generatorとしてloweringへ進む。なければ、bbox affineだけでは不足で、object-role target selectionへ移る。
"""


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    target_ids = load_targets()
    rows: list[CandidateResult] = []
    no_candidate_tasks: list[int] = []
    for task_id in target_ids:
        task = load_task(task_id)
        roles = infer_color_roles(list(task["train"]))
        formula_sets = infer_formula_sets(task)
        task_count = 0
        for role in roles:
            for formulas in formula_sets:
                row = evaluate(task_id, task, role, formulas)
                if row.train_pass == row.train_examples:
                    rows.append(row)
                    task_count += 1
        if task_count == 0:
            no_candidate_tasks.append(task_id)
    rows = sorted(rows, key=lambda r: (r.status != "full_pass", r.status != "train_test_pass", -(r.total_pass / r.total_examples), r.formula_count, r.task_id, r.candidate))
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
        "evaluated_candidate_count": len(rows),
        "full_pass_hit_count": len(full),
        "train_test_pass_hit_count": len(train_test),
        "full_pass_hits": [asdict(r) for r in full[:20]],
        "train_test_hits": [asdict(r) for r in train_test[:20]],
        "best_partial": asdict(best) if best else None,
        "decision": "lower full pass if any; otherwise target object-role selection instead of coordinate formula only",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: no ONNX candidate generated" if not full else "no_submit_yet: lower full pass first",
        "leakage_risk": "low-to-medium: formulas are train-derived but explanatory.",
        "overfitting_risk": "medium: must pass all arc-gen and Kaggle delta before acceptance.",
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
