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

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp_b008_shape_conditioned_sparse_fill"
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
    branch_count: int
    avg_cells_per_branch: float
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


def rel_bbox(cells: set[tuple[int, int]], b: tuple[int, int, int, int]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = b
    return frozenset((r - r0, c - c0) for r, c in cells)


def key_for(x: np.ndarray, mode: str) -> tuple[Any, ...] | None:
    b = bbox(x != 0)
    if b is None:
        return None
    r0, c0, r1, c1 = b
    h = r1 - r0
    w = c1 - c0
    colors = tuple(sorted(int(v) for v in np.unique(x) if int(v) != 0))
    counts = tuple(sorted(Counter(int(v) for v in x.ravel() if int(v) != 0).values()))
    if mode == "bbox_shape":
        return (h, w)
    if mode == "bbox_shape_colors":
        return (h, w, len(colors))
    if mode == "bbox_shape_color_set":
        return (h, w, colors)
    if mode == "bbox_shape_count_signature":
        return (h, w, counts)
    return None


def infer_color_role(train_examples: list[dict[str, Any]]) -> list[str]:
    roles = ["const", "min_nonzero", "max_nonzero", "most_common_nonzero", "least_common_nonzero"]
    ok_roles = []
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
            if role == "const":
                ok_roles.append(f"const_{const_color}")
            else:
                ok_roles.append(role)
    return ok_roles


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


def build_candidate(task: dict[str, Any], key_mode: str, color_role: str) -> dict[tuple[Any, ...], frozenset[tuple[int, int]]] | None:
    branches: dict[tuple[Any, ...], frozenset[tuple[int, int]]] = {}
    for ex in task["train"]:
        x, _, cells, color = changed_info(ex)
        if color is None or not cells:
            return None
        b = bbox(x != 0)
        key = key_for(x, key_mode)
        if b is None or key is None:
            return None
        rels = rel_bbox(cells, b)
        if key in branches and branches[key] != rels:
            return None
        branches[key] = rels
    return branches


def apply_candidate(x: np.ndarray, key_mode: str, color_role: str, branches: dict[tuple[Any, ...], frozenset[tuple[int, int]]]) -> tuple[np.ndarray, str]:
    key = key_for(x, key_mode)
    b = bbox(x != 0)
    if key is None or b is None:
        return x.copy(), "no_key"
    if key not in branches:
        return x.copy(), "unseen_key"
    color = resolve_color(x, color_role)
    if color is None:
        return x.copy(), "no_color"
    r0, c0, _, _ = b
    out = x.copy()
    for dr, dc in branches[key]:
        r = r0 + dr
        c = c0 + dc
        if not (0 <= r < x.shape[0] and 0 <= c < x.shape[1]):
            return x.copy(), "cell_oob"
        if out[r, c] != 0:
            return x.copy(), "target_not_background"
        out[r, c] = color
    return out, "ok"


def evaluate(task_id: int, task: dict[str, Any], key_mode: str, color_role: str, branches: dict[tuple[Any, ...], frozenset[tuple[int, int]]]) -> CandidateResult:
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
        pred, reason = apply_candidate(x, key_mode, color_role, branches)
        if np.array_equal(pred, y):
            split_pass[split] += 1
        else:
            fail_reasons[reason] += 1
    total_pass = sum(split_pass.values())
    status = "full_pass" if total_pass == len(examples) else ("train_test_pass" if split_pass["train"] == split_total["train"] and split_pass["test"] == split_total["test"] else "partial")
    avg_cells = float(sum(len(v) for v in branches.values()) / max(1, len(branches)))
    cost_class = "target_cost<=250_possible" if len(branches) <= 4 and avg_cells <= 8 else "target_cost<=250_unlikely_without_compression"
    serial = {str(k): sorted(v) for k, v in branches.items()}
    return CandidateResult(
        task_id=task_id,
        candidate=f"{key_mode}:{color_role}:{json.dumps(serial, ensure_ascii=False, sort_keys=True)}",
        status=status,
        train_pass=split_pass["train"],
        train_examples=split_total["train"],
        test_pass=split_pass["test"],
        test_examples=split_total["test"],
        arc_pass=split_pass["arc-gen"],
        arc_examples=split_total["arc-gen"],
        total_pass=total_pass,
        total_examples=len(examples),
        branch_count=len(branches),
        avg_cells_per_branch=avg_cells,
        estimated_lowering_cost_class=cost_class,
        fail_reasons=json.dumps(dict(fail_reasons.most_common(8)), ensure_ascii=False),
        lowering_plan="small decision tree over bbox/shape key plus tiny coordinate fill; submit only if full pass and branch table remains tiny",
    )


def notes_text(result: dict[str, Any]) -> str:
    return f"""# {EXP_ID}

## 目的

固定relative座標ではなく、bbox shape / color signatureで分岐するrelative fill ruleを試す。cost<=250にはbranch数が小さいものだけが候補。

## 結果

- target tasks: {result["target_task_count"]}
- evaluated candidates: {result["evaluated_candidate_count"]}
- full pass hits: {result["full_pass_hit_count"]}
- train/test pass hits: {result["train_test_pass_hit_count"]}
- best partial: {result["best_partial"]}

## 判断

full passかつbranch tableが小さいものだけloweringへ進む。unseen_keyが多ければ、shape keyではなくobject-role/symmetry生成規則が必要。
"""


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    target_ids = load_targets()
    key_modes = ["bbox_shape", "bbox_shape_colors", "bbox_shape_color_set", "bbox_shape_count_signature"]
    rows: list[CandidateResult] = []
    no_candidate_tasks: list[int] = []
    for task_id in target_ids:
        task = load_task(task_id)
        roles = infer_color_role(list(task["train"]))
        task_rows = 0
        for key_mode in key_modes:
            for color_role in roles:
                branches = build_candidate(task, key_mode, color_role)
                if not branches:
                    continue
                row = evaluate(task_id, task, key_mode, color_role, branches)
                rows.append(row)
                task_rows += 1
        if task_rows == 0:
            no_candidate_tasks.append(task_id)
    rows = sorted(rows, key=lambda r: (r.status != "full_pass", r.status != "train_test_pass", -(r.total_pass / r.total_examples), r.branch_count, r.task_id, r.candidate))
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
        "decision": "lower full-pass tiny branch rules; otherwise move from shape-conditioned lookup to generative object-role grammar",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: no ONNX candidate generated" if not full else "no_submit_yet: lower full pass first",
        "leakage_risk": "medium: shape-conditioned branches are train-derived; must remain tiny and explanatory.",
        "overfitting_risk": "medium-high unless all arc-gen full pass and Kaggle delta submission confirm.",
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
