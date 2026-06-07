from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp070_local_predicate_fill_feature_profile"
EXP_DIR = ROOT / "experiments" / EXP_ID
QUEUE_CSV = ROOT / "experiments" / "exp069_karnak_prior_compiler_queue" / "compiler_priority_queue.csv"
TARGET_LANE = "LOCAL_PREDICATE_FILL_COMPILER"
TOP_N = 10


@dataclass(frozen=True)
class TaskProfile:
    task_id: int
    examples: int
    changed_examples: int
    same_shape_examples: int
    all_changes_zero_to_color: bool
    changed_cell_count_hist: str
    target_color_hist: str
    bbox_shape_hist: str
    local_template_count: int
    top_templates: str
    profile_class: str
    recommended_action: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def canonical_translate(points: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not points:
        return tuple()
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return tuple(sorted((r - min_r, c - min_c) for r, c in points))


def d4_variants(points: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    if not points:
        return [tuple()]
    max_r = max(r for r, _ in points)
    max_c = max(c for _, c in points)
    h = max(max_r + 1, max_c + 1)
    variants = []
    for mode in range(8):
        out = []
        for r, c in points:
            if mode == 0:
                rr, cc = r, c
            elif mode == 1:
                rr, cc = r, h - 1 - c
            elif mode == 2:
                rr, cc = h - 1 - r, c
            elif mode == 3:
                rr, cc = h - 1 - r, h - 1 - c
            elif mode == 4:
                rr, cc = c, r
            elif mode == 5:
                rr, cc = c, h - 1 - r
            elif mode == 6:
                rr, cc = h - 1 - c, r
            else:
                rr, cc = h - 1 - c, h - 1 - r
            out.append((rr, cc))
        variants.append(canonical_translate(out))
    return sorted(set(variants))


def canon_template(points: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    base = canonical_translate(points)
    return min(d4_variants(base))


def read_targets() -> list[int]:
    rows = []
    with QUEUE_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["compiler_lane"] == TARGET_LANE:
                rows.append((float(row["compiler_priority_score"]), int(row["task_id"])))
    rows.sort(reverse=True)
    return [task_id for _, task_id in rows[:TOP_N]]


def profile_task(task_id: int) -> TaskProfile:
    task = load_task(task_id)
    examples = examples_for(task, -1)
    changed_counts = Counter()
    target_colors = Counter()
    bbox_shapes = Counter()
    templates = Counter()
    changed_examples = 0
    same_shape = 0
    all_zero_to_color = True
    for ex in examples:
        x = arr(ex["input"])
        y = arr(ex["output"])
        if x.shape == y.shape:
            same_shape += 1
        else:
            all_zero_to_color = False
            continue
        diff = x != y
        if not np.any(diff):
            changed_counts[0] += 1
            continue
        changed_examples += 1
        coords = [(int(r), int(c)) for r, c in np.argwhere(diff)]
        changed_counts[len(coords)] += 1
        out_colors = [int(y[r, c]) for r, c in coords]
        in_colors = [int(x[r, c]) for r, c in coords]
        if any(v != 0 for v in in_colors):
            all_zero_to_color = False
        if len(set(out_colors)) != 1:
            all_zero_to_color = False
        else:
            target_colors[out_colors[0]] += 1
        b = bbox(x != 0)
        if b is not None:
            bbox_shapes[(b[2] - b[0], b[3] - b[1])] += 1
            local_points = [(r - b[0], c - b[1]) for r, c in coords]
        else:
            bbox_shapes[(0, 0)] += 1
            local_points = coords
        templates[canon_template(local_points)] += 1

    if same_shape != len(examples):
        cls = "not_same_shape"
        action = "route to crop/object compiler before sparse fill"
    elif all_zero_to_color and len(templates) <= 8 and changed_examples > 0:
        cls = "template_sparse_fill_promising"
        action = "mine selector for small canonical templates, then lower with sparse one-hot writeback"
    elif all_zero_to_color and len(changed_counts) <= 4:
        cls = "predicate_sparse_fill_promising"
        action = "build changed/nonchanged cell dataset and synthesize local predicate tree"
    elif changed_examples > 0:
        cls = "complex_sparse_or_object_edit"
        action = "profile components/color roles before rule mining"
    else:
        cls = "no_change_or_unknown"
        action = "deprioritize"
    return TaskProfile(
        task_id=task_id,
        examples=len(examples),
        changed_examples=changed_examples,
        same_shape_examples=same_shape,
        all_changes_zero_to_color=all_zero_to_color,
        changed_cell_count_hist=json.dumps(dict(changed_counts), ensure_ascii=False, sort_keys=True),
        target_color_hist=json.dumps(dict(target_colors), ensure_ascii=False, sort_keys=True),
        bbox_shape_hist=json.dumps({str(k): v for k, v in bbox_shapes.items()}, ensure_ascii=False, sort_keys=True),
        local_template_count=len(templates),
        top_templates=json.dumps([{"template": str(k), "count": v} for k, v in templates.most_common(8)], ensure_ascii=False),
        profile_class=cls,
        recommended_action=action,
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    targets = read_targets()
    profiles = [profile_task(task_id) for task_id in targets]
    with (EXP_DIR / "feature_profile.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TaskProfile.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in profiles])
    class_counts = Counter(row.profile_class for row in profiles)
    promising = [row.task_id for row in profiles if "promising" in row.profile_class]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "profile_ready",
        "target_lane": TARGET_LANE,
        "targets": targets,
        "class_counts": dict(class_counts),
        "promising_tasks": promising,
        "profiles": [asdict(row) for row in profiles],
        "decision": "Use promising template/predicate sparse fill tasks for the next rule miner. If none are promising, switch to component/color-role profiling.",
        "leakage_risk": "low: profile uses train/test/arc-gen structure only to choose compiler family, not to emit lookup tables.",
        "overfitting_risk": "medium if templates are used directly; next step must synthesize explainable selectors and validate all arc-gen.",
        "outputs": {"feature_profile": "feature_profile.csv"},
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`exp069` LOCAL_PREDICATE_FILL_COMPILER上位taskを、changed-cell feature profileで分類する。

## 結果

- targets: {targets}
- class counts: {dict(class_counts)}
- promising tasks: {promising}

## Decision

promising taskがあれば次にselector/rule minerへ進む。なければcomponent/color-role profileへ深掘りする。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
