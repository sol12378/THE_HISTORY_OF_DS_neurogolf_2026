from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from statistics import mean, median
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp_b004_teacher_gain_p0_structural_taxonomy"
EXP_DIR = ROOT / "experiments" / EXP_ID
EXP048_RESULT = ROOT / "experiments" / "exp048_submit_safe_seed_inventory" / "result.json"


@dataclass(frozen=True)
class TaskTaxonomy:
    task_id: int
    route: str
    teacher_gain_vs_strict: float
    example_count: int
    same_shape_rate: float
    sparse_edit_rate: float
    only_background_fill_rate: float
    only_deletion_rate: float
    bbox_preserved_rate: float
    changed_count_median: float
    changed_count_min: int
    changed_count_max: int
    shape_pair_count: int
    dominant_change_pairs: str
    suggested_grammar: str
    lowering_bias: str
    priority_note: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def load_p0_tasks() -> list[dict[str, Any]]:
    payload = json.loads(EXP048_RESULT.read_text(encoding="utf-8"))
    return list(payload["top_p0_tasks"])


def suggest(row: dict[str, Any]) -> tuple[str, str, str]:
    only_fill = row["only_background_fill_rate"]
    sparse = row["sparse_edit_rate"]
    bbox_keep = row["bbox_preserved_rate"]
    med = row["changed_count_median"]
    shape_pairs = row["shape_pair_count"]
    if sparse > 0.95 and only_fill > 0.95 and bbox_keep > 0.95 and med <= 6:
        return (
            "sparse background fill with object-role/orbit grammar",
            "tiny ScatterND or one small static mask; avoid full-grid Where chains",
            "highest: likely cheap if rule is found",
        )
    if sparse > 0.95 and only_fill > 0.80:
        return (
            "sparse color-role fill with local neighborhood predicates",
            "small Conv kernels or tiny coordinate program",
            "high: changed cells are sparse but role inference is needed",
        )
    if shape_pairs > 1:
        return (
            "shape-transform/object crop grammar",
            "constant Slice/Gather when shape rule is static; reject full-grid bbox GatherND",
            "medium: first infer shape rule",
        )
    if bbox_keep < 0.5:
        return (
            "object movement/copy grammar",
            "small object masks and static placement; avoid large ScatterND",
            "medium: object correspondence first",
        )
    return (
        "general sparse/object completion grammar",
        "small Conv/Reduce masks before any ScatterND",
        "medium: needs task-specific profile",
    )


def rate(xs: list[bool]) -> float:
    if not xs:
        return 0.0
    return float(sum(1 for x in xs if bool(x)) / len(xs))


def profile_task(item: dict[str, Any]) -> TaskTaxonomy:
    task_id = int(item["task_id"])
    task = load_task(task_id)
    examples = examples_for(task, -1)
    same_shape = []
    sparse = []
    only_fill = []
    only_delete = []
    bbox_keep = []
    changed_counts = []
    shape_pairs = Counter()
    change_pairs = Counter()
    for ex in examples:
        x = arr(ex["input"])
        y = arr(ex["output"])
        same_shape.append(x.shape == y.shape)
        shape_pairs[(tuple(x.shape), tuple(y.shape))] += 1
        if x.shape != y.shape:
            sparse.append(False)
            only_fill.append(False)
            only_delete.append(False)
            bbox_keep.append(False)
            changed_counts.append(-1)
            continue
        changed = x != y
        changed_counts.append(int(changed.sum()))
        sparse.append(np.all(y[~changed] == x[~changed]))
        only_fill.append(bool(np.any(changed)) and np.all(x[changed] == 0) and np.all(y[changed] != 0))
        only_delete.append(bool(np.any(changed)) and np.all(x[changed] != 0) and np.all(y[changed] == 0))
        bbox_keep.append(bbox(x != 0) == bbox(y != 0))
        for a, b in zip(x[changed].ravel(), y[changed].ravel()):
            change_pairs[(int(a), int(b))] += 1
    valid_changed = [c for c in changed_counts if c >= 0]
    raw = {
        "only_background_fill_rate": rate(only_fill),
        "sparse_edit_rate": rate(sparse),
        "bbox_preserved_rate": rate(bbox_keep),
        "changed_count_median": median(valid_changed) if valid_changed else -1,
        "shape_pair_count": len(shape_pairs),
    }
    grammar, lowering, note = suggest(raw)
    return TaskTaxonomy(
        task_id=task_id,
        route=str(item["route"]),
        teacher_gain_vs_strict=float(item["teacher_gain_vs_strict"]),
        example_count=len(examples),
        same_shape_rate=rate(same_shape),
        sparse_edit_rate=rate(sparse),
        only_background_fill_rate=rate(only_fill),
        only_deletion_rate=rate(only_delete),
        bbox_preserved_rate=rate(bbox_keep),
        changed_count_median=raw["changed_count_median"],
        changed_count_min=min(valid_changed) if valid_changed else -1,
        changed_count_max=max(valid_changed) if valid_changed else -1,
        shape_pair_count=len(shape_pairs),
        dominant_change_pairs=json.dumps({str(k): v for k, v in change_pairs.most_common(6)}, ensure_ascii=False),
        suggested_grammar=grammar,
        lowering_bias=lowering,
        priority_note=note,
    )


def notes_text(result: dict[str, Any]) -> str:
    lines = [
        f"# {EXP_ID}",
        "",
        "## 目的",
        "",
        "teacher-gain P0 18 taskについて、次に追加すべき説明可能rule grammarを構造から決める。",
        "",
        "## 結果",
        "",
        f"- profiled tasks: {result['profiled_task_count']}",
        f"- grammar counts: {result['grammar_counts']}",
        "",
        "## Top Priority",
        "",
        "| task | gain | grammar | lowering | note |",
        "|---:|---:|---|---|---|",
    ]
    for row in result["top_rows"][:10]:
        lines.append(
            f"| {row['task_id']} | {row['teacher_gain_vs_strict']:.3f} | {row['suggested_grammar']} | "
            f"{row['lowering_bias']} | {row['priority_note']} |"
        )
    lines.extend(
        [
            "",
            "## 解釈",
            "",
            "P0は単純なD4/rectangle/lineでは足りない。次は、変更セル数・bbox保存・背景fill条件を利用したobject-role/local-neighborhood grammarを追加する。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    rows = [profile_task(item) for item in load_p0_tasks()]
    rows = sorted(rows, key=lambda r: (-r.teacher_gain_vs_strict, r.task_id))
    grammar_counts = Counter(r.suggested_grammar for r in rows)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "taxonomy_ready",
        "profiled_task_count": len(rows),
        "grammar_counts": dict(grammar_counts),
        "top_rows": [asdict(r) for r in rows],
        "decision": "add object-role/local-neighborhood sparse fill grammar before more ONNX lowering attempts",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: diagnostic taxonomy",
        "leakage_risk": "low: descriptive structure only.",
        "overfitting_risk": "medium: all arc-gen included for diagnosis; future rules require holdout and LB calibration.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "task_taxonomy.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TaskTaxonomy.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    (EXP_DIR / "notes.md").write_text(notes_text(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
