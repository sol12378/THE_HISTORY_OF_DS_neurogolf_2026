from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from statistics import median
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp054_signature_lookup_family_taxonomy"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_TARGETS = ROOT / "experiments" / "exp053_all_task_cost_250_600_inventory" / "task_cost_targets.csv"


@dataclass(frozen=True)
class TaskTaxonomy:
    task_id: int
    priority: str
    route: str
    strict_cost: int
    teacher_cost: int
    gain_to_600: float
    gain_to_250: float
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
    output_shape_pair_count: int
    dominant_output_shapes: str
    dominant_change_pairs: str
    compiler_lane: str
    lowering_bias: str
    first_probe: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def rate(xs: list[bool]) -> float:
    if not xs:
        return 0.0
    return float(sum(1 for x in xs if bool(x)) / len(xs))


def load_signature_tasks() -> list[dict[str, str]]:
    with TASK_TARGETS.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["family"] == "signature_lookup_current"]


def choose_lane(raw: dict[str, Any]) -> tuple[str, str, str]:
    same_shape = raw["same_shape_rate"]
    sparse = raw["sparse_edit_rate"]
    only_fill = raw["only_background_fill_rate"]
    only_delete = raw["only_deletion_rate"]
    bbox_keep = raw["bbox_preserved_rate"]
    med = raw["changed_count_median"]
    shape_pairs = raw["shape_pair_count"]
    output_shapes = raw["output_shape_pair_count"]

    if same_shape > 0.98 and sparse > 0.95 and only_fill > 0.95 and bbox_keep > 0.90 and med <= 8:
        return (
            "L1_static_sparse_background_fill",
            "tiny coordinate program or one small mask; target cost 250-450",
            "object-role/orbit/local-neighborhood fill search",
        )
    if same_shape > 0.98 and sparse > 0.95 and only_fill > 0.80:
        return (
            "L2_local_predicate_sparse_fill",
            "small Conv kernels plus one Where; target cost 350-600",
            "3x3/5x5 neighborhood predicate and color-role search",
        )
    if same_shape > 0.98 and sparse > 0.95 and (only_delete > 0.20 or bbox_keep < 0.50):
        return (
            "L3_object_move_or_erase",
            "object masks with small Gather/Slice; avoid large ScatterND",
            "component correspondence and role-preserving copy/erase search",
        )
    if shape_pairs > 1 or output_shapes > 1:
        return (
            "L4_shape_crop_resize",
            "constant or small shape-conditioned Slice/Gather/Pad; reject full-grid bbox GatherND",
            "shape rule, bbox anchor, fixed output-size crop/resize probe",
        )
    if same_shape > 0.98 and sparse > 0.80:
        return (
            "L5_general_sparse_completion",
            "Reduce/Conv masks before any ScatterND; target cost under 600 only after proof",
            "changed-cell decision tree over color roles and local geometry",
        )
    return (
        "L6_task_specific_or_defer",
        "profile teacher and existing strict artifact before lowering",
        "manual profile; not first-wave compiler target",
    )


def profile_task(item: dict[str, str]) -> TaskTaxonomy:
    task_id = int(item["task_id"])
    task = load_task(task_id)
    examples = examples_for(task, -1)

    same_shape: list[bool] = []
    sparse: list[bool] = []
    only_fill: list[bool] = []
    only_delete: list[bool] = []
    bbox_keep: list[bool] = []
    changed_counts: list[int] = []
    shape_pairs: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter()
    output_shapes: Counter[tuple[int, ...]] = Counter()
    change_pairs: Counter[tuple[int, int]] = Counter()

    for ex in examples:
        x = arr(ex["input"])
        y = arr(ex["output"])
        same = x.shape == y.shape
        same_shape.append(same)
        shape_pairs[(tuple(x.shape), tuple(y.shape))] += 1
        output_shapes[tuple(y.shape)] += 1
        if not same:
            sparse.append(False)
            only_fill.append(False)
            only_delete.append(False)
            bbox_keep.append(False)
            changed_counts.append(-1)
            continue

        changed = x != y
        changed_counts.append(int(changed.sum()))
        sparse.append(bool(np.all(y[~changed] == x[~changed])))
        only_fill.append(bool(np.any(changed)) and bool(np.all(x[changed] == 0)) and bool(np.all(y[changed] != 0)))
        only_delete.append(bool(np.any(changed)) and bool(np.all(x[changed] != 0)) and bool(np.all(y[changed] == 0)))
        bbox_keep.append(bbox(x != 0) == bbox(y != 0))
        for a, b in zip(x[changed].ravel(), y[changed].ravel()):
            change_pairs[(int(a), int(b))] += 1

    valid_changed = [c for c in changed_counts if c >= 0]
    raw = {
        "same_shape_rate": rate(same_shape),
        "sparse_edit_rate": rate(sparse),
        "only_background_fill_rate": rate(only_fill),
        "only_deletion_rate": rate(only_delete),
        "bbox_preserved_rate": rate(bbox_keep),
        "changed_count_median": median(valid_changed) if valid_changed else -1,
        "shape_pair_count": len(shape_pairs),
        "output_shape_pair_count": len(output_shapes),
    }
    lane, lowering, first_probe = choose_lane(raw)

    return TaskTaxonomy(
        task_id=task_id,
        priority=item["priority"],
        route=item["route"],
        strict_cost=int(float(item["strict_cost"])),
        teacher_cost=int(float(item["teacher_cost"])),
        gain_to_600=float(item["gain_to_600"]),
        gain_to_250=float(item["gain_to_250"]),
        teacher_gain_vs_strict=float(item["teacher_gain_vs_strict"]),
        example_count=len(examples),
        same_shape_rate=raw["same_shape_rate"],
        sparse_edit_rate=raw["sparse_edit_rate"],
        only_background_fill_rate=raw["only_background_fill_rate"],
        only_deletion_rate=raw["only_deletion_rate"],
        bbox_preserved_rate=raw["bbox_preserved_rate"],
        changed_count_median=raw["changed_count_median"],
        changed_count_min=min(valid_changed) if valid_changed else -1,
        changed_count_max=max(valid_changed) if valid_changed else -1,
        shape_pair_count=len(shape_pairs),
        output_shape_pair_count=len(output_shapes),
        dominant_output_shapes=json.dumps({str(k): v for k, v in output_shapes.most_common(6)}, ensure_ascii=False),
        dominant_change_pairs=json.dumps({str(k): v for k, v in change_pairs.most_common(6)}, ensure_ascii=False),
        compiler_lane=lane,
        lowering_bias=lowering,
        first_probe=first_probe,
    )


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def notes_text(result: dict[str, Any]) -> str:
    lines = [
        f"# {EXP_ID}",
        "",
        "## 目的",
        "",
        "`signature_lookup_current` 196 taskを、250〜600 cost帯へ落とすための量産compiler laneへ分類する。",
        "",
        "## 結果",
        "",
        f"- profiled tasks: {result['profiled_task_count']}",
        f"- total gain to cost<=600: {result['total_gain_to_600']:.6f}",
        f"- total gain to cost<=250: {result['total_gain_to_250']:.6f}",
        "",
        "## Compiler Lane Summary",
        "",
        "| lane | tasks | gain<=600 | gain<=250 | top tasks |",
        "|---|---:|---:|---:|---|",
    ]
    for row in result["lane_summary"]:
        lines.append(
            f"| {row['compiler_lane']} | {row['task_count']} | {row['gain_to_600_sum']:.3f} | "
            f"{row['gain_to_250_sum']:.3f} | {row['top_tasks']} |"
        )
    lines.extend(
        [
            "",
            "## 解釈",
            "",
            "最大familyの中でも、最初に狙うべきは同shapeで変更が疎な背景fill系である。"
            "これらはfull-grid処理を避け、local predicate / object-role / small maskへ落とせる可能性が高い。",
            "",
            "shape/crop系はgainが大きいが、過去のfull-grid bbox loweringが失敗しているため、"
            "constant shape ruleまたは小さいshape-conditioned Slice/Gatherに限定する。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_signature_tasks()
    rows = [profile_task(item) for item in tasks]
    rows = sorted(rows, key=lambda r: (-r.gain_to_600, -r.gain_to_250, r.task_id))

    lane_acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "compiler_lane": "",
            "task_count": 0,
            "gain_to_600_sum": 0.0,
            "gain_to_250_sum": 0.0,
            "strict_cost_sum": 0,
            "teacher_cost_sum": 0,
            "top_tasks": [],
        }
    )
    for row in rows:
        acc = lane_acc[row.compiler_lane]
        acc["compiler_lane"] = row.compiler_lane
        acc["task_count"] += 1
        acc["gain_to_600_sum"] += row.gain_to_600
        acc["gain_to_250_sum"] += row.gain_to_250
        acc["strict_cost_sum"] += row.strict_cost
        acc["teacher_cost_sum"] += row.teacher_cost
        if len(acc["top_tasks"]) < 10:
            acc["top_tasks"].append(row.task_id)

    lane_summary = []
    for acc in lane_acc.values():
        lane_summary.append(
            {
                "compiler_lane": acc["compiler_lane"],
                "task_count": acc["task_count"],
                "gain_to_600_sum": acc["gain_to_600_sum"],
                "gain_to_250_sum": acc["gain_to_250_sum"],
                "strict_cost_sum": acc["strict_cost_sum"],
                "teacher_cost_sum": acc["teacher_cost_sum"],
                "top_tasks": " ".join(str(x) for x in acc["top_tasks"]),
            }
        )
    lane_summary.sort(key=lambda x: float(x["gain_to_600_sum"]), reverse=True)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "taxonomy_ready",
        "source": str(TASK_TARGETS.relative_to(ROOT)),
        "profiled_task_count": len(rows),
        "total_gain_to_600": sum(r.gain_to_600 for r in rows),
        "total_gain_to_250": sum(r.gain_to_250 for r in rows),
        "lane_counts": dict(Counter(r.compiler_lane for r in rows)),
        "lane_summary": lane_summary,
        "top_rows": [asdict(r) for r in rows[:40]],
        "decision": "First wave compiler should target L2/L1 sparse fill lanes, then shape/crop lanes with strict cost guards.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: diagnostic taxonomy only",
        "leakage_risk": "low: descriptive profiling only; no teacher table is emitted.",
        "overfitting_risk": "medium: all examples are used for taxonomy; future generated rules require holdout and Kaggle calibration.",
        "outputs": {
            "task_taxonomy": "task_taxonomy.csv",
            "lane_summary": "lane_summary.csv",
            "notes": "notes.md",
        },
    }

    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(EXP_DIR / "task_taxonomy.csv", [asdict(r) for r in rows], list(TaskTaxonomy.__dataclass_fields__.keys()))
    write_csv(
        EXP_DIR / "lane_summary.csv",
        lane_summary,
        [
            "compiler_lane",
            "task_count",
            "gain_to_600_sum",
            "gain_to_250_sum",
            "strict_cost_sum",
            "teacher_cost_sum",
            "top_tasks",
        ],
    )
    (EXP_DIR / "notes.md").write_text(notes_text(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
