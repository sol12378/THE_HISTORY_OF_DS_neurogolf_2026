from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


EXP_ID = "exp069_karnak_prior_compiler_queue"
EXP_DIR = ROOT / "experiments" / EXP_ID
COST_TARGETS = ROOT / "experiments" / "exp053_all_task_cost_250_600_inventory" / "task_cost_targets.csv"
SIGNATURE_TAXONOMY = ROOT / "experiments" / "exp054_signature_lookup_family_taxonomy" / "task_taxonomy.csv"
KARNAK_CSV = ROOT / "experiments" / "exp067_public_notebook_utilization_audit" / "karnak_dataset" / "arc_primitives.csv"
CURRENT_MANIFEST = ROOT / "experiments" / "exp068_seddik_style_strict_scalarization" / "selected_manifest.csv"


@dataclass(frozen=True)
class CompilerQueueRow:
    task_id: int
    family: str
    route: str
    signature_lane: str
    primary_category: str
    transformations: str
    estimated_complexity: int
    grid_size_changed: bool
    current_cost: int
    current_points: float
    strict_cost: int
    gain_to_600: float
    gain_to_250: float
    teacher_gain_vs_strict: float
    compiler_lane: str
    compiler_priority_score: float
    next_action: str
    lowering_bias: str
    risk_note: str


def read_csv_by_task(path: pathlib.Path, task_key: str = "task_id") -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get(task_key, "")
            digits = "".join(ch for ch in raw if ch.isdigit())
            if digits:
                rows[int(digits)] = row
    return rows


def split_transforms(text: str) -> set[str]:
    return {part.strip() for part in text.split("|") if part.strip()}


def boolish(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def choose_compiler_lane(family: str, route: str, signature_lane: str, category: str, transforms: set[str], grid_changed: bool) -> tuple[str, str, str]:
    if "Cropping" in transforms or grid_changed or route == "crop_or_resize" or "shape_crop" in signature_lane:
        return (
            "CROP_SHAPE_COMPILER",
            "dynamic bbox Slice/Gather first; reject full-grid GatherND; prefer existing low-cost crop artifacts as templates",
            "bbox/generalization risk: require all-arc pass and cost guard",
        )
    if "Filling Regions" in transforms or "Flood Fill" in transforms:
        if "Object Detection" in transforms or "Color Mapping" in transforms or route == "sparse_edit_or_object_completion":
            return (
                "LOCAL_PREDICATE_FILL_COMPILER",
                "bbox-local 3x3/5x5 predicates, color-role counts, sparse one-hot writeback",
                "medium: predicate tree can overfit; require feature profile and all-arc pass",
            )
        return (
            "REGION_FILL_COMPILER",
            "closed-form masks/prefix or component artifact mining; reject naive flood unroll",
            "high cost risk: naive full-grid fill is too expensive",
        )
    if "Line Extrapolation" in transforms or "Symmetry Completion" in transforms or "Reflection" in transforms:
        return (
            "SYMMETRY_LINE_COMPILER",
            "D4/line orbit masks, diagonal basis, small Gather/Where writeback",
            "medium: orientation/class selection must be nonlookup",
        )
    if "Translation" in transforms or "Shifting" in transforms or "Gravity" in transforms:
        return (
            "OBJECT_MOVE_COMPILER",
            "component bbox extraction, anchor offset, sparse copy/erase",
            "medium-high: dynamic object selection may be expensive",
        )
    if "Color Mapping" in transforms or category == "Color_and_Logical":
        return (
            "COLOR_ROLE_COMPILER",
            "small one-hot color-role map or scalar compare chain; reject dynamic MatMul lookup",
            "low-medium: hidden color roles must generalize",
        )
    if "Tiling" in transforms or "Magnification" in transforms:
        return (
            "TILE_SCALE_COMPILER",
            "static Slice/Gather/Tile only with cost guard; prefer fixed shapes",
            "medium: full-grid Tile often loses on cost",
        )
    if family == "signature_lookup_current":
        return (
            "LOOKUP_COMPRESSION_COMPILER",
            "mine teacher as oracle, compress to explainable decision rule before ONNX",
            "high: raw lookup/teacher artifact is not submit-safe",
        )
    return (
        "GENERAL_DSL_SYNTHESIS",
        "bounded DSL/DAG search with cost-aware lowering guardrails",
        "unknown: route after feature profiling",
    )


def priority_score(gain_to_600: float, gain_to_250: float, teacher_gain: float, current_cost: int, complexity: int, lane: str) -> float:
    score = gain_to_250 * 4.0 + gain_to_600 * 2.0 + max(0.0, teacher_gain) * 1.5
    if current_cost > 100000:
        score += 3.0
    elif current_cost > 50000:
        score += 1.5
    if complexity <= 4:
        score += 1.5
    elif complexity >= 8:
        score -= 1.0
    if lane in {"LOCAL_PREDICATE_FILL_COMPILER", "CROP_SHAPE_COMPILER"}:
        score += 1.0
    if lane == "LOOKUP_COMPRESSION_COMPILER":
        score -= 0.5
    return score


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    cost_rows = read_csv_by_task(COST_TARGETS)
    sig_rows = read_csv_by_task(SIGNATURE_TAXONOMY)
    karnak_rows = read_csv_by_task(KARNAK_CSV, "Task_ID")
    current_rows = read_csv_by_task(CURRENT_MANIFEST)

    queue: list[CompilerQueueRow] = []
    for task_id, cost in sorted(cost_rows.items()):
        k = karnak_rows.get(task_id, {})
        sig = sig_rows.get(task_id, {})
        cur = current_rows.get(task_id, {})
        transforms = split_transforms(k.get("All_Used_Transformations", ""))
        family = cost.get("family", "")
        route = cost.get("route", "")
        signature_lane = sig.get("lane", "")
        category = k.get("Primary_Category", "")
        complexity = int(float(k.get("Estimated_Complexity", 9) or 9))
        grid_changed = boolish(k.get("Grid_Size_Changed", "False"))
        compiler_lane, lowering_bias, risk_note = choose_compiler_lane(family, route, signature_lane, category, transforms, grid_changed)
        current_cost = int(float(cur.get("cost", cost.get("strict_cost", 0)) or 0))
        current_points = float(cur.get("local_points", cost.get("strict_points", 0.0)) or 0.0)
        gain_to_600 = float(cost.get("gain_to_600", 0.0) or 0.0)
        gain_to_250 = float(cost.get("gain_to_250", 0.0) or 0.0)
        teacher_gain = float(cost.get("teacher_gain_vs_strict", 0.0) or 0.0)
        score = priority_score(gain_to_600, gain_to_250, teacher_gain, current_cost, complexity, compiler_lane)
        next_action = "profile examples and synthesize nonlookup rule"
        if compiler_lane == "CROP_SHAPE_COMPILER":
            next_action = "mine low-cost crop artifacts, then compile bounded bbox Slice/Gather variants"
        elif compiler_lane == "LOCAL_PREDICATE_FILL_COMPILER":
            next_action = "build changed-cell feature profile, then synthesize small predicate tree"
        elif compiler_lane == "SYMMETRY_LINE_COMPILER":
            next_action = "derive orbit/class selector, then lower with sparse D4/line masks"
        elif compiler_lane == "REGION_FILL_COMPILER":
            next_action = "seek closed-form component masks; avoid unrolled flood fill"
        queue.append(
            CompilerQueueRow(
                task_id=task_id,
                family=family,
                route=route,
                signature_lane=signature_lane,
                primary_category=category,
                transformations=" | ".join(sorted(transforms)),
                estimated_complexity=complexity,
                grid_size_changed=grid_changed,
                current_cost=current_cost,
                current_points=current_points,
                strict_cost=int(float(cost.get("strict_cost", 0) or 0)),
                gain_to_600=gain_to_600,
                gain_to_250=gain_to_250,
                teacher_gain_vs_strict=teacher_gain,
                compiler_lane=compiler_lane,
                compiler_priority_score=score,
                next_action=next_action,
                lowering_bias=lowering_bias,
                risk_note=risk_note,
            )
        )

    queue.sort(key=lambda r: r.compiler_priority_score, reverse=True)
    with (EXP_DIR / "compiler_priority_queue.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CompilerQueueRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in queue])

    lane_summary: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[CompilerQueueRow]] = defaultdict(list)
    for row in queue:
        grouped[row.compiler_lane].append(row)
    for lane, rows in grouped.items():
        lane_summary[lane] = {
            "tasks": len(rows),
            "gain_to_600": sum(r.gain_to_600 for r in rows),
            "gain_to_250": sum(r.gain_to_250 for r in rows),
            "top_tasks": [r.task_id for r in rows[:12]],
            "top_transformations": dict(Counter(t for r in rows for t in split_transforms(r.transformations)).most_common(10)),
        }
    with (EXP_DIR / "compiler_lane_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["compiler_lane", "tasks", "gain_to_600", "gain_to_250", "top_tasks", "top_transformations"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for lane, item in sorted(lane_summary.items(), key=lambda kv: kv[1]["gain_to_250"], reverse=True):
            writer.writerow(
                {
                    "compiler_lane": lane,
                    "tasks": item["tasks"],
                    "gain_to_600": item["gain_to_600"],
                    "gain_to_250": item["gain_to_250"],
                    "top_tasks": " ".join(f"{t:03d}" for t in item["top_tasks"]),
                    "top_transformations": json.dumps(item["top_transformations"], ensure_ascii=False),
                }
            )

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "queue_ready",
        "tasks": len(queue),
        "top20": [asdict(row) for row in queue[:20]],
        "lane_summary": lane_summary,
        "recommended_next_experiment": {
            "exp_id": "exp070_local_predicate_fill_compiler_priority_batch",
            "target_lane": "LOCAL_PREDICATE_FILL_COMPILER",
            "reason": "large gain, high overlap with Object Detection/Color Mapping/Filling Regions, and exp066 proved bbox-local fill can transfer to LB when explicit.",
            "target_tasks": [row.task_id for row in queue if row.compiler_lane == "LOCAL_PREDICATE_FILL_COMPILER"][:10],
            "acceptance": "at least one full-arc exact nonlookup rule lowered to ONNX with cost improvement over exp068 base; submit if improved.",
        },
        "decision": "Use this queue to choose compiler work. Prefer local predicate fill first, crop/shape second, symmetry/line third. Apply Seddik-style post-pass after any accepted bundle.",
        "leakage_risk": "low: queue only; Karnak descriptions are priors, not labels.",
        "overfitting_risk": "low for queue; future rules require full arc-gen and LB calibration.",
        "outputs": {
            "compiler_priority_queue": "compiler_priority_queue.csv",
            "compiler_lane_summary": "compiler_lane_summary.csv",
        },
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

Karnak task description libraryを、exp053 cost target queue・exp054 signature lane・exp068 current costへjoinし、cost 250〜600化のcompiler優先順位を作る。

## 結果

- queue tasks: {len(queue)}
- top lane by gain_to_250: {max(lane_summary.items(), key=lambda kv: kv[1]["gain_to_250"])[0]}
- recommended next: `exp070_local_predicate_fill_compiler_priority_batch`

## 解釈

Karnakのdescriptionは直接の正解ではなく、探索順序を決めるpriorとして使う。`Object Detection`, `Color Mapping`, `Filling Regions` が多いので、`exp066` task020型のbbox-local/color-role fill compilerを横展開する価値が高い。

## Decision

次はLOCAL_PREDICATE_FILL_COMPILER上位taskに対し、changed-cell feature profileから小decision treeを合成する。full-arc exactかつcost改善したら、exp068 baseへ差し替えてSeddik post-pass後に提出する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
