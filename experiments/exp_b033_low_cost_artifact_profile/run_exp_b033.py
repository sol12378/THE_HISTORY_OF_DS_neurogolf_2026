from __future__ import annotations

import csv
import json
import pathlib
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import onnx

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


EXP_ID = "exp_b033_low_cost_artifact_profile"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"


@dataclass(frozen=True)
class ArtifactProfile:
    task_id: int
    route: str
    source: str
    status: str
    cost: int
    local_points: float
    file_bytes: int
    node_count: int
    initializer_count: int
    initializer_params: int
    op_counts: str
    first_ops: str
    compiler_lesson: str


def read_manifest() -> dict[int, dict[str, str]]:
    with (BASE_EXP / "selected_manifest.csv").open(encoding="utf-8", newline="") as f:
        return {int(row["task_id"]): row for row in csv.DictReader(f)}


def compiler_lesson(row: dict[str, str], op_counts: Counter[str], init_params: int, node_count: int) -> str:
    route = row.get("route", "")
    cost = int(float(row["cost"]))
    if cost <= 250:
        if node_count <= 3 and ("Identity" in op_counts or "Gather" in op_counts or "Slice" in op_counts):
            return "tiny fixed transform/crop; compiler should prefer static Slice/Gather/Identity where rule shape is fixed"
        if op_counts.get("Conv", 0) <= 1 and init_params <= 120:
            return "small color/map kernel; compiler can use 1x1 Conv or tiny fixed weights"
        return "already ultra-low; inspect as target template"
    if cost <= 600:
        if "Conv" in op_counts and node_count <= 5:
            return "near-250 small Conv pattern; useful for color-role or local mask lowering"
        if "Slice" in op_counts or "Gather" in op_counts:
            return "near-250 shape/index pattern; useful for crop/object-anchor lowering"
        return "near-250 minimal graph; profile manually"
    if cost <= 2000:
        if route == "crop_or_resize":
            return "low-cost crop/resize artifact; mine for object-anchor crop compiler"
        if "Conv" in op_counts:
            return "low-cost Conv artifact; mine kernel/channel pattern"
        return "low-cost non-crop artifact; possible sparse rule template"
    return "not low-cost exemplar"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    rows: list[ArtifactProfile] = []
    with zipfile.ZipFile(BASE_EXP / "submission.zip") as zf:
        for task_id, row in sorted(manifest.items()):
            raw = zf.read(f"task{task_id:03d}.onnx")
            model = onnx.load_from_string(raw)
            op_counts = Counter(node.op_type for node in model.graph.node)
            init_params = 0
            for init in model.graph.initializer:
                size = 1
                for dim in init.dims:
                    size *= int(dim)
                init_params += size
            first_ops = " ".join(node.op_type for node in list(model.graph.node)[:8])
            rows.append(
                ArtifactProfile(
                    task_id=task_id,
                    route=row.get("route", ""),
                    source=row.get("source", ""),
                    status=row.get("status", ""),
                    cost=int(float(row["cost"])),
                    local_points=float(row["local_points"]),
                    file_bytes=len(raw),
                    node_count=len(model.graph.node),
                    initializer_count=len(model.graph.initializer),
                    initializer_params=init_params,
                    op_counts=json.dumps(dict(op_counts), sort_keys=True),
                    first_ops=first_ops,
                    compiler_lesson=compiler_lesson(row, op_counts, init_params, len(model.graph.node)),
                )
            )

    rows_sorted = sorted(rows, key=lambda r: (r.cost, r.task_id))
    with (EXP_DIR / "artifact_profile.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ArtifactProfile.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows_sorted])

    buckets = {
        "cost_le_250": [r for r in rows if r.cost <= 250],
        "cost_le_600": [r for r in rows if r.cost <= 600],
        "cost_le_2000": [r for r in rows if r.cost <= 2000],
    }
    lesson_summary: dict[str, int] = dict(Counter(r.compiler_lesson for r in rows if r.cost <= 2000).most_common())
    route_summary: dict[str, dict[str, Any]] = {}
    by_route: dict[str, list[ArtifactProfile]] = defaultdict(list)
    for row in rows:
        by_route[row.route].append(row)
    for route, items in sorted(by_route.items()):
        route_summary[route] = {
            "tasks": len(items),
            "cost_le_250": sum(1 for r in items if r.cost <= 250),
            "cost_le_600": sum(1 for r in items if r.cost <= 600),
            "median_cost": sorted(r.cost for r in items)[len(items) // 2],
            "top_low_cost_tasks": [r.task_id for r in sorted(items, key=lambda r: (r.cost, r.task_id))[:10]],
        }
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "profile_ready",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "total_tasks": len(rows),
        "cost_le_250_count": len(buckets["cost_le_250"]),
        "cost_le_600_count": len(buckets["cost_le_600"]),
        "cost_le_2000_count": len(buckets["cost_le_2000"]),
        "low_cost_task_ids": [r.task_id for r in buckets["cost_le_250"]],
        "near_cost_task_ids": [r.task_id for r in buckets["cost_le_600"]],
        "lesson_summary": lesson_summary,
        "route_summary": route_summary,
        "top_30_low_cost_profiles": [asdict(r) for r in rows_sorted[:30]],
        "decision": "Use low-cost artifacts as compiler templates. Focus next on crop/object-anchor patterns with Slice/Gather and small Conv/local-mask patterns; do not continue unrolled region fill.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: diagnostic profile only",
        "leakage_risk": "low: profiles current submitted-safe artifacts only.",
        "overfitting_risk": "low: no candidate generation; future compiler templates still require full-arc and LB calibration.",
        "outputs": {"artifact_profile": "artifact_profile.csv"},
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

current submit-safe best `exp_b025` から、cost<=250/600に近いartifactのONNX構造を抽出し、compilerが真似るべき低cost patternを作る。

## 結果

- cost<=250: `{len(buckets["cost_le_250"])}` tasks
- cost<=600: `{len(buckets["cost_le_600"])}` tasks
- cost<=2000: `{len(buckets["cost_le_2000"])}` tasks

## 判断

低cost exemplarはstatic Slice/Gather/Identity、小Conv、低param shape/index patternに寄っている。次はobject-anchor cropとsmall Conv/local-mask compilerを優先し、region-fill unrollは続けない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
