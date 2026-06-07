from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import load_task  # noqa: E402


EXP_ID = "exp087_small_output_crop_candidate_scan"
EXP_DIR = ROOT / "experiments" / EXP_ID
COST_TARGETS = ROOT / "experiments" / "exp053_all_task_cost_250_600_inventory" / "task_cost_targets.csv"
COMPILER_QUEUE = ROOT / "experiments" / "exp069_karnak_prior_compiler_queue" / "compiler_priority_queue.csv"


def read_rows(path: pathlib.Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return {int(row["task_id"]): row for row in csv.DictReader(f)}


def shape_of(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0])


def task_shape_stats(task_id: int) -> dict[str, Any]:
    task = load_task(task_id)
    examples = task["train"] + task["test"] + task["arc-gen"]
    in_shapes = [shape_of(ex["input"]) for ex in examples]
    out_shapes = [shape_of(ex["output"]) for ex in examples]
    out_areas = [h * w for h, w in out_shapes]
    in_areas = [h * w for h, w in in_shapes]
    hist = Counter(out_shapes)
    return {
        "example_count": len(examples),
        "input_shape_count": len(set(in_shapes)),
        "output_shape_count": len(set(out_shapes)),
        "max_output_area": max(out_areas),
        "min_output_area": min(out_areas),
        "median_output_area": sorted(out_areas)[len(out_areas) // 2],
        "max_input_area": max(in_areas),
        "grid_size_changed_examples": sum(1 for i, o in zip(in_shapes, out_shapes) if i != o),
        "output_shape_hist": ";".join(f"{h}x{w}:{n}" for (h, w), n in sorted(hist.items(), key=lambda x: (x[0][0] * x[0][1], x[0], x[1]))),
    }


def lowerability_bucket(max_area: int, shape_count: int) -> str:
    # exp086 measured 3x3 padded one-hot at 381 and 6x6 at 1461. Since memory
    # scales approximately with output crop area, <=14 cells is the first
    # realistic <=600 proxy band before selector overhead.
    if max_area <= 14 and shape_count <= 4:
        return "P0_600_plausible"
    if max_area <= 14:
        return "P1_area_ok_shape_branch"
    if max_area <= 25 and shape_count <= 4:
        return "P2_1000_plausible"
    if max_area <= 36:
        return "P3_large_crop_like_task365"
    return "P4_not_600_by_area_floor"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    cost_rows = read_rows(COST_TARGETS)
    compiler_rows = read_rows(COMPILER_QUEUE)
    rows: list[dict[str, Any]] = []
    for task_id in range(1, 401):
        stats = task_shape_stats(task_id)
        cost = cost_rows.get(task_id, {})
        compiler = compiler_rows.get(task_id, {})
        route = cost.get("route", "")
        compiler_lane = compiler.get("compiler_lane", "")
        is_cropish = (
            route == "crop_or_resize"
            or compiler_lane == "CROP_SHAPE_COMPILER"
            or stats["grid_size_changed_examples"] > 0
        )
        bucket = lowerability_bucket(stats["max_output_area"], stats["output_shape_count"])
        strict_cost = int(float(cost.get("strict_cost", "0") or 0))
        gain_to_600 = float(cost.get("gain_to_600", "0") or 0)
        gain_to_250 = float(cost.get("gain_to_250", "0") or 0)
        priority = 0.0
        if is_cropish:
            priority += 100.0
        priority += gain_to_600 * 10.0
        priority += max(0.0, 40.0 - stats["max_output_area"])
        priority -= stats["output_shape_count"] * 2.0
        if bucket.startswith("P0"):
            priority += 50.0
        elif bucket.startswith("P1"):
            priority += 25.0
        row = {
            "task_id": task_id,
            "route": route,
            "compiler_lane": compiler_lane,
            "is_cropish": is_cropish,
            "bucket": bucket,
            "priority": round(priority, 6),
            "strict_cost": strict_cost,
            "gain_to_600": round(gain_to_600, 6),
            "gain_to_250": round(gain_to_250, 6),
            **stats,
        }
        rows.append(row)

    rows.sort(key=lambda r: (-float(r["priority"]), int(r["task_id"])))
    fields = [
        "task_id",
        "route",
        "compiler_lane",
        "is_cropish",
        "bucket",
        "priority",
        "strict_cost",
        "gain_to_600",
        "gain_to_250",
        "example_count",
        "input_shape_count",
        "output_shape_count",
        "max_output_area",
        "min_output_area",
        "median_output_area",
        "max_input_area",
        "grid_size_changed_examples",
        "output_shape_hist",
    ]
    with (EXP_DIR / "small_output_candidates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    p0 = [r for r in rows if r["bucket"] == "P0_600_plausible" and r["is_cropish"]]
    p1 = [r for r in rows if r["bucket"] == "P1_area_ok_shape_branch" and r["is_cropish"]]
    top = rows[:20]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "candidate_scan_ready",
        "hypothesis": "exp086のarea floorから、max output area <=14 のcrop/shape taskはcost<=600級の候補になり得る。",
        "task_count": len(rows),
        "p0_cropish_count": len(p0),
        "p1_cropish_count": len(p1),
        "top20": top,
        "decision": "Prioritize P0/P1 cropish tasks for rule probing and Slice/Pad proxy measurement; keep task365 as >600 but still large-reduction candidate.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: scan only",
        "leakage_risk": "low: uses task shapes and existing queue metadata only.",
        "overfitting_risk": "low: no rule or ONNX candidate emitted.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`exp086` のcost実測から、`max_output_area <= 14` のcrop/shape taskを `cost<=600` 候補として抽出する。

## 結果

- task count: `{len(rows)}`
- P0 cropish count: `{len(p0)}`
- P1 cropish count: `{len(p1)}`

## 判断

`task365` は6x6 area floorで600級が厳しいため、大幅削減候補として保持する。一方で、最大出力面積が14セル以下のcropish taskは、`Slice+Pad` proxyで600に入る可能性があるため次のrule probe対象にする。

## Risk

- leakage risk: low。shape統計とqueue metadataのみ。
- overfitting risk: low。提出候補は生成していない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
