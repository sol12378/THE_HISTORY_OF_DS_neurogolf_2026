from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import date
from typing import Any

import numpy as np
from onnx import helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (
    BaseTask,
    Candidate,
    evaluate_candidate,
    examples_for,
    load_base_tasks,
    load_neurogolf_utils,
    load_task,
    make_model,
    point,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp043_fixed_dsl_hit_lowering"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
OVERLAY_EXP040 = ROOT / "experiments" / "exp040_deeper_fullarc_safe_bypass"
OVERLAY_EXP = ROOT / "experiments" / "exp041_task145_deeper_mul_chain"
EXP042_HITS = ROOT / "experiments" / "exp042_freeop_dag_search_top200" / "program_hits.csv"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
AUTHORITATIVE_BASE_SCORE = 6480.302477938763


@dataclass
class LoweringRow:
    task_id: int
    program: str
    baseline_cost: int
    candidate_cost: int | str
    baseline_points: float
    candidate_points: float | str
    file_bytes: int | str
    validation_status: str
    status: str
    reason: str
    sha256: str


def load_current_best_tasks() -> dict[int, BaseTask]:
    base = load_base_tasks(BASE_EXP)
    for overlay_exp in [OVERLAY_EXP040, OVERLAY_EXP]:
        if not (overlay_exp / "selected_manifest.csv").exists():
            continue
        with zipfile.ZipFile(overlay_exp / "submission.zip") as zf:
            overlay_raws = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}
        with (overlay_exp / "selected_manifest.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "improved":
                    continue
                task_id = int(row["task_id"])
                if "candidate_cost" in row:
                    cost = int(float(row["candidate_cost"]))
                    points = float(row["candidate_points"])
                else:
                    cost = int(float(row["cost"]))
                    points = float(row["local_points"])
                base[task_id] = replace(
                    base[task_id],
                    cost=cost,
                    points=points,
                    source=f"{EXP_ID}_base_overlay_{overlay_exp.name}",
                    template_name=row["template_name"],
                    raw=overlay_raws[task_id],
                )
    return base


def grid_shape_set(task_id: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    task = load_task(task_id)
    inputs = {(len(ex["input"]), len(ex["input"][0])) for ex in examples_for(task, -1)}
    outputs = {(len(ex["output"]), len(ex["output"][0])) for ex in examples_for(task, -1)}
    return inputs, outputs


def slice_input_node(ih: int, iw: int) -> tuple[list[Any], list[Any], str]:
    initializers = [
        numpy_helper.from_array(np.asarray([0, 0, 0, 0], dtype=np.int64), "starts"),
        numpy_helper.from_array(np.asarray([1, 10, ih, iw], dtype=np.int64), "ends"),
        numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "axes"),
    ]
    nodes = [helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["x0"])]
    return nodes, initializers, "x0"


def pad_to_30(nodes: list[Any], initializers: list[Any], current: str, oh: int, ow: int) -> None:
    nodes.append(helper.make_node("Pad", [current], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30 - oh, 30 - ow], value=0.0))


def build_fixed_program_candidate(task_id: int, program: str, route: str) -> Candidate:
    input_shapes, output_shapes = grid_shape_set(task_id)
    if len(input_shapes) != 1 or len(output_shapes) != 1:
        return Candidate(task_id, f"dsl_{program}", route, None, "skipped", f"variable shapes input={sorted(input_shapes)} output={sorted(output_shapes)}")
    ih, iw = next(iter(input_shapes))
    oh, ow = next(iter(output_shapes))
    nodes, initializers, current = slice_input_node(ih, iw)

    if program == "rot180":
        initializers.extend(
            [
                numpy_helper.from_array(np.arange(ih - 1, -1, -1, dtype=np.int64), "idx_h"),
                numpy_helper.from_array(np.arange(iw - 1, -1, -1, dtype=np.int64), "idx_w"),
            ]
        )
        nodes.append(helper.make_node("Gather", [current, "idx_h"], ["x1"], axis=2))
        nodes.append(helper.make_node("Gather", ["x1", "idx_w"], ["x2"], axis=3))
        current = "x2"
    elif program == "rot90_ccw":
        initializers.append(numpy_helper.from_array(np.arange(iw - 1, -1, -1, dtype=np.int64), "idx_w"))
        nodes.append(helper.make_node("Transpose", [current], ["x1"], perm=[0, 1, 3, 2]))
        nodes.append(helper.make_node("Gather", ["x1", "idx_w"], ["x2"], axis=2))
        current = "x2"
    elif program == "crop_tl":
        if oh > ih or ow > iw:
            return Candidate(task_id, f"dsl_{program}", route, None, "skipped", "output larger than input")
        initializers.extend(
            [
                numpy_helper.from_array(np.asarray([0, 0, 0, 0], dtype=np.int64), "crop_starts"),
                numpy_helper.from_array(np.asarray([1, 10, oh, ow], dtype=np.int64), "crop_ends"),
                numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "crop_axes"),
            ]
        )
        nodes.append(helper.make_node("Slice", [current, "crop_starts", "crop_ends", "crop_axes"], ["x1"]))
        current = "x1"
    elif program == "crop_tr":
        if oh > ih or ow > iw:
            return Candidate(task_id, f"dsl_{program}", route, None, "skipped", "output larger than input")
        c0 = iw - ow
        initializers.extend(
            [
                numpy_helper.from_array(np.asarray([0, 0, 0, c0], dtype=np.int64), "crop_starts"),
                numpy_helper.from_array(np.asarray([1, 10, oh, iw], dtype=np.int64), "crop_ends"),
                numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "crop_axes"),
            ]
        )
        nodes.append(helper.make_node("Slice", [current, "crop_starts", "crop_ends", "crop_axes"], ["x1"]))
        current = "x1"
    elif program.startswith("nearest_upscale_"):
        scale = int(program.rsplit("_", 1)[1])
        if ih * scale != oh or iw * scale != ow:
            return Candidate(task_id, f"dsl_{program}", route, None, "skipped", f"shape mismatch for scale {scale}")
        initializers.append(numpy_helper.from_array(np.asarray([1, 1, scale, scale], dtype=np.int64), "repeats"))
        nodes.append(helper.make_node("Tile", [current, "repeats"], ["x1"]))
        current = "x1"
    else:
        return Candidate(task_id, f"dsl_{program}", route, None, "skipped", "program not in fixed lowering set")

    pad_to_30(nodes, initializers, current, oh, ow)
    raw = make_model(nodes, initializers, f"{EXP_ID}_{program}")
    return Candidate(task_id, f"dsl_{program}", route, raw, "generated", f"input_shape={(ih, iw)},output_shape={(oh, ow)}")


def read_exp042_hits() -> list[dict[str, str]]:
    with EXP042_HITS.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def selected_row(task_id: int, base: BaseTask, row: LoweringRow | None, raw: bytes) -> dict[str, Any]:
    if row is None:
        return {
            "task_id": task_id,
            "source": base.source,
            "template_name": base.template_name,
            "route": base.route,
            "cost": base.cost,
            "local_points": base.points,
            "file_bytes": len(base.raw),
            "status": "baseline",
            "reason": "no fixed DSL lowering gain",
            "sha256": sha256(base.raw),
        }
    return {
        "task_id": task_id,
        "source": f"{EXP_ID}_{row.program}",
        "template_name": f"dsl_{row.program}",
        "route": base.route,
        "cost": row.candidate_cost,
        "local_points": row.candidate_points,
        "file_bytes": len(raw),
        "status": "improved",
        "reason": row.reason,
        "sha256": sha256(raw),
    }


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_current_best_tasks()
    final_raw = {task_id: base.raw for task_id, base in base_tasks.items()}
    rows: list[LoweringRow] = []
    improved: dict[int, tuple[LoweringRow, bytes]] = {}
    fixed_programs = {"rot180", "rot90_ccw", "crop_tl", "crop_tr", "nearest_upscale_3"}

    for hit in read_exp042_hits():
        task_id = int(hit["task_id"])
        program = hit["program"]
        if program not in fixed_programs:
            rows.append(
                LoweringRow(
                    task_id,
                    program,
                    base_tasks[task_id].cost,
                    "",
                    base_tasks[task_id].points,
                    "",
                    "",
                    "not_run",
                    "skipped",
                    "requires variable-shape/object lowering",
                    "",
                )
            )
            continue
        base = base_tasks[task_id]
        candidate = build_fixed_program_candidate(task_id, program, base.route)
        eval_row, raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
        row = LoweringRow(
            task_id=eval_row.task_id,
            program=program,
            baseline_cost=eval_row.baseline_cost,
            candidate_cost=eval_row.candidate_cost,
            baseline_points=eval_row.baseline_points,
            candidate_points=eval_row.candidate_points,
            file_bytes=eval_row.file_bytes,
            validation_status=eval_row.validation_status,
            status=eval_row.status,
            reason=eval_row.reason,
            sha256=eval_row.sha256,
        )
        rows.append(row)
        if raw is not None and row.status == "improved" and isinstance(row.candidate_cost, int):
            improved[task_id] = (row, raw)
            final_raw[task_id] = raw

    write_zip(OUTPUT_ZIP, final_raw)
    selected_rows = [selected_row(task_id, base_tasks[task_id], improved.get(task_id, (None, b""))[0], final_raw[task_id]) for task_id in sorted(base_tasks)]

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(LoweringRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])

    selected_fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected_fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    accepted_delta = sum(
        float(item["local_points"]) - base_tasks[int(item["task_id"])].points
        for item in selected_rows
        if item["status"] == "improved"
    )
    baseline_score = AUTHORITATIVE_BASE_SCORE
    new_score = baseline_score + accepted_delta
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "target_reached_6500" if new_score >= 6500 else ("improved_below_target" if improved else "no_gain"),
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "overlay_exps": [str(OVERLAY_EXP040.relative_to(ROOT)), str(OVERLAY_EXP.relative_to(ROOT))],
        "baseline_local_estimate": baseline_score,
        "new_local_estimate": new_score,
        "delta": new_score - baseline_score,
        "gap_to_6500": 6500.0 - new_score,
        "gap_to_7000": 7000.0 - new_score,
        "gap_to_7400": 7400.0 - new_score,
        "gap_to_7600": 7600.0 - new_score,
        "gap_to_7700": 7700.0 - new_score,
        "candidate_status_counts": dict(Counter(row.status for row in rows)),
        "improved_task_count": len(improved),
        "improved_tasks": sorted(improved),
        "top_improvements": [asdict(improved[task_id][0]) for task_id in sorted(improved, key=lambda k: float(improved[k][0].candidate_points) - improved[k][0].baseline_points, reverse=True)],
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "accounting_note": "Total score uses exp041 authoritative best plus accepted deltas; selected_manifest is a convenience bundle manifest.",
        "leakage_risk": "low: candidates are explicit DSL transforms and pass train/test/all arc-gen; no signature lookup.",
        "overfitting_risk": "low-to-medium: fixed-shape transform families are simple, but still ARC train-task specific.",
        "decision": "adopt if improved; continue with variable-shape/object DSL lowering for exp042 skipped hits.",
    }
    with (EXP_DIR / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    notes = f"""# {EXP_ID}

## 仮説

exp042のproxy hitのうち固定shapeのflip/rot/crop/upscale系は、実ONNXへloweringしても既存artifactより低costになり、proxyではなくsubmit候補bundleの改善として採用できる。

## 実験

- base: exp040 + exp041 task145 overlay
- 対象: exp042 hit 9件
- 実装対象: `rot180`, `rot90_ccw`, `crop_tl`, `crop_tr`, `nearest_upscale_3`
- 除外: variable-shape `flip_h/flip_v` と `bbox_nonzero` は次段のobject/shape compilerへ回した
- validation: train + test + all arc-gen

## 結果

- baseline local estimate: {baseline_score:.6f}
- new local estimate: {new_score:.6f}
- delta: {new_score - baseline_score:.6f}
- improved tasks: {sorted(improved)}
- candidate status counts: {dict(Counter(row.status for row in rows))}

## 解釈

固定shape DSL loweringが実costで改善できるかを検証した。改善が小さい/ゼロの場合でも、proxyとofficial costの乖離をguardrailとして扱い、次はvariable-shape/object loweringに集中する。

## リスク

- leakage risk: 低。signature lookupは使わず、明示的DSL変換のみ。
- overfitting risk: 低〜中。all arc-gen通過だが、task-specificな固定変換である。

## 次

1. 改善taskを次baseに採用する。
2. variable-shape flip用の幅/高さ検出mask compilerを作る。
3. task031 `bbox_nonzero` の低cost動的bbox loweringを試す。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
