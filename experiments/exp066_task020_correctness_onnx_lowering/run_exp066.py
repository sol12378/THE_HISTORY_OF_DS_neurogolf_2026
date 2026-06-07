from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import date
from typing import Any

import numpy as np
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    evaluate_candidate,
    load_routes,
    load_neurogolf_utils,
    make_model,
    point,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp066_task020_correctness_onnx_lowering"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp005_top_cost_rewrite_strict"
TARGET_TASK = 20
STRICT_SEED_SCORE = 6282.230228092811
OUTPUT_ZIP = EXP_DIR / "submission.zip"


@dataclass(frozen=True)
class EvalRow:
    task_id: int
    variant: str
    baseline_cost: int
    candidate_cost: int | str
    baseline_points: float
    candidate_points: float | str
    file_bytes: int | str
    validation_status: str
    status: str
    reason: str
    sha256: str


def init_i(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.int64), name)


def init_f(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.float32), name)


def row_cost_points(row: dict[str, str]) -> tuple[int, float]:
    if row.get("cost", "") != "":
        cost = int(float(row["cost"]))
    elif row.get("new_cost", "") != "":
        cost = int(float(row["new_cost"]))
    elif row.get("candidate_cost", "") != "":
        cost = int(float(row["candidate_cost"]))
    else:
        raise KeyError(row)
    if row.get("local_points", "") != "":
        pts = float(row["local_points"])
    elif row.get("new_points", "") != "":
        pts = float(row["new_points"])
    elif row.get("candidate_points", "") != "":
        pts = float(row["candidate_points"])
    else:
        pts = point(cost)
    return cost, pts


def load_exp_tasks(exp_dir: pathlib.Path) -> dict[int, BaseTask]:
    manifest_path = next((p for p in [exp_dir / "selected_manifest.csv", exp_dir / "rewrite_manifest.csv"] if p.exists()), None)
    if manifest_path is None:
        raise FileNotFoundError(f"no manifest in {exp_dir}")
    routes = load_routes()
    with zipfile.ZipFile(exp_dir / "submission.zip") as zf:
        raws = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}
    out: dict[int, BaseTask] = {}
    with manifest_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            task_id = int(row["task_id"])
            cost, pts = row_cost_points(row)
            out[task_id] = BaseTask(
                task_id=task_id,
                cost=cost,
                points=pts,
                source=row.get("source", exp_dir.name),
                template_name=row.get("template_name", row.get("variant", "")),
                route=row.get("route", routes.get(task_id, "")),
                raw=raws[task_id],
            )
    return out


def group_mask(points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((1, 1, 5, 5), dtype=np.float32)
    for r, c in points:
        mask[0, 0, r, c] = 1.0
    return mask


def build_task020_rule_candidate(base: BaseTask) -> Candidate:
    nodes: list[Any] = []
    inits: list[Any] = [
        init_i("ch_starts", np.asarray([0, 1, 0, 0])),
        init_i("ch_ends", np.asarray([1, 10, 30, 30])),
        init_i("axes4", np.asarray([0, 1, 2, 3])),
        init_i("range5", np.arange(5)),
        init_i("range10", np.arange(10).reshape(1, 10, 1, 1)),
        init_i("one_i", np.asarray([1])),
        init_i("zero_i", np.asarray([0])),
        init_f("zero_f", np.asarray([0.0])),
        init_f("one_f", np.asarray([1.0])),
        init_f("ten_f", np.asarray([10.0])),
        init_f("five_f", np.asarray([5.0])),
        init_f("neg100_f", np.asarray([-100.0])),
        init_f("range30_f", np.arange(30, dtype=np.float32).reshape(1, 30)),
        init_i("range30_i", np.arange(30, dtype=np.int64).reshape(1, 30)),
        init_f("corners_mask", group_mask([(0, 0), (0, 4), (4, 0), (4, 4)])),
        init_f("edge_mask", group_mask([(0, 2), (2, 0), (2, 4), (4, 2)])),
        init_f("inner_mask", group_mask([(1, 1), (1, 3), (3, 1), (3, 3)])),
        init_f("center_mask", group_mask([(2, 2)])),
        init_i("corners_r", np.asarray([0, 0, 4, 4])),
        init_i("corners_c", np.asarray([0, 4, 0, 4])),
        init_i("edge_r", np.asarray([0, 2, 2, 4])),
        init_i("edge_c", np.asarray([2, 0, 4, 2])),
        init_i("inner_r", np.asarray([1, 1, 3, 3])),
        init_i("inner_c", np.asarray([1, 3, 1, 3])),
    ]

    # Dynamic bbox over non-background channels.
    nodes.append(helper.make_node("Slice", ["input", "ch_starts", "ch_ends", "axes4"], ["fg_ch"]))
    nodes.append(helper.make_node("ReduceSum", ["fg_ch"], ["fg_hw"], axes=[1], keepdims=0))
    nodes.append(helper.make_node("ReduceSum", ["fg_hw"], ["row_count"], axes=[2], keepdims=0))
    nodes.append(helper.make_node("ReduceSum", ["fg_hw"], ["col_count"], axes=[1], keepdims=0))
    nodes.append(helper.make_node("Greater", ["row_count", "zero_f"], ["row_has"]))
    nodes.append(helper.make_node("Greater", ["col_count", "zero_f"], ["col_has"]))
    nodes.append(helper.make_node("Cast", ["row_has"], ["row_has_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Cast", ["col_has"], ["col_has_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("ArgMax", ["row_has_f"], ["r0_2d"], axis=1, keepdims=1))
    nodes.append(helper.make_node("ArgMax", ["col_has_f"], ["c0_2d"], axis=1, keepdims=1))
    nodes.append(helper.make_node("Squeeze", ["r0_2d"], ["r0"], axes=[0, 1]))
    nodes.append(helper.make_node("Squeeze", ["c0_2d"], ["c0"], axes=[0, 1]))
    nodes.append(helper.make_node("Add", ["range5", "r0"], ["crop_rows"]))
    nodes.append(helper.make_node("Add", ["range5", "c0"], ["crop_cols"]))
    nodes.append(helper.make_node("Gather", ["input", "crop_rows"], ["crop_rows_g"], axis=2))
    nodes.append(helper.make_node("Gather", ["crop_rows_g", "crop_cols"], ["crop"], axis=3))
    nodes.append(helper.make_node("Slice", ["crop", "ch_starts", "ch_ends", "axes4"], ["crop_colors"]))

    # Color scoring from exp065: ignore center-only singleton colors.
    for name, mask in [
        ("corners", "corners_mask"),
        ("edge", "edge_mask"),
        ("inner", "inner_mask"),
        ("center", "center_mask"),
    ]:
        nodes.append(helper.make_node("Mul", ["crop_colors", mask], [f"{name}_vals"]))
        nodes.append(helper.make_node("ReduceSum", [f"{name}_vals"], [f"{name}_count"], axes=[2, 3], keepdims=0))
    nodes.append(helper.make_node("ReduceSum", ["crop_colors"], ["total_count"], axes=[2, 3], keepdims=0))
    nodes.append(helper.make_node("Equal", ["corners_count", "one_f"], ["corners_eq1"]))
    nodes.append(helper.make_node("Equal", ["edge_count", "one_f"], ["edge_eq1"]))
    nodes.append(helper.make_node("Greater", ["inner_count", "zero_f"], ["inner_gt0"]))
    nodes.append(helper.make_node("Equal", ["total_count", "center_count"], ["center_only"]))
    nodes.append(helper.make_node("Where", ["corners_eq1", "ten_f", "zero_f"], ["score_corners"]))
    nodes.append(helper.make_node("Where", ["edge_eq1", "ten_f", "zero_f"], ["score_edge"]))
    nodes.append(helper.make_node("Where", ["inner_gt0", "five_f", "zero_f"], ["score_inner"]))
    nodes.append(helper.make_node("Add", ["score_corners", "score_edge"], ["score_a"]))
    nodes.append(helper.make_node("Add", ["score_a", "score_inner"], ["score_b"]))
    nodes.append(helper.make_node("Sub", ["score_b", "total_count"], ["score_raw"]))
    nodes.append(helper.make_node("Where", ["center_only", "neg100_f", "score_raw"], ["score"]))
    nodes.append(helper.make_node("ArgMax", ["score"], ["color0_2d"], axis=1, keepdims=1))
    nodes.append(helper.make_node("Add", ["color0_2d", "one_i"], ["color_2d"]))
    nodes.append(helper.make_node("Squeeze", ["color_2d"], ["color"], axes=[0, 1]))
    nodes.append(helper.make_node("Gather", ["corners_count", "color0_2d"], ["sel_corners_3d"], axis=1))
    nodes.append(helper.make_node("Gather", ["edge_count", "color0_2d"], ["sel_edge_3d"], axis=1))
    nodes.append(helper.make_node("Gather", ["inner_count", "color0_2d"], ["sel_inner_3d"], axis=1))
    nodes.append(helper.make_node("Squeeze", ["sel_corners_3d"], ["sel_corners"], axes=[0, 1, 2]))
    nodes.append(helper.make_node("Squeeze", ["sel_edge_3d"], ["sel_edge"], axes=[0, 1, 2]))
    nodes.append(helper.make_node("Squeeze", ["sel_inner_3d"], ["sel_inner"], axes=[0, 1, 2]))
    nodes.append(helper.make_node("Equal", ["sel_corners", "one_f"], ["use_corners"]))
    nodes.append(helper.make_node("Equal", ["sel_edge", "one_f"], ["use_edge_raw"]))
    nodes.append(helper.make_node("Greater", ["sel_inner", "zero_f"], ["use_inner_raw"]))
    nodes.append(helper.make_node("Not", ["use_corners"], ["not_corners"]))
    nodes.append(helper.make_node("And", ["not_corners", "use_edge_raw"], ["use_edge"]))
    nodes.append(helper.make_node("Not", ["use_edge"], ["not_edge"]))
    nodes.append(helper.make_node("And", ["not_corners", "not_edge"], ["no_corner_edge"]))
    nodes.append(helper.make_node("And", ["no_corner_edge", "use_inner_raw"], ["use_inner"]))

    # Select the four spatial points of the chosen orbit group.
    nodes.append(helper.make_node("Where", ["use_edge", "edge_r", "corners_r"], ["row_base_a"]))
    nodes.append(helper.make_node("Where", ["use_edge", "edge_c", "corners_c"], ["col_base_a"]))
    nodes.append(helper.make_node("Where", ["use_inner", "inner_r", "row_base_a"], ["row_base"]))
    nodes.append(helper.make_node("Where", ["use_inner", "inner_c", "col_base_a"], ["col_base"]))
    nodes.append(helper.make_node("Add", ["row_base", "r0"], ["fill_rows_i"]))
    nodes.append(helper.make_node("Add", ["col_base", "c0"], ["fill_cols_i"]))
    nodes.append(helper.make_node("Cast", ["fill_rows_i"], ["fill_rows_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Cast", ["fill_cols_i"], ["fill_cols_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Unsqueeze", ["fill_rows_f"], ["fill_rows_4d"], axes=[0, 1, 2]))
    nodes.append(helper.make_node("Unsqueeze", ["fill_cols_f"], ["fill_cols_4d"], axes=[0, 1, 2]))
    nodes.append(helper.make_node("Unsqueeze", ["range30_f"], ["range30_r"], axes=[2, 3]))
    nodes.append(helper.make_node("Unsqueeze", ["range30_f"], ["range30_c"], axes=[1, 3]))
    nodes.append(helper.make_node("Equal", ["range30_r", "fill_rows_4d"], ["row_match"]))
    nodes.append(helper.make_node("Equal", ["range30_c", "fill_cols_4d"], ["col_match"]))
    nodes.append(helper.make_node("And", ["row_match", "col_match"], ["point_match"]))
    nodes.append(helper.make_node("Cast", ["point_match"], ["point_match_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("ReduceSum", ["point_match_f"], ["spatial_count"], axes=[3], keepdims=0))
    nodes.append(helper.make_node("Greater", ["spatial_count", "zero_f"], ["spatial_mask_2d"]))
    nodes.append(helper.make_node("Unsqueeze", ["spatial_mask_2d"], ["spatial_mask"], axes=[1]))

    nodes.append(helper.make_node("Equal", ["range10", "color"], ["channel_mask"]))
    nodes.append(helper.make_node("Cast", ["channel_mask"], ["channel_mask_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Where", ["spatial_mask", "channel_mask_f", "input"], ["output"]))

    raw = make_model(nodes, inits, EXP_ID, opset_version=11)
    return Candidate(
        TARGET_TASK,
        "task020_bbox5_group_fill_rule_v1",
        base.route,
        raw,
        "generated",
        "task020 explicit bbox5 group-fill rule from exp065",
    )


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_exp_tasks(BASE_EXP)
    base = base_tasks[TARGET_TASK]
    candidate = build_task020_rule_candidate(base)
    (EXP_DIR / "candidate.onnx").write_bytes(candidate.raw)
    eval_row, raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
    row = EvalRow(
        eval_row.task_id,
        eval_row.template_name,
        eval_row.baseline_cost,
        eval_row.candidate_cost,
        eval_row.baseline_points,
        eval_row.candidate_points,
        eval_row.file_bytes,
        eval_row.validation_status,
        eval_row.status,
        eval_row.reason,
        eval_row.sha256,
    )
    improved = raw is not None and row.status == "improved"
    final_raw = {task_id: task.raw for task_id, task in base_tasks.items()}
    if improved and raw is not None:
        final_raw[TARGET_TASK] = raw
    write_zip(OUTPUT_ZIP, final_raw)

    selected_rows: list[dict[str, Any]] = []
    for task_id in sorted(base_tasks):
        task = base_tasks[task_id]
        if task_id == TARGET_TASK and improved and raw is not None:
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": f"{EXP_ID}_{row.variant}",
                    "template_name": row.variant,
                    "route": task.route,
                    "cost": row.candidate_cost,
                    "local_points": row.candidate_points,
                    "file_bytes": len(raw),
                    "status": "improved",
                    "reason": row.reason,
                    "sha256": sha256(raw),
                }
            )
        else:
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": task.source,
                    "template_name": task.template_name,
                    "route": task.route,
                    "cost": task.cost,
                    "local_points": task.points,
                    "file_bytes": len(task.raw),
                    "status": "baseline",
                    "reason": "no accepted task020 ONNX gain",
                    "sha256": sha256(task.raw),
                }
            )

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(EvalRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerow(asdict(row))
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    accepted_delta = float(row.candidate_points) - row.baseline_points if improved else 0.0
    new_score = STRICT_SEED_SCORE + accepted_delta
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_submit_candidate" if improved else "no_gain_or_validation_failed",
        "target_task": TARGET_TASK,
        "baseline_local_estimate": STRICT_SEED_SCORE,
        "new_local_estimate": new_score,
        "delta": accepted_delta,
        "candidate": asdict(row),
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "leakage_risk": "low: explicit nonlookup rule derived from input geometry/color roles.",
        "overfitting_risk": "medium: task-specific; submit single-task delta if improved to calibrate LB.",
        "submission_decision": "submit_next_if_improved_and_zip_sane" if improved else "no_submit",
        "decision": "If validation passes but cost is high, optimize bbox/count/writeback. If improved, submit strict-seed single-task delta for LB calibration.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`exp065` のtask020入力only referenceをONNXへloweringし、strict seed task020 cost `90133` 未満のsingle-task deltaを作る。

## 結果

- status: {row.status}
- validation: {row.validation_status}
- baseline cost: {row.baseline_cost}
- candidate cost: {row.candidate_cost}
- delta: {accepted_delta:.6f}
- reason: {row.reason}

## 解釈

正しさ優先で、5x5 bbox crop、色1〜9のgroup count、center-only singleton除外、corners/edge/inner group fillをONNX化した。

## Decision

改善した場合はKaggle single-task delta提出へ進む。改善しない場合はbbox cropとwritebackのcostをprofileして削る。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
