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

from experiments.phase1_rewrite_utils import (
    BaseTask,
    Candidate,
    evaluate_candidate,
    examples_for,
    load_base_tasks,
    load_neurogolf_utils,
    load_task,
    make_model,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp046_task031_small_bbox_compiler"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
OVERLAY_EXPS = [
    ROOT / "experiments" / "exp040_deeper_fullarc_safe_bypass",
    ROOT / "experiments" / "exp041_task145_deeper_mul_chain",
]
AUTHORITATIVE_BASE_SCORE = 6480.302477938763
TARGET_TASK = 31
OUTPUT_ZIP = EXP_DIR / "submission.zip"


@dataclass
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


def load_current_best_tasks() -> dict[int, BaseTask]:
    base = load_base_tasks(BASE_EXP)
    for overlay_exp in OVERLAY_EXPS:
        if not (overlay_exp / "selected_manifest.csv").exists():
            continue
        with zipfile.ZipFile(overlay_exp / "submission.zip") as zf:
            overlay_raws = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}
        with (overlay_exp / "selected_manifest.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "improved":
                    continue
                task_id = int(row["task_id"])
                cost = int(float(row["candidate_cost"])) if "candidate_cost" in row else int(float(row["cost"]))
                points = float(row["candidate_points"]) if "candidate_points" in row else float(row["local_points"])
                base[task_id] = replace(
                    base[task_id],
                    cost=cost,
                    points=points,
                    source=f"{EXP_ID}_base_overlay_{overlay_exp.name}",
                    template_name=row["template_name"],
                    raw=overlay_raws[task_id],
                )
    return base


def max_output_shape(task_id: int) -> tuple[int, int]:
    examples = examples_for(load_task(task_id), -1)
    return max(len(ex["output"]) for ex in examples), max(len(ex["output"][0]) for ex in examples)


def init_i(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.int64), name)


def build_small_bbox_candidate(task_id: int, route: str) -> Candidate:
    max_h, max_w = max_output_shape(task_id)
    nodes: list[Any] = []
    inits: list[Any] = [
        init_i("ch_starts", np.asarray([0, 1, 0, 0])),
        init_i("ch_ends", np.asarray([1, 10, 30, 30])),
        init_i("axes4", np.asarray([0, 1, 2, 3])),
        numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero_f"),
        numpy_helper.from_array(np.arange(30, dtype=np.float32).reshape(1, 30), "range30_f"),
        init_i("one_i", np.asarray([1])),
        init_i("range_h", np.arange(max_h)),
        init_i("range_w", np.arange(max_w)),
        numpy_helper.from_array(np.zeros((1, 10, max_h, max_w), dtype=np.float32), "zero_out"),
        init_i("pad_pads", np.asarray([0, 0, 0, 0, 0, 0, 30 - max_h, 30 - max_w])),
    ]

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
    nodes.append(helper.make_node("Where", ["row_has", "range30_f", "zero_f"], ["row_weighted"]))
    nodes.append(helper.make_node("Where", ["col_has", "range30_f", "zero_f"], ["col_weighted"]))
    nodes.append(helper.make_node("ArgMax", ["row_weighted"], ["r_last_2d"], axis=1, keepdims=1))
    nodes.append(helper.make_node("ArgMax", ["col_weighted"], ["c_last_2d"], axis=1, keepdims=1))
    nodes.append(helper.make_node("Sub", ["r_last_2d", "r0_2d"], ["height_m1"]))
    nodes.append(helper.make_node("Sub", ["c_last_2d", "c0_2d"], ["width_m1"]))
    nodes.append(helper.make_node("Add", ["height_m1", "one_i"], ["height"]))
    nodes.append(helper.make_node("Add", ["width_m1", "one_i"], ["width"]))
    nodes.append(helper.make_node("Squeeze", ["r0_2d"], ["r0"], axes=[0, 1]))
    nodes.append(helper.make_node("Squeeze", ["c0_2d"], ["c0"], axes=[0, 1]))
    nodes.append(helper.make_node("Squeeze", ["height"], ["height_s"], axes=[0, 1]))
    nodes.append(helper.make_node("Squeeze", ["width"], ["width_s"], axes=[0, 1]))
    nodes.append(helper.make_node("Add", ["range_h", "r0"], ["row_idx"]))
    nodes.append(helper.make_node("Add", ["range_w", "c0"], ["col_idx"]))
    nodes.append(helper.make_node("Gather", ["input", "row_idx"], ["row_gather"], axis=2))
    nodes.append(helper.make_node("Gather", ["row_gather", "col_idx"], ["crop"], axis=3))
    nodes.append(helper.make_node("Less", ["range_h", "height_s"], ["valid_h"]))
    nodes.append(helper.make_node("Less", ["range_w", "width_s"], ["valid_w"]))
    nodes.append(helper.make_node("Unsqueeze", ["valid_h"], ["valid_h4"], axes=[0, 1, 3]))
    nodes.append(helper.make_node("Unsqueeze", ["valid_w"], ["valid_w4"], axes=[0, 1, 2]))
    nodes.append(helper.make_node("And", ["valid_h4", "valid_w4"], ["valid"]))
    nodes.append(helper.make_node("Where", ["valid", "crop", "zero_out"], ["masked"]))
    nodes.append(helper.make_node("Pad", ["masked", "pad_pads", "zero_f"], ["output"], mode="constant"))

    raw = make_model(nodes, inits, EXP_ID, opset_version=11)
    return Candidate(task_id, f"small_bbox_nonzero_{max_h}x{max_w}", route, raw, "generated", "small dynamic Gather bbox crop")


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_current_best_tasks()
    base = base_tasks[TARGET_TASK]
    candidate = build_small_bbox_candidate(TARGET_TASK, base.route)
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
                    "reason": "no small bbox compiler gain",
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
    new_score = AUTHORITATIVE_BASE_SCORE + accepted_delta
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_below_target" if improved else "no_gain",
        "target_task": TARGET_TASK,
        "baseline_local_estimate": AUTHORITATIVE_BASE_SCORE,
        "new_local_estimate": new_score,
        "delta": accepted_delta,
        "gap_to_6500": 6500.0 - new_score,
        "gap_to_7000": 7000.0 - new_score,
        "gap_to_7400": 7400.0 - new_score,
        "gap_to_7600": 7600.0 - new_score,
        "gap_to_7700": 7700.0 - new_score,
        "candidate": asdict(row),
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "leakage_risk": "low: explicit bbox rule, no signature lookup.",
        "overfitting_risk": "low-to-medium: task-specific but all arc-gen validated if accepted.",
        "decision": "Adopt if improved; otherwise existing artifact already implements this compiler pattern efficiently.",
    }
    with (EXP_DIR / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    notes = f"""# {EXP_ID}

## 仮説

exp045で見えた既存artifact型のsmall dynamic gather bboxを自前compilerとして再合成すれば、task031の既存artifactに近い、またはそれ以下のcostを達成できる。

## 結果

- status: {row.status}
- baseline cost: {row.baseline_cost}
- candidate cost: {row.candidate_cost}
- validation: {row.validation_status}
- score delta: {accepted_delta:.6f}
- reason: {row.reason}

## 解釈

full-grid `GatherND` ではなく最大bboxサイズだけを `Gather` する形にした。これで既存artifactの構造をcompilerとして再現できるかを確認する。

## 次

改善しない場合でも、このtemplateを他のbbox系taskへ適用し、既存artifactより高いtaskだけを狙う。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
