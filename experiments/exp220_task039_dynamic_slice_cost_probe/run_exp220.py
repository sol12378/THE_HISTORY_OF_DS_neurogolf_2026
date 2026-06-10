from __future__ import annotations

import csv
import json
import pathlib
import sys
from dataclasses import asdict
from datetime import date

import numpy as np
from onnx import helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import Candidate, evaluate_candidate, load_base_tasks, load_neurogolf_utils, make_model  # noqa: E402


EXP_ID = "exp220_task039_dynamic_slice_cost_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 39


def build_dynamic_slice() -> bytes:
    initializers = [
        numpy_helper.from_array(np.asarray([0, 1, 0, 0], dtype=np.int64), "color_starts"),
        numpy_helper.from_array(np.asarray([1, 10, 30, 30], dtype=np.int64), "color_ends"),
        numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "axes4"),
        numpy_helper.from_array(np.asarray([0, 0], dtype=np.int64), "batch_channel_start"),
        numpy_helper.from_array(np.asarray([1, 10], dtype=np.int64), "batch_channel_end"),
        numpy_helper.from_array(np.asarray([3], dtype=np.int64), "three"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "shape1"),
        numpy_helper.from_array(np.asarray([0, 0, 0, 0, 0, 0, 27, 27], dtype=np.int64), "pad_pads"),
        numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "pad_value"),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "color_starts", "color_ends", "axes4"], ["nz_channels"]),
        helper.make_node("ReduceSum", ["nz_channels"], ["nz_mask"], axes=[1], keepdims=1),
        helper.make_node("ReduceMax", ["nz_mask"], ["row_has"], axes=[3], keepdims=0),
        helper.make_node("ReduceMax", ["nz_mask"], ["col_has"], axes=[2], keepdims=0),
        helper.make_node("ArgMax", ["row_has"], ["r0_raw"], axis=1, keepdims=0),
        helper.make_node("ArgMax", ["col_has"], ["c0_raw"], axis=2, keepdims=0),
        helper.make_node("Reshape", ["r0_raw", "three"], ["bad_shape_probe"]),
    ]
    # Placeholder shape probe kept intentionally impossible? No, replace with dynamic slice below.
    nodes = [
        helper.make_node("Slice", ["input", "color_starts", "color_ends", "axes4"], ["nz_channels"]),
        helper.make_node("ReduceSum", ["nz_channels"], ["nz_mask"], axes=[1], keepdims=1),
        helper.make_node("ReduceMax", ["nz_mask"], ["row_has"], axes=[3], keepdims=0),
        helper.make_node("ReduceMax", ["nz_mask"], ["col_has"], axes=[2], keepdims=0),
        helper.make_node("ArgMax", ["row_has"], ["r0"], axis=1, keepdims=0),
        helper.make_node("ArgMax", ["col_has"], ["c0"], axis=2, keepdims=0),
        helper.make_node("Add", ["r0", "three"], ["r1_raw"]),
        helper.make_node("Add", ["c0", "three"], ["c1_raw"]),
        helper.make_node("Reshape", ["r0", "shape1"], ["r0_flat"]),
        helper.make_node("Reshape", ["c0", "shape1"], ["c0_flat"]),
        helper.make_node("Reshape", ["r1_raw", "shape1"], ["r1"]),
        helper.make_node("Reshape", ["c1_raw", "shape1"], ["c1"]),
        helper.make_node("Concat", ["batch_channel_start", "r0_flat", "c0_flat"], ["starts"], axis=0),
        helper.make_node("Concat", ["batch_channel_end", "r1", "c1"], ["ends"], axis=0),
        helper.make_node("Slice", ["input", "starts", "ends", "axes4"], ["crop"]),
        helper.make_node("Pad", ["crop", "pad_pads", "pad_value"], ["output"], mode="constant"),
    ]
    return make_model(nodes, initializers, f"{EXP_ID}_dynamic_slice", opset_version=11)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()[TASK_ID]
    candidate = Candidate(TASK_ID, "bbox_top_left_3x3_dynamic_slice", base.route, build_dynamic_slice(), "generated", "dynamic Slice bbox crop")
    eval_row, raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)

    with (EXP_DIR / "cost_probe.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(eval_row).keys()))
        writer.writeheader()
        writer.writerow(asdict(eval_row))

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "candidate": asdict(eval_row),
        "improved": raw is not None and eval_row.status == "improved",
        "decision": "static checkとcostが通れば提出候補。dynamic shape rejectならこのlaneは現状保留。",
        "submission_decision": "no_submit: cost probe only",
        "leakage_risk": "low",
        "overfitting_risk": "medium-low",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task039 bbox 3x3 cropを `GatherElements` ではなく dynamic `Slice` で軽量化できるか試す。

## 結果

- validation: `{eval_row.validation_status}`
- status: `{eval_row.status}`
- candidate_cost: `{eval_row.candidate_cost}`
- reason: `{eval_row.reason}`

## 判断

static shape rejectならdynamic Slice laneは現状保留。通って改善ならbundle化へ進む。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
