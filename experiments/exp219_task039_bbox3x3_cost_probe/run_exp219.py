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

from experiments.phase1_rewrite_utils import (  # noqa: E402
    Candidate,
    evaluate_candidate,
    load_base_tasks,
    load_neurogolf_utils,
    make_model,
)


EXP_ID = "exp219_task039_bbox3x3_cost_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 39


def build_bbox_top_left_3x3() -> bytes:
    initializers = [
        numpy_helper.from_array(np.asarray([0, 1, 0, 0], dtype=np.int64), "color_starts"),
        numpy_helper.from_array(np.asarray([1, 10, 30, 30], dtype=np.int64), "color_ends"),
        numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "axes4"),
        numpy_helper.from_array(np.asarray([0, 1, 2], dtype=np.int64).reshape(1, 1, 3, 1), "row_offsets"),
        numpy_helper.from_array(np.asarray([0, 1, 2], dtype=np.int64).reshape(1, 1, 1, 3), "col_offsets"),
        numpy_helper.from_array(np.asarray([1, 10, 1, 30], dtype=np.int64), "row_tile_repeats"),
        numpy_helper.from_array(np.asarray([1, 10, 3, 1], dtype=np.int64), "col_tile_repeats"),
        numpy_helper.from_array(np.asarray([0, 0, 0, 0, 0, 0, 27, 27], dtype=np.int64), "pad_pads"),
        numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "pad_value"),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "color_starts", "color_ends", "axes4"], ["nz_channels"]),
        helper.make_node("ReduceSum", ["nz_channels"], ["nz_mask"], axes=[1], keepdims=1),
        helper.make_node("ReduceMax", ["nz_mask"], ["row_has"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["nz_mask"], ["col_has"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_has"], ["r0"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["col_has"], ["c0"], axis=3, keepdims=1),
        helper.make_node("Add", ["r0", "row_offsets"], ["row_idx_base"]),
        helper.make_node("Tile", ["row_idx_base", "row_tile_repeats"], ["row_idx"]),
        helper.make_node("GatherElements", ["input", "row_idx"], ["rows"], axis=2),
        helper.make_node("Add", ["c0", "col_offsets"], ["col_idx_base"]),
        helper.make_node("Tile", ["col_idx_base", "col_tile_repeats"], ["col_idx"]),
        helper.make_node("GatherElements", ["rows", "col_idx"], ["crop"], axis=3),
        helper.make_node("Pad", ["crop", "pad_pads", "pad_value"], ["output"], mode="constant"),
    ]
    return make_model(nodes, initializers, f"{EXP_ID}_bbox_tl_3x3", opset_version=11)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()[TASK_ID]
    candidate = Candidate(
        TASK_ID,
        "bbox_top_left_3x3_gatherelements",
        base.route,
        build_bbox_top_left_3x3(),
        "generated",
        "dynamic nonzero bbox top-left 3x3 crop",
    )
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
        "decision": "improvedならbundle化して提出候補。no_cost_gainならdynamic bbox cropのGatherElements loweringは保留。",
        "submission_decision": "no_submit: cost probe only",
        "leakage_risk": "low: input-only bbox crop rule, exp218で全264例pass。",
        "overfitting_risk": "medium-low: bbox crop rule is simple, but hidden shape/color edgeはsingle-task submissionで較正する。",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp218でfull hitしたtask039 `bbox_top_left 3x3 crop` をONNX化し、baseline cost `7772` より安いか確認する。

## 結果

- validation: `{eval_row.validation_status}`
- status: `{eval_row.status}`
- candidate_cost: `{eval_row.candidate_cost}`
- baseline_cost: `{eval_row.baseline_cost}`

## 判断

improvedならbundle化して提出候補。no_cost_gainまたはvalidation failureなら、このdynamic bbox crop loweringは現状保留。

## リスク

- leakage risk: low。入力nonzero bboxだけを使う。
- overfitting risk: medium-low。arc-gen全例pass前提だが、hidden shape edgeは提出較正で確認する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
