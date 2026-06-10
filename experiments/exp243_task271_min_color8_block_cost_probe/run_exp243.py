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


EXP_ID = "exp243_task271_min_color8_block_cost_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 271


def build_candidate() -> bytes:
    initializers = [
        numpy_helper.from_array(np.asarray([0, 1, 0, 0], dtype=np.int64), "nz_starts"),
        numpy_helper.from_array(np.asarray([1, 10, 30, 30], dtype=np.int64), "nz_ends"),
        numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "axes4"),
        numpy_helper.from_array(np.asarray([0, 8, 0, 0], dtype=np.int64), "c8_starts"),
        numpy_helper.from_array(np.asarray([1, 9, 30, 30], dtype=np.int64), "c8_ends"),
        numpy_helper.from_array(np.ones((1, 9, 3, 3), dtype=np.float32), "nz_kernel"),
        numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), "c8_kernel"),
        numpy_helper.from_array(np.asarray(9.0, dtype=np.float32), "nine_f"),
        numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), "one_f"),
        numpy_helper.from_array(np.asarray(100.0, dtype=np.float32), "penalty_f"),
        numpy_helper.from_array(np.asarray([1, 784], dtype=np.int64), "shape_flat"),
        numpy_helper.from_array(np.asarray(28, dtype=np.int64), "twenty8"),
        numpy_helper.from_array(np.asarray([1, 1, 1, 1], dtype=np.int64), "shape1111"),
        numpy_helper.from_array(np.asarray([0, 1, 2], dtype=np.int64).reshape(1, 1, 3, 1), "row_offsets"),
        numpy_helper.from_array(np.asarray([0, 1, 2], dtype=np.int64).reshape(1, 1, 1, 3), "col_offsets"),
        numpy_helper.from_array(np.asarray([1, 10, 1, 30], dtype=np.int64), "row_tile_repeats"),
        numpy_helper.from_array(np.asarray([1, 10, 3, 1], dtype=np.int64), "col_tile_repeats"),
        numpy_helper.from_array(np.asarray([0, 0, 0, 0, 0, 0, 27, 27], dtype=np.int64), "pad_pads"),
        numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "pad_value"),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "nz_starts", "nz_ends", "axes4"], ["nz_channels"]),
        helper.make_node("Conv", ["nz_channels", "nz_kernel"], ["nz_count"]),
        helper.make_node("Equal", ["nz_count", "nine_f"], ["full_bool"]),
        helper.make_node("Cast", ["full_bool"], ["full_f"], to=1),
        helper.make_node("Sub", ["one_f", "full_f"], ["not_full"]),
        helper.make_node("Mul", ["not_full", "penalty_f"], ["penalty"]),
        helper.make_node("Slice", ["input", "c8_starts", "c8_ends", "axes4"], ["c8_channel"]),
        helper.make_node("Conv", ["c8_channel", "c8_kernel"], ["c8_count"]),
        helper.make_node("Add", ["c8_count", "penalty"], ["score"]),
        helper.make_node("Reshape", ["score", "shape_flat"], ["score_flat"]),
        helper.make_node("ArgMin", ["score_flat"], ["flat_idx"], axis=1, keepdims=1),
        helper.make_node("Div", ["flat_idx", "twenty8"], ["r0"]),
        helper.make_node("Mod", ["flat_idx", "twenty8"], ["c0"], fmod=0),
        helper.make_node("Reshape", ["r0", "shape1111"], ["r0_4d"]),
        helper.make_node("Reshape", ["c0", "shape1111"], ["c0_4d"]),
        helper.make_node("Add", ["r0_4d", "row_offsets"], ["row_idx_base"]),
        helper.make_node("Tile", ["row_idx_base", "row_tile_repeats"], ["row_idx"]),
        helper.make_node("GatherElements", ["input", "row_idx"], ["rows"], axis=2),
        helper.make_node("Add", ["c0_4d", "col_offsets"], ["col_idx_base"]),
        helper.make_node("Tile", ["col_idx_base", "col_tile_repeats"], ["col_idx"]),
        helper.make_node("GatherElements", ["rows", "col_idx"], ["crop"], axis=3),
        helper.make_node("Pad", ["crop", "pad_pads", "pad_value"], ["output"], mode="constant"),
    ]
    return make_model(nodes, initializers, f"{EXP_ID}_candidate", opset_version=11)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()[TASK_ID]
    candidate = Candidate(
        TASK_ID,
        "min_color8_full_block_crop3x3",
        base.route,
        build_candidate(),
        "generated",
        "select full nonzero 3x3 block with minimum color8 count, then GatherElements crop",
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
        "decision": "improvedならsingle-task delta提出候補。no_cost_gainならtask271 ruleはsolved assetとして保持し、別候補へpivot。",
        "submission_decision": "no_submit: cost probe only",
        "leakage_risk": "low: input-only structural rule.",
        "overfitting_risk": "medium-low: exp242で267/267だがhiddenのtie-break edgeは提出較正が必要。",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp242で見つけたtask271 rule「3x3 full-nonzero block 4候補からcolor8_count最小を選ぶ」をONNX化し、baseline cost `{base.cost}` より安くなるか確認する。

## 結果

- validation: `{eval_row.validation_status}`
- status: `{eval_row.status}`
- candidate_cost: `{eval_row.candidate_cost}`
- baseline_cost: `{eval_row.baseline_cost}`
- reason: `{eval_row.reason}`

## 判断

improvedならsingle-task delta提出候補。no_cost_gainならsolved assetとして保持し、別候補へpivotする。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
