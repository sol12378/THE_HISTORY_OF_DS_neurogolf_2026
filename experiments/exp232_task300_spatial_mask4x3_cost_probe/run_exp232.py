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


EXP_ID = "exp232_task300_spatial_mask4x3_cost_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 300


def build_candidate() -> bytes:
    initializers = [
        numpy_helper.from_array(np.asarray([0, 1], dtype=np.int64), "count_starts"),
        numpy_helper.from_array(np.asarray([1, 10], dtype=np.int64), "count_ends"),
        numpy_helper.from_array(np.asarray([0, 1], dtype=np.int64), "axes2"),
        numpy_helper.from_array(np.asarray(1, dtype=np.int64), "one_i64"),
        numpy_helper.from_array(np.arange(10, dtype=np.int64).reshape(1, 10), "color_range10"),
        numpy_helper.from_array(np.arange(1, 10, dtype=np.int64).reshape(1, 9), "color_range9"),
        numpy_helper.from_array(np.asarray([1, 10, 1, 1], dtype=np.int64), "color_shape10"),
        numpy_helper.from_array(np.asarray([1, 9, 1, 1], dtype=np.int64), "color_shape9"),
        numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64).reshape(1, 1, 4, 1), "row_offsets"),
        numpy_helper.from_array(np.asarray([0, 1, 2], dtype=np.int64).reshape(1, 1, 1, 3), "col_offsets"),
        numpy_helper.from_array(np.asarray([1, 1, 1, 30], dtype=np.int64), "row_tile_repeats"),
        numpy_helper.from_array(np.asarray([1, 1, 4, 1], dtype=np.int64), "col_tile_repeats"),
        numpy_helper.from_array(np.asarray([0, 0, 0, 0, 0, 0, 26, 27], dtype=np.int64), "pad_pads"),
        numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "pad_zero"),
    ]
    nodes = [
        helper.make_node("ReduceSum", ["input"], ["counts"], axes=[2, 3], keepdims=0),
        helper.make_node("Slice", ["counts", "count_starts", "count_ends", "axes2"], ["counts_nz"]),
        helper.make_node("ArgMax", ["counts_nz"], ["sel0"], axis=1, keepdims=1),
        helper.make_node("Add", ["sel0", "one_i64"], ["sel_color"]),
        helper.make_node("Equal", ["color_range10", "sel_color"], ["sel_color_bool10"]),
        helper.make_node("Cast", ["sel_color_bool10"], ["sel_color_float10"], to=1),
        helper.make_node("Reshape", ["sel_color_float10", "color_shape10"], ["sel_color_4d10"]),
        helper.make_node("Mul", ["input", "sel_color_4d10"], ["selected_allch"]),
        helper.make_node("ReduceSum", ["selected_allch"], ["selected_spatial"], axes=[1], keepdims=1),
        helper.make_node("ReduceMax", ["selected_spatial"], ["row_has"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["selected_spatial"], ["col_has"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_has"], ["r0"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["col_has"], ["c0"], axis=3, keepdims=1),
        helper.make_node("Add", ["r0", "row_offsets"], ["row_idx_base"]),
        helper.make_node("Tile", ["row_idx_base", "row_tile_repeats"], ["row_idx"]),
        helper.make_node("GatherElements", ["selected_spatial", "row_idx"], ["rows"], axis=2),
        helper.make_node("Add", ["c0", "col_offsets"], ["col_idx_base"]),
        helper.make_node("Tile", ["col_idx_base", "col_tile_repeats"], ["col_idx"]),
        helper.make_node("GatherElements", ["rows", "col_idx"], ["mask4x3"], axis=3),
        helper.make_node("ReduceMax", ["mask4x3"], ["row_any"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["mask4x3"], ["col_any"], axes=[2], keepdims=1),
        helper.make_node("Mul", ["row_any", "col_any"], ["shape_mask"]),
        helper.make_node("Sub", ["shape_mask", "mask4x3"], ["bg_crop"]),
        helper.make_node("Pad", ["bg_crop", "pad_pads", "pad_zero"], ["bg_full"], mode="constant"),
        helper.make_node("Pad", ["mask4x3", "pad_pads", "pad_zero"], ["mask_full"], mode="constant"),
        helper.make_node("Equal", ["color_range9", "sel_color"], ["sel_color_bool9"]),
        helper.make_node("Cast", ["sel_color_bool9"], ["sel_color_float9"], to=1),
        helper.make_node("Reshape", ["sel_color_float9", "color_shape9"], ["sel_color_4d9"]),
        helper.make_node("Mul", ["mask_full", "sel_color_4d9"], ["nz_full"]),
        helper.make_node("Concat", ["bg_full", "nz_full"], ["output"], axis=1),
    ]
    return make_model(nodes, initializers, f"{EXP_ID}_candidate", opset_version=11)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()[TASK_ID]
    candidate = Candidate(
        TASK_ID,
        "max_color_spatial_mask4x3",
        base.route,
        build_candidate(),
        "generated",
        "max-color selected spatial mask crop4x3 then channelize",
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
        "decision": "improvedならbundle/submission候補。no_cost_gainならtask300は更なるdtype/graph optimizationが必要。",
        "submission_decision": "no_submit: cost probe only",
        "leakage_risk": "low",
        "overfitting_risk": "medium-low",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task300 max-color proxyのGatherElements対象を10chから1ch spatial maskへ縮小し、exp231のcost `92595` をbaseline `77546` 未満に落とせるか確認する。

## 結果

- validation: `{eval_row.validation_status}`
- status: `{eval_row.status}`
- candidate_cost: `{eval_row.candidate_cost}`
- baseline_cost: `{eval_row.baseline_cost}`
- reason: `{eval_row.reason}`

## 判断

improvedならbundle/submission候補。no_cost_gainなら更なるdtype/graph optimizationが必要。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
