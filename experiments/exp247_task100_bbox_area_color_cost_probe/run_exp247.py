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


EXP_ID = "exp247_task100_bbox_area_color_cost_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 100


def build_candidate() -> bytes:
    initializers = [
        numpy_helper.from_array(np.asarray([0, 1, 0, 0], dtype=np.int64), "nz_starts"),
        numpy_helper.from_array(np.asarray([1, 10, 30, 30], dtype=np.int64), "nz_ends"),
        numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "axes4"),
        numpy_helper.from_array(np.asarray(1, dtype=np.int64), "one_i64"),
        numpy_helper.from_array(np.arange(1, 10, dtype=np.int64).reshape(1, 9), "color_range9"),
        numpy_helper.from_array(np.asarray([1, 9, 1, 1], dtype=np.int64), "shape1911"),
        numpy_helper.from_array(np.asarray([1, 1, 2, 2], dtype=np.int64), "tile_2x2"),
        numpy_helper.from_array(np.asarray([0, 0, 0, 0, 0, 0, 28, 28], dtype=np.int64), "pad_pads"),
        numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "pad_value"),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "nz_starts", "nz_ends", "axes4"], ["nz_channels"]),
        helper.make_node("ReduceMax", ["nz_channels"], ["row_has"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["nz_channels"], ["col_has"], axes=[2], keepdims=1),
        helper.make_node("ReduceSum", ["row_has"], ["row_span"], axes=[2, 3], keepdims=0),
        helper.make_node("ReduceSum", ["col_has"], ["col_span"], axes=[2, 3], keepdims=0),
        helper.make_node("Mul", ["row_span", "col_span"], ["bbox_area_proxy"]),
        helper.make_node("ArgMax", ["bbox_area_proxy"], ["sel0"], axis=1, keepdims=1),
        helper.make_node("Add", ["sel0", "one_i64"], ["sel_color"]),
        helper.make_node("Equal", ["color_range9", "sel_color"], ["sel_bool"]),
        helper.make_node("Cast", ["sel_bool"], ["sel_float"], to=1),
        helper.make_node("Reshape", ["sel_float", "shape1911"], ["sel_4d"]),
        helper.make_node("Tile", ["sel_4d", "tile_2x2"], ["nz_2x2"]),
        helper.make_node("Pad", ["nz_2x2", "pad_pads", "pad_value"], ["nz_full"], mode="constant"),
        helper.make_node("ReduceSum", ["nz_full"], ["bg_sum"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["bg_sum", "bg_sum"], ["bg_full_zero"]),
        helper.make_node("Concat", ["bg_full_zero", "nz_full"], ["output"], axis=1),
    ]
    return make_model(nodes, initializers, f"{EXP_ID}_candidate", opset_version=11)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()[TASK_ID]
    candidate = Candidate(
        TASK_ID,
        "bbox_area_max_color_static_2x2",
        base.route,
        build_candidate(),
        "generated",
        "select nonzero color with max bbox area, output static 2x2 block",
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
        "decision": "improvedならsingle-task delta提出候補。no_cost_gain/validation failならtask100はsolved assetとして保持。",
        "submission_decision": "no_submit: cost probe only",
        "leakage_risk": "low: input-only bbox-area color rule.",
        "overfitting_risk": "medium-low: exp246で266/266だがhidden tie edgeは提出較正が必要。",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task100 rule「bbox_area最大の非zero色を選び、2x2全同色templateを出力」をONNX化し、baseline cost `{base.cost}` より安いか確認する。

## 結果

- validation: `{eval_row.validation_status}`
- status: `{eval_row.status}`
- candidate_cost: `{eval_row.candidate_cost}`
- baseline_cost: `{eval_row.baseline_cost}`
- reason: `{eval_row.reason}`

## 判断

improvedならsingle-task delta提出候補。no_cost_gain/validation failならtask100はsolved assetとして保持。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
