from __future__ import annotations

import csv
import json
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    evaluate_candidate,
    load_neurogolf_utils,
    make_model,
    point,
)


EXP_ID = "exp165_task025_line_projection_onnx_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp152_task023_b035_repair_probe"
TASK_ID = 25
BASE_COST = 89286


def init_i(name: str, arr: np.ndarray) -> Any:
    return numpy_helper.from_array(arr.astype(np.int64), name)


def init_f(name: str, arr: np.ndarray) -> Any:
    return numpy_helper.from_array(arr.astype(np.float32), name)


def init_b(name: str, arr: np.ndarray) -> Any:
    return numpy_helper.from_array(arr.astype(bool), name)


def build_candidate() -> Candidate:
    nodes: list[Any] = []
    inits: list[Any] = [
        init_i("axes_ch", np.asarray([1])),
        init_i("axes_hw", np.asarray([2, 3])),
        init_i("axes_w", np.asarray([3])),
        init_i("axes_h", np.asarray([2])),
        init_i("axes_squeeze", np.asarray([0, 1])),
        init_i("idx_bg", np.asarray([0])),
        init_f("half", np.asarray([0.5])),
        init_f("zero", np.asarray([0.0])),
        init_f("one", np.asarray([1.0])),
        init_f("before_row", before_matrix(axis="row")),
        init_f("after_row", after_matrix(axis="row")),
        init_f("before_col", before_matrix(axis="col")),
        init_f("after_col", after_matrix(axis="col")),
        init_i("gather_up", np.asarray(list(range(1, 30)) + [29])),
        init_i("gather_down", np.asarray([0] + list(range(0, 29)))),
        init_b("not_last_row", mask_not_last_row()),
        init_b("not_first_row", mask_not_first_row()),
        init_b("not_last_col", mask_not_last_col()),
        init_b("not_first_col", mask_not_first_col()),
    ]

    nodes.extend(
        [
            helper.make_node("ReduceSum", ["input"], ["valid_sum"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["valid_sum", "half"], ["valid"]),
            helper.make_node("Cast", ["valid"], ["valid_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["valid_f"], ["valid_row_count"], axes=[3], keepdims=1),
            helper.make_node("ReduceSum", ["valid_f"], ["valid_col_count"], axes=[2], keepdims=1),
            helper.make_node("Greater", ["valid_row_count", "zero"], ["row_has_valid"]),
            helper.make_node("Greater", ["valid_col_count", "zero"], ["col_has_valid"]),
        ]
    )

    nonzero_channels: list[str] = []
    for color in range(1, 10):
        ch = f"c{color}"
        inits.append(init_i(f"{ch}_idx", np.asarray([color])))
        nodes.extend(
            [
                helper.make_node("Gather", ["input", f"{ch}_idx"], [f"{ch}_x"], axis=1),
                helper.make_node("Greater", [f"{ch}_x", "half"], [f"{ch}_is"]),
                helper.make_node("ReduceSum", [f"{ch}_x"], [f"{ch}_row_count"], axes=[3], keepdims=1),
                helper.make_node("ReduceSum", [f"{ch}_x"], [f"{ch}_col_count"], axes=[2], keepdims=1),
                helper.make_node("Equal", [f"{ch}_row_count", "valid_row_count"], [f"{ch}_row_full_raw"]),
                helper.make_node("Equal", [f"{ch}_col_count", "valid_col_count"], [f"{ch}_col_full_raw"]),
                helper.make_node("And", [f"{ch}_row_full_raw", "row_has_valid"], [f"{ch}_row_full"]),
                helper.make_node("And", [f"{ch}_col_full_raw", "col_has_valid"], [f"{ch}_col_full"]),
                helper.make_node("Or", [f"{ch}_row_full", f"{ch}_col_full"], [f"{ch}_guide"]),
                helper.make_node("And", [f"{ch}_guide", "valid"], [f"{ch}_line_keep"]),
                helper.make_node("Squeeze", [f"{ch}_x"], [f"{ch}_x2"], axes=[0, 1]),
                helper.make_node("Squeeze", [f"{ch}_row_full"], [f"{ch}_row_line2_col"], axes=[0, 1, 3]),
                helper.make_node("Unsqueeze", [f"{ch}_row_line2_col"], [f"{ch}_row_line2"], axes=[1]),
                helper.make_node("Squeeze", [f"{ch}_col_full"], [f"{ch}_col_line2_row"], axes=[0, 1, 2]),
                helper.make_node("Unsqueeze", [f"{ch}_col_line2_row"], [f"{ch}_col_line2"], axes=[0]),
                helper.make_node("MatMul", ["before_row", f"{ch}_x2"], [f"{ch}_above_count"]),
                helper.make_node("MatMul", ["after_row", f"{ch}_x2"], [f"{ch}_below_count"]),
                helper.make_node("Greater", [f"{ch}_above_count", "zero"], [f"{ch}_above_any"]),
                helper.make_node("Greater", [f"{ch}_below_count", "zero"], [f"{ch}_below_any"]),
                helper.make_node("And", [f"{ch}_above_any", f"{ch}_row_line2"], [f"{ch}_above_at_line"]),
                helper.make_node("And", [f"{ch}_below_any", f"{ch}_row_line2"], [f"{ch}_below_at_line"]),
                helper.make_node("Gather", [f"{ch}_above_at_line", "gather_up"], [f"{ch}_proj_up_raw"], axis=0),
                helper.make_node("Gather", [f"{ch}_below_at_line", "gather_down"], [f"{ch}_proj_down_raw"], axis=0),
                helper.make_node("And", [f"{ch}_proj_up_raw", "not_last_row"], [f"{ch}_proj_up"]),
                helper.make_node("And", [f"{ch}_proj_down_raw", "not_first_row"], [f"{ch}_proj_down"]),
                helper.make_node("MatMul", [f"{ch}_x2", "before_col"], [f"{ch}_left_count"]),
                helper.make_node("MatMul", [f"{ch}_x2", "after_col"], [f"{ch}_right_count"]),
                helper.make_node("Greater", [f"{ch}_left_count", "zero"], [f"{ch}_left_any"]),
                helper.make_node("Greater", [f"{ch}_right_count", "zero"], [f"{ch}_right_any"]),
                helper.make_node("And", [f"{ch}_left_any", f"{ch}_col_line2"], [f"{ch}_left_at_line"]),
                helper.make_node("And", [f"{ch}_right_any", f"{ch}_col_line2"], [f"{ch}_right_at_line"]),
                helper.make_node("Gather", [f"{ch}_left_at_line", "gather_up"], [f"{ch}_proj_left_raw"], axis=1),
                helper.make_node("Gather", [f"{ch}_right_at_line", "gather_down"], [f"{ch}_proj_right_raw"], axis=1),
                helper.make_node("And", [f"{ch}_proj_left_raw", "not_last_col"], [f"{ch}_proj_left"]),
                helper.make_node("And", [f"{ch}_proj_right_raw", "not_first_col"], [f"{ch}_proj_right"]),
                helper.make_node("Or", [f"{ch}_proj_up", f"{ch}_proj_down"], [f"{ch}_proj_h"]),
                helper.make_node("Or", [f"{ch}_proj_left", f"{ch}_proj_right"], [f"{ch}_proj_v"]),
                helper.make_node("Or", [f"{ch}_proj_h", f"{ch}_proj_v"], [f"{ch}_proj2"]),
                helper.make_node("Unsqueeze", [f"{ch}_proj2"], [f"{ch}_proj"], axes=[0, 1]),
                helper.make_node("And", [f"{ch}_proj", "valid"], [f"{ch}_proj_valid"]),
                helper.make_node("Or", [f"{ch}_line_keep", f"{ch}_proj_valid"], [f"{ch}_out_bool"]),
                helper.make_node("Cast", [f"{ch}_out_bool"], [f"{ch}_out"], to=TensorProto.FLOAT),
            ]
        )
        nonzero_channels.append(f"{ch}_out")

    nodes.append(helper.make_node("Concat", nonzero_channels, ["nonzero_out"], axis=1))
    nodes.extend(
        [
            helper.make_node("ReduceSum", ["nonzero_out"], ["nonzero_sum"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["nonzero_sum", "half"], ["has_nonzero"]),
            helper.make_node("Not", ["has_nonzero"], ["no_nonzero"]),
            helper.make_node("And", ["valid", "no_nonzero"], ["bg_bool"]),
            helper.make_node("Cast", ["bg_bool"], ["bg_out"], to=TensorProto.FLOAT),
            helper.make_node("Concat", ["bg_out", "nonzero_out"], ["output"], axis=1),
        ]
    )
    raw = make_model(nodes, inits, "exp165_task025_line_projection", opset_version=11)
    return Candidate(TASK_ID, "task025_line_projection_correctness_first", "same_shape", raw, "generated", "full-line guide projection")


def before_matrix(axis: str) -> np.ndarray:
    mat = np.zeros((30, 30), dtype=np.float32)
    if axis == "row":
        for target in range(30):
            mat[target, :target] = 1.0
    else:
        for src in range(30):
            mat[src, src + 1 :] = 1.0
    return mat


def after_matrix(axis: str) -> np.ndarray:
    mat = np.zeros((30, 30), dtype=np.float32)
    if axis == "row":
        for target in range(30):
            mat[target, target + 1 :] = 1.0
    else:
        for src in range(30):
            mat[src, :src] = 1.0
    return mat


def mask_not_last_row() -> np.ndarray:
    mask = np.ones((30, 30), dtype=bool)
    mask[-1, :] = False
    return mask


def mask_not_first_row() -> np.ndarray:
    mask = np.ones((30, 30), dtype=bool)
    mask[0, :] = False
    return mask


def mask_not_last_col() -> np.ndarray:
    mask = np.ones((30, 30), dtype=bool)
    mask[:, -1] = False
    return mask


def mask_not_first_col() -> np.ndarray:
    mask = np.ones((30, 30), dtype=bool)
    mask[:, 0] = False
    return mask


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp165_task025_line_projection_onnx_probe",
        "",
        "## 目的",
        "",
        "exp164で見つかったtask025 guide-line projection ruleをcorrectness-first ONNXに落とし、full validationとofficial costを測る。",
        "",
        "## 結果",
        "",
        f"- status: `{result['eval']['status']}`",
        f"- validation_status: `{result['eval']['validation_status']}`",
        f"- baseline_cost: `{result['eval']['baseline_cost']}`",
        f"- candidate_cost: `{result['eval']['candidate_cost']}`",
        f"- decision: {result['decision']}",
        "",
        "## リスク",
        "",
        result["leakage_risk"],
        result["overfitting_risk"],
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE_EXP / "submission.zip") as zf:
        base_raw = zf.read(f"task{TASK_ID:03d}.onnx")
    base = BaseTask(
        task_id=TASK_ID,
        cost=BASE_COST,
        points=point(BASE_COST),
        source=BASE_EXP.name,
        template_name="current_public_zero_task025",
        route="same_shape",
        raw=base_raw,
    )
    utils = load_neurogolf_utils()
    candidate = build_candidate()
    if candidate.raw is not None:
        (EXP_DIR / "candidate_raw.onnx").write_bytes(candidate.raw)
        sanitized = utils.sanitize_model(onnx.load_model_from_string(candidate.raw))
        if sanitized is not None:
            (EXP_DIR / "candidate_sanitized.onnx").write_bytes(sanitized.SerializeToString())
    eval_row, raw = evaluate_candidate(utils, candidate, base, arc_gen_sample=-1, exp_dir=EXP_DIR)
    if raw is not None:
        (EXP_DIR / "candidate.onnx").write_bytes(raw)
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(eval_row).keys()))
        writer.writeheader()
        writer.writerow(asdict(eval_row))
    improved = eval_row.status == "improved"
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-10",
        "status": "complete",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "task_id": TASK_ID,
        "eval": asdict(eval_row),
        "elapsed_s": round(time.time() - t0, 3),
        "decision": "adopt_and_build_repair_submission" if improved else "do_not_adopt_refine_lowering",
        "leakage_risk": "low: input-only rule lowered to ONNX; no labels or raw data beyond provided validation examples are embedded.",
        "overfitting_risk": "medium: rule assumes complete guide lines and may need hidden robustness review before final-safe classification.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
