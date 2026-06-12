from __future__ import annotations

import json
import pathlib
import sys
import time
import zipfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    evaluate_candidate,
    load_neurogolf_utils,
    point,
    score_model,
    sha256,
)

EXP_ID = "exp332_task037_cropped_diag_shift_lowering"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 37
BASE_EXP = ROOT / "experiments" / "exp331_task185_gather_axis_cost_shave"
FALLBACK_BASE_EXP = ROOT / "experiments" / "exp297_exp262_skip_task048_336_fresh_candidates"
BASE_PUBLIC_LB = 6009.15
MAX_DISTANCE = 5
SIZE = 10


def init_i(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value.astype(np.int64), name)


def init_f(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value.astype(np.float32), name)


def make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto], producer: str) -> bytes:
    graph = helper.make_graph(
        nodes,
        f"{producer}_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(graph, producer_name=producer, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    return model.SerializeToString()


def slice_named(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    input_name: str,
    output_name: str,
    starts: list[int],
    ends: list[int],
) -> str:
    initializers.extend(
        [
            init_i(f"{output_name}_starts", np.asarray(starts)),
            init_i(f"{output_name}_ends", np.asarray(ends)),
            init_i(f"{output_name}_axes", np.asarray([0, 1, 2, 3])),
        ]
    )
    nodes.append(helper.make_node("Slice", [input_name, f"{output_name}_starts", f"{output_name}_ends", f"{output_name}_axes"], [output_name]))
    return output_name


def shift_tensor(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    name: str,
    input_name: str,
    dr: int,
    dc: int,
    distance: int,
) -> str:
    sr0 = max(0, dr * distance)
    sr1 = SIZE + min(0, dr * distance)
    sc0 = max(0, dc * distance)
    sc1 = SIZE + min(0, dc * distance)
    pad_top = max(0, -dr * distance)
    pad_bottom = max(0, dr * distance)
    pad_left = max(0, -dc * distance)
    pad_right = max(0, dc * distance)
    sliced = slice_named(nodes, initializers, input_name, f"{name}_slice", [0, 0, sr0, sc0], [1, 10, sr1, sc1])
    shifted = f"{name}_shift"
    nodes.append(
        helper.make_node(
            "Pad",
            [sliced],
            [shifted],
            mode="constant",
            pads=[0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right],
            value=0.0,
        )
    )
    return shifted


def build_candidate(base: BaseTask) -> Candidate:
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = [init_f("one", np.asarray([1.0]))]
    crop = slice_named(nodes, initializers, "input", "crop10", [0, 0, 0, 0], [1, 10, SIZE, SIZE])
    center_zero = slice_named(nodes, initializers, crop, "center_zero", [0, 0, 0, 0], [1, 1, SIZE, SIZE])
    shifted_input: dict[tuple[int, int, int], str] = {}
    shifted_zero: dict[tuple[int, int, int], str] = {}
    for dr, dc in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        for distance in range(1, MAX_DISTANCE + 1):
            full = shift_tensor(nodes, initializers, f"s_{dr}_{dc}_{distance}", crop, dr, dc, distance)
            shifted_input[(dr, dc, distance)] = full
            shifted_zero[(dr, dc, distance)] = slice_named(
                nodes,
                initializers,
                full,
                f"z_{dr}_{dc}_{distance}",
                [0, 0, 0, 0],
                [1, 1, SIZE, SIZE],
            )

    fill_terms: list[str] = []
    for plus, minus in (((1, 1), (-1, -1)), ((1, -1), (-1, 1))):
        pdr, pdc = plus
        mdr, mdc = minus
        for dp in range(1, MAX_DISTANCE + 1):
            for dm in range(1, MAX_DISTANCE + 1):
                prefix = f"ray_{pdr}_{pdc}_{dp}_{dm}"
                nodes.append(helper.make_node("Mul", [shifted_input[(pdr, pdc, dp)], shifted_input[(mdr, mdc, dm)]], [f"{prefix}_same_color"]))
                nodes.append(helper.make_node("Sub", ["one", shifted_zero[(pdr, pdc, dp)]], [f"{prefix}_p_nonzero"]))
                nodes.append(helper.make_node("Sub", ["one", shifted_zero[(mdr, mdc, dm)]], [f"{prefix}_m_nonzero"]))
                condition = center_zero
                for name in (f"{prefix}_p_nonzero", f"{prefix}_m_nonzero"):
                    out = f"{prefix}_cond_{len(nodes)}"
                    nodes.append(helper.make_node("Mul", [condition, name], [out]))
                    condition = out
                for step in range(1, dp):
                    out = f"{prefix}_p_gap_{step}"
                    nodes.append(helper.make_node("Mul", [condition, shifted_zero[(pdr, pdc, step)]], [out]))
                    condition = out
                for step in range(1, dm):
                    out = f"{prefix}_m_gap_{step}"
                    nodes.append(helper.make_node("Mul", [condition, shifted_zero[(mdr, mdc, step)]], [out]))
                    condition = out
                term = f"{prefix}_term"
                nodes.append(helper.make_node("Mul", [f"{prefix}_same_color", condition], [term]))
                fill_terms.append(term)

    nodes.append(helper.make_node("Sum", fill_terms, ["fill_all_sum"]))
    nodes.append(helper.make_node("Clip", ["fill_all_sum"], ["fill_all"], min=0.0, max=1.0))
    fill_colors = slice_named(nodes, initializers, "fill_all", "fill_colors", [0, 1, 0, 0], [1, 10, SIZE, SIZE])
    nodes.append(helper.make_node("ReduceMax", [fill_colors], ["any_fill"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Sub", ["one", "any_fill"], ["keep_zero"]))
    nodes.append(helper.make_node("Mul", [center_zero, "keep_zero"], ["out_zero"]))
    input_colors = slice_named(nodes, initializers, crop, "input_colors", [0, 1, 0, 0], [1, 10, SIZE, SIZE])
    nodes.append(helper.make_node("Add", [input_colors, fill_colors], ["out_colors_pre"]))
    nodes.append(helper.make_node("Clip", ["out_colors_pre"], ["out_colors"], min=0.0, max=1.0))
    nodes.append(helper.make_node("Concat", ["out_zero", "out_colors"], ["small_output"], axis=1))
    nodes.append(helper.make_node("Pad", ["small_output"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 20, 20], value=0.0))
    raw = make_model(nodes, initializers, EXP_ID)
    return Candidate(TASK_ID, f"cropped_diag_shift_d{MAX_DISTANCE}", base.route, raw, "generated", "crop task037 to 10x10 before bounded diagonal shifts")


def load_current_base_task(utils) -> BaseTask:
    base_zip = BASE_EXP / "submission.zip"
    base_label = "exp331_current_best"
    if not base_zip.exists():
        base_zip = FALLBACK_BASE_EXP / "submission.zip"
        base_label = "exp297_current_best"
    with zipfile.ZipFile(base_zip) as zf:
        raw = zf.read(f"task{TASK_ID:03d}.onnx")
    memory, params, reason = score_model(utils, raw, TASK_ID, f"exp332_{base_label}", EXP_DIR)
    if memory is None or params is None:
        raise RuntimeError(f"base score failed: {reason}")
    cost = int(memory) + int(params)
    return BaseTask(TASK_ID, cost, point(cost), base_label, "current_best_task037", "sparse_edit_or_object_completion", raw)


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_current_base_task(utils)
    candidate = build_candidate(base)
    (EXP_DIR / "cropped_diag_shift_d5.onnx").write_bytes(candidate.raw)
    eval_row, improved_raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
    accepted = improved_raw is not None and eval_row.status == "improved"
    delta = (float(eval_row.candidate_points) - base.points) if accepted else 0.0
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-12",
        "status": "improved_candidate" if accepted else "no_gain",
        "task_id": TASK_ID,
        "hypothesis": "Cropping task037 to its true 10x10 working area before the bounded diagonal shift rule will reduce full-grid intermediate memory enough to approach cost competitiveness.",
        "base": {"cost": base.cost, "points": base.points, "sha256": sha256(base.raw), "source": base.source},
        "candidate": {
            "template_name": eval_row.template_name,
            "validation_status": eval_row.validation_status,
            "status": eval_row.status,
            "reason": eval_row.reason,
            "cost": eval_row.candidate_cost,
            "points": eval_row.candidate_points,
            "file_bytes": eval_row.file_bytes,
            "sha256": eval_row.sha256,
        },
        "local_estimate_delta": delta,
        "expected_public_lb_if_calibrated": BASE_PUBLIC_LB + delta,
        "submission_decision": "submit_if_bundled_or_single_delta" if accepted else "no_submit: validation failed or no cost gain",
        "decision": "Adopt if improved; otherwise treat cropped dense shift as still too expensive and move to true per-diagonal sparse representation or task251.",
        "leakage_risk": "low: explicit input-only diagonal ray rule; no output lookup or public feedback.",
        "overfitting_risk": "medium: task-specific geometry is full-arc validated locally, but cost expression may not generalize beyond this task family.",
        "runtime_seconds": time.time() - started,
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp327 の full-grid 30x30 bounded shift stack は正しいが cost `6633317` で不採用だった。task037 の実 working area は 10x10 なので、最初に crop してから同じ rule を実行し、最後に 30x30 へ Pad すれば memory cost が下がるか確認する。

## 結果

- base cost: `{base.cost}`
- candidate validation: `{eval_row.validation_status}`
- candidate status: `{eval_row.status}`
- candidate cost: `{eval_row.candidate_cost}`
- local delta: `{delta}`

## 判断

{result['submission_decision']}

## 解釈

{result['decision']}

## Risk

- leakage risk: {result['leakage_risk']}
- overfitting risk: {result['overfitting_risk']}
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
