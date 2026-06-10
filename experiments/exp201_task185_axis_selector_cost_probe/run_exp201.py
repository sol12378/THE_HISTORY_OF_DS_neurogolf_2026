from __future__ import annotations

import csv
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import infer_static_ok, load_base_tasks, load_neurogolf_utils, score_model, sha256  # noqa: E402


EXP_ID = "exp201_task185_axis_selector_cost_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185
CURRENT_PUBLIC_BEST_LB = 6005.93


@dataclass(frozen=True)
class Row:
    variant: str
    status: str
    node_count: int
    memory: int | str
    params: int | str
    cost: int | str
    file_bytes: int
    sha256: str
    static_reason: str
    score_reason: str
    runtime_reason: str
    notes: str


def init_i(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.int64), name)


def init_f(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.float32), name)


def make_model(nodes: list[Any], inits: list[Any], name: str) -> bytes:
    graph = helper.make_graph(
        nodes,
        f"{name}_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, producer_name=name, ir_version=10, opset_imports=[helper.make_opsetid("", 11)])
    return model.SerializeToString()


def model_axis_conv_score_proxy() -> bytes:
    # Axis-separable proxy: aggregate channels, score all rows/cols, then score
    # 4-line windows independently. This approximates exp200's winning selector
    # without pairwise row-window x col-window interaction.
    row_line_w = np.ones((1, 1, 1, 30), dtype=np.float32)
    col_line_w = np.ones((1, 1, 30, 1), dtype=np.float32)
    row_win_w = np.ones((1, 1, 4, 1), dtype=np.float32)
    col_win_w = np.ones((1, 1, 1, 4), dtype=np.float32)
    nodes = [
        helper.make_node("ReduceSum", ["input"], ["any_color"], axes=[1], keepdims=1),
        helper.make_node("Conv", ["any_color", "row_line_w"], ["row_line_score"]),
        helper.make_node("Conv", ["any_color", "col_line_w"], ["col_line_score"]),
        helper.make_node("Conv", ["row_line_score", "row_win_w"], ["row_window_score"]),
        helper.make_node("Conv", ["col_line_score", "col_win_w"], ["col_window_score"]),
        helper.make_node("ArgMax", ["row_window_score"], ["best_row"], axis=2, keepdims=0),
        helper.make_node("ArgMax", ["col_window_score"], ["best_col"], axis=3, keepdims=0),
        helper.make_node("Cast", ["best_row"], ["best_row_f"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["best_col"], ["best_col_f"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["best_row_f", "best_col_f"], ["selector_sum"]),
        helper.make_node("ConstantOfShape", ["out_shape"], ["output"], value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])),
    ]
    inits = [
        init_f("row_line_w", row_line_w),
        init_f("col_line_w", col_line_w),
        init_f("row_win_w", row_win_w),
        init_f("col_win_w", col_win_w),
        init_i("out_shape", np.asarray([1, 10, 30, 30])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_axis_conv_score_proxy")


def model_axis_static_slice_score_proxy() -> bytes:
    # A less fused but implementation-realistic static proxy: Slice the known
    # 4-line windows by spacing family and reduce each window independently.
    nodes: list[Any] = []
    inits: list[Any] = [
        init_i("axes", np.asarray([0, 1, 2, 3])),
        init_i("out_shape", np.asarray([1, 10, 30, 30])),
    ]
    row_score_names: list[str] = []
    col_score_names: list[str] = []
    idx = 0
    for spacing, starts in [(3, [2, 5, 8, 11, 14, 17]), (4, [3, 7, 11]), (5, [4, 9])]:
        for start in starts:
            name = f"r{idx}"
            nodes.extend(
                [
                    helper.make_node("Slice", ["input", f"{name}_starts", f"{name}_ends", "axes", f"{name}_steps"], [f"{name}_slice"]),
                    helper.make_node("ReduceSum", [f"{name}_slice"], [f"{name}_score"], axes=[1, 2, 3], keepdims=0),
                ]
            )
            inits.extend(
                [
                    init_i(f"{name}_starts", np.asarray([0, 0, start, 0])),
                    init_i(f"{name}_ends", np.asarray([1, 10, start + 4 * spacing, 30])),
                    init_i(f"{name}_steps", np.asarray([1, 1, spacing, 1])),
                ]
            )
            row_score_names.append(f"{name}_score")
            idx += 1
    idx = 0
    for spacing, starts in [(3, [2, 5, 8, 11, 14, 17]), (4, [3, 7, 11]), (5, [4, 9])]:
        for start in starts:
            name = f"c{idx}"
            nodes.extend(
                [
                    helper.make_node("Slice", ["input", f"{name}_starts", f"{name}_ends", "axes", f"{name}_steps"], [f"{name}_slice"]),
                    helper.make_node("ReduceSum", [f"{name}_slice"], [f"{name}_score"], axes=[1, 2, 3], keepdims=0),
                ]
            )
            inits.extend(
                [
                    init_i(f"{name}_starts", np.asarray([0, 0, 0, start])),
                    init_i(f"{name}_ends", np.asarray([1, 10, 30, start + 4 * spacing])),
                    init_i(f"{name}_steps", np.asarray([1, 1, 1, spacing])),
                ]
            )
            col_score_names.append(f"{name}_score")
            idx += 1
    nodes.extend(
        [
            helper.make_node("Concat", row_score_names, ["row_scores"], axis=0),
            helper.make_node("Concat", col_score_names, ["col_scores"], axis=0),
            helper.make_node("ArgMax", ["row_scores"], ["best_row"], axis=0, keepdims=0),
            helper.make_node("ArgMax", ["col_scores"], ["best_col"], axis=0, keepdims=0),
            helper.make_node("ConstantOfShape", ["out_shape"], ["output"], value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])),
        ]
    )
    return make_model(nodes, inits, f"{EXP_ID}_axis_static_slice_score_proxy")


def smoke(raw: bytes) -> str:
    try:
        sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
        y = sess.run(None, {"input": np.zeros((1, 10, 30, 30), dtype=np.float32)})[0]
        return "ok" if tuple(y.shape) == (1, 10, 30, 30) else f"bad shape {tuple(y.shape)}"
    except Exception as exc:
        return f"runtime failed: {str(exc)[:180]}"


def score_raw(utils: Any, variant: str, raw: bytes, notes: str) -> Row:
    runtime_reason = smoke(raw)
    model = onnx.load_model_from_string(raw)
    ok, static_reason = infer_static_ok(model)
    if runtime_reason != "ok" or not ok:
        return Row(variant, "rejected", len(model.graph.node), "", "", "", len(raw), sha256(raw), static_reason, "", runtime_reason, notes)
    memory, params, score_reason = score_model(utils, raw, TASK_ID, variant, EXP_DIR)
    if memory is None or params is None:
        return Row(variant, "score_failed", len(model.graph.node), "", "", "", len(raw), sha256(raw), static_reason, score_reason, runtime_reason, notes)
    return Row(variant, "scored", len(model.graph.node), int(memory), int(params), int(memory) + int(params), len(raw), sha256(raw), static_reason, score_reason, runtime_reason, notes)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()[TASK_ID]
    builders = [
        ("axis_conv_score_proxy", model_axis_conv_score_proxy, "Fused Conv proxy for independent row/col 4-window scores."),
        ("axis_static_slice_score_proxy", model_axis_static_slice_score_proxy, "Static Slice+ReduceSum proxy for independent row/col 4-window scores."),
    ]
    rows: list[Row] = []
    for variant, build, notes in builders:
        raw = build()
        (EXP_DIR / f"{variant}.onnx").write_bytes(raw)
        rows.append(score_raw(utils, variant, raw, notes))

    with (EXP_DIR / "cost_probe.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])

    scored = [row for row in rows if row.status == "scored"]
    best = None if not scored else min(scored, key=lambda row: int(row.cost))
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "cost_probe_ready",
        "task_id": TASK_ID,
        "hypothesis": "exp200で成立したaxis-separable selectorのONNX cost floorはpairwise proxyより低い。",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "current_public_best_lb": CURRENT_PUBLIC_BEST_LB,
        "rows": [asdict(row) for row in rows],
        "best_scored": None if best is None else asdict(best),
        "pairwise_static_proxy_cost_from_exp117": 76194,
        "line_mask_proxy_cost_from_exp117": 4204,
        "core_stack_proxy_cost_from_exp117": 5160,
        "decision": (
            "If axis selector cost is below pairwise proxy and leaves room under baseline, "
            "attempt a correctness-first task185 ONNX lowering; otherwise pivot."
        ),
        "submission_decision": "no_submit: cost probe only",
        "leakage_risk": "low: cost proxy only.",
        "overfitting_risk": "medium-low: selector is structural, but this is not a candidate artifact.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp200で成立したtask185 axis-separable selectorについて、pairwise window scoring proxyより安いONNX cost floorになるか測る。

## 結果

- best_scored: `{None if best is None else best.variant}`
- best_cost: `{None if best is None else best.cost}`
- pairwise_static_proxy_cost_from_exp117: `76194`

## 判断

axis selectorが十分安ければ、次はcorrectness-first task185 loweringへ進む。baseline `59584` を超えるなら、selectorは発見済みでもscore laneとしては弱い。

## リスク

- leakage risk: low。cost proxyのみ。
- overfitting risk: medium-low。提出candidateではない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
