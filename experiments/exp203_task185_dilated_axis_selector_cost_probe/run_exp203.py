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


EXP_ID = "exp203_task185_dilated_axis_selector_cost_probe"
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


def model_dilated_axis_selector_proxy(include_argmax: bool = True) -> bytes:
    # More realistic than exp201: score 4 grid lines with spacings 3/4/5 using
    # dilated Conv over per-row/per-column specialness scores.
    row_line_w = np.ones((1, 1, 1, 30), dtype=np.float32)
    col_line_w = np.ones((1, 1, 30, 1), dtype=np.float32)
    row_win_w = np.ones((1, 1, 4, 1), dtype=np.float32)
    col_win_w = np.ones((1, 1, 1, 4), dtype=np.float32)
    nodes: list[Any] = [
        helper.make_node("ReduceSum", ["input"], ["any_color"], axes=[1], keepdims=1),
        helper.make_node("Conv", ["any_color", "row_line_w"], ["row_line_score"]),
        helper.make_node("Conv", ["any_color", "col_line_w"], ["col_line_score"]),
    ]
    for spacing in (3, 4, 5):
        nodes.append(
            helper.make_node(
                "Conv",
                ["row_line_score", "row_win_w"],
                [f"row_s{spacing}_score"],
                dilations=[spacing, 1],
            )
        )
        nodes.append(
            helper.make_node(
                "Conv",
                ["col_line_score", "col_win_w"],
                [f"col_s{spacing}_score"],
                dilations=[1, spacing],
            )
        )
        nodes.append(helper.make_node("Flatten", [f"row_s{spacing}_score"], [f"row_s{spacing}_flat"], axis=0))
        nodes.append(helper.make_node("Flatten", [f"col_s{spacing}_score"], [f"col_s{spacing}_flat"], axis=0))
    nodes.extend(
        [
            helper.make_node("Concat", ["row_s3_flat", "row_s4_flat", "row_s5_flat"], ["row_all"], axis=1),
            helper.make_node("Concat", ["col_s3_flat", "col_s4_flat", "col_s5_flat"], ["col_all"], axis=1),
        ]
    )
    if include_argmax:
        nodes.extend(
            [
                helper.make_node("ArgMax", ["row_all"], ["best_row"], axis=1, keepdims=0),
                helper.make_node("ArgMax", ["col_all"], ["best_col"], axis=1, keepdims=0),
                helper.make_node("Cast", ["best_row"], ["best_row_f"], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["best_col"], ["best_col_f"], to=TensorProto.FLOAT),
                helper.make_node("Add", ["best_row_f", "best_col_f"], ["selector_sum"]),
            ]
        )
    nodes.append(helper.make_node("ConstantOfShape", ["out_shape"], ["output"], value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])))
    inits = [
        init_f("row_line_w", row_line_w),
        init_f("col_line_w", col_line_w),
        init_f("row_win_w", row_win_w),
        init_f("col_win_w", col_win_w),
        init_i("out_shape", np.asarray([1, 10, 30, 30])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_dilated_axis_selector_proxy")


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
        ("dilated_axis_scores_only", lambda: model_dilated_axis_selector_proxy(False), "Spacing 3/4/5 dilated row/col window scores, no ArgMax."),
        ("dilated_axis_selector_argmax", lambda: model_dilated_axis_selector_proxy(True), "Spacing 3/4/5 dilated row/col window scores plus ArgMax selector ids."),
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
    extraction_core_cost = 7660
    projected_with_extraction = None if best is None else int(best.cost) + extraction_core_cost
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "cost_probe_ready",
        "task_id": TASK_ID,
        "hypothesis": "task185 realistic spacing-3/4/5 axis selector can still fit under baseline when combined with selected-lattice extraction.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "current_public_best_lb": CURRENT_PUBLIC_BEST_LB,
        "rows": [asdict(row) for row in rows],
        "best_scored": None if best is None else asdict(best),
        "extraction_core_cost_from_exp202": extraction_core_cost,
        "projected_with_extraction": projected_with_extraction,
        "projected_under_baseline": None if projected_with_extraction is None else projected_with_extraction < base.cost,
        "decision": "If under baseline, continue to correctness-first dynamic-index ONNX; otherwise selector realism erases the cost lane.",
        "submission_decision": "no_submit: cost probe only",
        "leakage_risk": "low: cost proxy only.",
        "overfitting_risk": "medium-low: spacing-specific selector mirrors known input geometry but emits no candidate.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task185 selector proxyをspacing 3/4/5のdilated 4-line windowへ近づけても、selected lattice extractionと合わせてbaseline内に収まるか測る。

## 結果

- best_scored: `{None if best is None else best.variant}`
- best_cost: `{None if best is None else best.cost}`
- extraction_core_cost_from_exp202: `{extraction_core_cost}`
- projected_with_extraction: `{projected_with_extraction}`
- baseline_cost: `{base.cost}`

## 判断

projected_under_baselineなら、次はcorrectness-first dynamic-index ONNXへ進む。

## リスク

- leakage risk: low。cost proxyのみ。
- overfitting risk: medium-low。spacing 3/4/5はtask185構造に由来するが、candidateは未生成。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
