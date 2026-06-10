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


EXP_ID = "exp215_task346_count_component_cost_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 346


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


def model_color_counts_argmin_proxy() -> bytes:
    # Count nonzero color channels, bias color0 upward, and ArgMin.
    nodes = [
        helper.make_node("ReduceSum", ["input"], ["color_counts_hw"], axes=[2, 3], keepdims=0),
        helper.make_node("Add", ["color_counts_hw", "color0_bias"], ["biased_counts"]),
        helper.make_node("ArgMin", ["biased_counts"], ["least_color"], axis=1, keepdims=0),
        helper.make_node("Cast", ["least_color"], ["least_color_f"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["least_color_f", "zero_scalar"], ["scalar"]),
        helper.make_node("ConstantOfShape", ["out_shape"], ["output"], value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])),
    ]
    inits = [
        init_f("color0_bias", np.asarray([[9999.0] + [0.0] * 9], dtype=np.float32)),
        init_f("zero_scalar", np.asarray([0.0], dtype=np.float32)),
        init_i("out_shape", np.asarray([1, 10, 30, 30])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_color_counts_argmin_proxy")


def model_component_growth_proxy(steps: int) -> bytes:
    # Cost proxy for repeated 4-neighbor dilation used to approximate connected
    # component reachability/largest component. This intentionally operates on
    # all 10 channels to reveal whether component-size logic is viable.
    weight = np.zeros((10, 1, 3, 3), dtype=np.float32)
    weight[:, 0, 1, 1] = 1.0
    weight[:, 0, 0, 1] = 1.0
    weight[:, 0, 2, 1] = 1.0
    weight[:, 0, 1, 0] = 1.0
    weight[:, 0, 1, 2] = 1.0
    nodes: list[Any] = []
    prev = "input"
    for i in range(steps):
        nodes.extend(
            [
                helper.make_node("Conv", [prev, "w"], [f"nbr{i}"], group=10, pads=[1, 1, 1, 1]),
                helper.make_node("Greater", [f"nbr{i}", "zero"], [f"gt{i}"]),
                helper.make_node("Cast", [f"gt{i}"], [f"dil{i}"], to=TensorProto.FLOAT),
                helper.make_node("Mul", [f"dil{i}", "input"], [f"masked{i}"]),
            ]
        )
        prev = f"masked{i}"
    nodes.extend(
        [
            helper.make_node("ReduceSum", [prev], ["component_scores"], axes=[2, 3], keepdims=0),
            helper.make_node("ConstantOfShape", ["out_shape"], ["output"], value=helper.make_tensor("zero_out", TensorProto.FLOAT, [1], [0.0])),
        ]
    )
    inits = [init_f("w", weight), init_f("zero", np.asarray([0.0])), init_i("out_shape", np.asarray([1, 10, 30, 30]))]
    return make_model(nodes, inits, f"{EXP_ID}_component_growth_{steps}")


def model_onecell_output_floor() -> bytes:
    # Minimal 1x1 one-hot-like output placed at top-left with channel0 default.
    small = np.zeros((1, 10, 1, 1), dtype=np.float32)
    small[:, 0, 0, 0] = 1.0
    nodes = [
        helper.make_node("Constant", [], ["small"], value=numpy_helper.from_array(small, "small_value")),
        helper.make_node("Pad", ["small", "pads", "zero"], ["output"], mode="constant"),
    ]
    inits = [init_i("pads", np.asarray([0, 0, 0, 0, 0, 0, 29, 29])), init_f("zero", np.asarray([0.0]))]
    return make_model(nodes, inits, f"{EXP_ID}_onecell_output_floor")


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
        ("onecell_output_floor", model_onecell_output_floor, "1x1 output floor with Pad to official 30x30 tensor."),
        ("color_counts_argmin_proxy", model_color_counts_argmin_proxy, "Count color channels and ArgMin nonzero color."),
        ("component_growth_1", lambda: model_component_growth_proxy(1), "One 4-neighbor dilation/reachability proxy across channels."),
        ("component_growth_4", lambda: model_component_growth_proxy(4), "Four-step component proxy across channels."),
        ("component_growth_8", lambda: model_component_growth_proxy(8), "Eight-step component proxy across channels."),
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
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "cost_probe_ready",
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "rows": [asdict(row) for row in rows],
        "best_scored": None if not scored else asdict(min(scored, key=lambda row: int(row.cost))),
        "decision": "If component proxy is too expensive, do not lower task346 directly; keep as solved-rule asset and pivot.",
        "submission_decision": "no_submit: cost probe only",
        "leakage_risk": "low: cost proxy only.",
        "overfitting_risk": "low: no candidate emitted.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task346 ruleのONNX化に必要なcolor countとcomponent-size proxyのcost floorを測る。

## 結果

- baseline_cost: `{base.cost}`
- rows: see `result.json`

## 判断

component proxyがbaseline/gainに対して重いなら、task346はsolved-rule assetとして保持し、直接loweringは保留する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
