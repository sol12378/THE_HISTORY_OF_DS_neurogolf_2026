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


EXP_ID = "exp117_task185_window_scoring_cost_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185
CURRENT_LOCAL_ESTIMATE = 6282.93615685804
CAMPAIGN_INDEX = 19


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


def model_line_mask_conv_proxy() -> bytes:
    # Approximate grid-line detection cost: aggregate color channels, then count
    # nonzero/background-like activity along rows/cols with large kernels.
    row_w = np.ones((1, 1, 1, 30), dtype=np.float32)
    col_w = np.ones((1, 1, 30, 1), dtype=np.float32)
    nodes = [
        helper.make_node("ReduceSum", ["input"], ["any_color"], axes=[1], keepdims=1),
        helper.make_node("Conv", ["any_color", "row_w"], ["row_count"]),
        helper.make_node("Conv", ["any_color", "col_w"], ["col_count"]),
        helper.make_node("Add", ["row_count", "row_zero"], ["row_pad"]),
        helper.make_node("Add", ["col_count", "col_zero"], ["col_pad"]),
        helper.make_node("ConstantOfShape", ["shape"], ["output"], value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])),
    ]
    inits = [
        init_f("row_w", row_w),
        init_f("col_w", col_w),
        init_f("row_zero", np.zeros((1, 1, 30, 1), dtype=np.float32)),
        init_f("col_zero", np.zeros((1, 1, 1, 30), dtype=np.float32)),
        init_i("shape", np.asarray([1, 10, 30, 30])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_line_mask_conv_proxy")


def model_window_score_static_positions_proxy() -> bytes:
    # Score all possible 4x4 window pairs from exp115 line families as a tiny
    # vector. This is intentionally static-position but measures selector vector
    # plumbing: GatherND -> ReduceSum -> ArgMax.
    coords: list[list[int]] = []
    for spacing, starts in [(3, [2, 5, 8, 11, 14, 17]), (4, [3, 7, 11]), (5, [4, 9])]:
        row_windows = [tuple(s + spacing * k for k in range(4)) for s in starts]
        col_windows = row_windows
        for rr in row_windows:
            for cc in col_windows:
                for r in rr:
                    for c in cc:
                        coords.append([0, r, c])
    max_windows = len(coords) // 16
    nodes = [
        helper.make_node("Transpose", ["input"], ["nhwc"], perm=[0, 2, 3, 1]),
        helper.make_node("GatherND", ["nhwc", "coords"], ["cells"]),
        helper.make_node("ReduceSum", ["cells"], ["cell_scores"], axes=[1], keepdims=0),
        helper.make_node("Reshape", ["cell_scores", "score_shape"], ["window_cells"]),
        helper.make_node("ReduceSum", ["window_cells"], ["window_scores"], axes=[1], keepdims=0),
        helper.make_node("ArgMax", ["window_scores"], ["best_idx"], axis=0, keepdims=0),
        helper.make_node("ConstantOfShape", ["out_shape"], ["output"], value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])),
    ]
    inits = [
        init_i("coords", np.asarray(coords, dtype=np.int64)),
        init_i("score_shape", np.asarray([max_windows, 16])),
        init_i("out_shape", np.asarray([1, 10, 30, 30])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_window_score_static_positions_proxy")


def model_three_branch_core_stack_proxy() -> bytes:
    # Measure the cost of computing three spacing-specific cores simultaneously.
    # This approximates a branchless implementation where selection happens
    # after candidate outputs are produced.
    nodes = []
    inits: list[Any] = [init_i("axes", np.asarray([0, 1, 2, 3])), init_f("zero", np.asarray([0.0]))]
    weight = np.zeros((10, 1, 2, 2), dtype=np.float32)
    weight[:, 0, :, :] = 1.0
    inits.append(init_f("w", weight))
    inits.append(init_f("four", np.asarray([4.0])))
    for name, spacing, start in [("s3", 3, 2), ("s4", 4, 3), ("s5", 5, 4)]:
        nodes.extend(
            [
                helper.make_node("Slice", ["input", f"{name}_starts", f"{name}_ends", "axes", f"{name}_steps"], [f"{name}_m4"]),
                helper.make_node("Conv", [f"{name}_m4", "w"], [f"{name}_sum2"], group=10),
                helper.make_node("Equal", [f"{name}_sum2", "four"], [f"{name}_eq4"]),
                helper.make_node("Cast", [f"{name}_eq4"], [f"{name}_small"], to=TensorProto.FLOAT),
            ]
        )
        inits.extend(
            [
                init_i(f"{name}_starts", np.asarray([0, 0, start, start])),
                init_i(f"{name}_ends", np.asarray([1, 10, start + 4 * spacing, start + 4 * spacing])),
                init_i(f"{name}_steps", np.asarray([1, 1, spacing, spacing])),
            ]
        )
    nodes.extend(
        [
            helper.make_node("Add", ["s3_small", "s4_small"], ["sum34"]),
            helper.make_node("Add", ["sum34", "s5_small"], ["small_sum"]),
            helper.make_node("Pad", ["small_sum", "pads", "zero"], ["output"], mode="constant"),
        ]
    )
    inits.append(init_i("pads", np.asarray([0, 0, 0, 0, 0, 0, 27, 27])))
    return make_model(nodes, inits, f"{EXP_ID}_three_branch_core_stack_proxy")


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
        ("line_mask_conv_proxy", model_line_mask_conv_proxy, "Cost proxy for row/col line detection with large Conv kernels."),
        ("window_score_static_positions_proxy", model_window_score_static_positions_proxy, "Cost proxy for scoring all static 4x4 window pairs with GatherND/ArgMax."),
        ("three_branch_core_stack_proxy", model_three_branch_core_stack_proxy, "Cost proxy for branchless spacing 3/4/5 core computation before selection."),
    ]
    rows = []
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
        "campaign_index": CAMPAIGN_INDEX,
        "task_id": TASK_ID,
        "hypothesis": "Measure ONNX cost floors for task185 window scoring/selection before implementing a full dynamic compiler.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "rows": [asdict(row) for row in rows],
        "best_scored": None if not scored else asdict(min(scored, key=lambda row: int(row.cost))),
        "base_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "local_estimate_delta": 0.0,
        "new_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "decision": "If selector plumbing is too expensive, avoid dynamic window scoring in ONNX and use lower-cost shape/spacing specialization or another high-gain task.",
        "submission_decision": "no_submit: cost probe only",
        "leakage_risk": "low: input-only cost proxies.",
        "overfitting_risk": "medium: static-position proxy is diagnostic, not a candidate.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## Hypothesis

task185 window selectorをONNX化する前に、line detection、window scoring、spacing branch coreのcost floorを測る。

## Result

- rows: see `result.json` / `cost_probe.csv`
- local delta: `0.000000`

## Interpretation

selectorが高costなら、full dynamic ONNXではなく、shape/spacing特化または別高gain taskへpivotする。

## Risk

- leakage risk: low。
- overfitting risk: medium。static position proxyは診断用。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
