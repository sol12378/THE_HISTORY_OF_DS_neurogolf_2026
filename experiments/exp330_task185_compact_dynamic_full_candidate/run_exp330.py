from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    evaluate_candidate,
    infer_static_ok,
    load_base_tasks,
    load_neurogolf_utils,
    point,
    score_model,
    sha256,
)


EXP_ID = "exp330_task185_compact_dynamic_full_candidate"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185
CURRENT_PUBLIC_BEST_LB = 6008.96


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
    validation_status: str
    candidate_status: str
    candidate_reason: str
    local_delta_if_any: float | str
    notes: str


def init_i(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.int64), name)


def init_f(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.float32), name)


def window_specs() -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    for spacing in (3, 4, 5):
        max_start = 29 - 3 * spacing
        for start in range(max_start + 1):
            specs.append((start, spacing))
    return specs


def make_model(
    nodes: list[Any],
    inits: list[Any],
    name: str,
    output_shape: list[int] | None = None,
    output_type: int = TensorProto.FLOAT,
) -> bytes:
    if output_shape is None:
        output_shape = [1, 10, 30, 30]
    graph = helper.make_graph(
        nodes,
        f"{name}_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", output_type, output_shape)],
        inits,
    )
    model = helper.make_model(graph, producer_name=name, ir_version=10, opset_imports=[helper.make_opsetid("", 11)])
    return model.SerializeToString()


def common_selector_nodes() -> list[Any]:
    nodes: list[Any] = [
        helper.make_node("ReduceSum", ["input"], ["counts_hw"], axes=[2, 3], keepdims=0),
        helper.make_node("Slice", ["counts_hw", "start1", "end10", "axis1"], ["nonzero_counts"]),
        helper.make_node("ArgMax", ["nonzero_counts"], ["bg0"], axis=1, keepdims=0),
        helper.make_node("Add", ["bg0", "one_i64"], ["bg_channel"]),
        helper.make_node("OneHot", ["bg_channel", "depth", "onehot_values"], ["bg_onehot_flat"], axis=1),
        helper.make_node("Reshape", ["bg_onehot_flat", "bg_shape"], ["bg_onehot"]),
        helper.make_node("Sub", ["one_f", "bg_onehot"], ["not_bg"]),
        helper.make_node("Mul", ["not_bg", "nonzero_channel_mask"], ["special_channel_mask"]),
        helper.make_node("Mul", ["input", "special_channel_mask"], ["score_input"]),
        helper.make_node("ReduceSum", ["score_input"], ["any_special"], axes=[1], keepdims=1),
        helper.make_node("Conv", ["any_special", "row_line_w"], ["row_line_score"]),
        helper.make_node("Conv", ["any_special", "col_line_w"], ["col_line_score"]),
    ]
    for spacing in (3, 4, 5):
        nodes.extend(
            [
                helper.make_node("Conv", ["row_line_score", "row_win_w"], [f"row_s{spacing}_score"], dilations=[spacing, 1]),
                helper.make_node("Conv", ["col_line_score", "col_win_w"], [f"col_s{spacing}_score"], dilations=[1, spacing]),
                helper.make_node("Flatten", [f"row_s{spacing}_score"], [f"row_s{spacing}_flat"], axis=0),
                helper.make_node("Flatten", [f"col_s{spacing}_score"], [f"col_s{spacing}_flat"], axis=0),
            ]
        )
    nodes.extend(
        [
            helper.make_node("Concat", ["row_s3_flat", "row_s4_flat", "row_s5_flat"], ["row_all"], axis=1),
            helper.make_node("Concat", ["col_s3_flat", "col_s4_flat", "col_s5_flat"], ["col_all"], axis=1),
            helper.make_node("ArgMax", ["row_all"], ["best_row"], axis=1, keepdims=0),
            helper.make_node("ArgMax", ["col_all"], ["best_col"], axis=1, keepdims=0),
        ]
    )
    return nodes


def compact_index_nodes() -> list[Any]:
    # Build row_idx [1,10,4,30] and col_idx [1,10,4,4] from compact
    # start/spacing arrays. This replaces exp204's huge row/col templates.
    return [
        helper.make_node("Gather", ["start_values", "best_row"], ["row_start"]),
        helper.make_node("Gather", ["spacing_values", "best_row"], ["row_spacing"]),
        helper.make_node("Gather", ["start_values", "best_col"], ["col_start"]),
        helper.make_node("Gather", ["spacing_values", "best_col"], ["col_spacing"]),
        helper.make_node("Mul", ["row_spacing", "steps4"], ["row_offsets"]),
        helper.make_node("Add", ["row_start", "row_offsets"], ["row_vec"]),
        helper.make_node("Mul", ["col_spacing", "steps4"], ["col_offsets"]),
        helper.make_node("Add", ["col_start", "col_offsets"], ["col_vec"]),
        helper.make_node("Reshape", ["row_vec", "row_vec_shape"], ["row_idx_seed"]),
        helper.make_node("Reshape", ["col_vec", "col_vec_shape"], ["col_idx_seed"]),
        helper.make_node("Tile", ["row_idx_seed", "row_repeats"], ["row_idx"]),
        helper.make_node("Tile", ["col_idx_seed", "col_repeats"], ["col_idx"]),
    ]


def build_candidate(*, output_uint8: bool) -> bytes:
    nodes = common_selector_nodes()
    nodes.extend(compact_index_nodes())
    nodes.extend(
        [
            helper.make_node("GatherElements", ["score_input", "row_idx"], ["selected_rows"], axis=2),
            helper.make_node("GatherElements", ["selected_rows", "col_idx"], ["lattice_4x4"], axis=3),
            helper.make_node("Conv", ["lattice_4x4", "core_w"], ["sum2"], group=10),
            helper.make_node("Equal", ["sum2", "four"], ["eq4"]),
            helper.make_node("Cast", ["eq4"], ["small_nonzero"], to=TensorProto.FLOAT),
            helper.make_node("Slice", ["small_nonzero", "c1_start", "c1_end", "axis1"], ["small_tail"]),
            helper.make_node("ReduceMax", ["small_tail"], ["any_tail"], axes=[1], keepdims=1),
            helper.make_node("Sub", ["one_f", "any_tail"], ["bg_small"]),
            helper.make_node("Concat", ["bg_small", "small_tail"], ["small_out"], axis=1),
        ]
    )
    if output_uint8:
        nodes.extend(
            [
                helper.make_node("Pad", ["small_out", "pads", "zero"], ["output_f"], mode="constant"),
                helper.make_node("Cast", ["output_f"], ["output"], to=TensorProto.UINT8),
            ]
        )
    else:
        nodes.append(helper.make_node("Pad", ["small_out", "pads", "zero"], ["output"], mode="constant"))
    specs = window_specs()
    nonzero_mask = np.ones((1, 10, 1, 1), dtype=np.float32)
    nonzero_mask[:, 0, :, :] = 0.0
    core_w = np.ones((10, 1, 2, 2), dtype=np.float32)
    inits = [
        init_i("start1", np.asarray([1])),
        init_i("end10", np.asarray([10])),
        init_i("axis1", np.asarray([1])),
        init_i("one_i64", np.asarray([1])),
        init_i("depth", np.asarray(10)),
        init_f("onehot_values", np.asarray([0.0, 1.0])),
        init_i("bg_shape", np.asarray([1, 10, 1, 1])),
        init_f("one_f", np.asarray([1.0])),
        init_f("nonzero_channel_mask", nonzero_mask),
        init_f("row_line_w", np.ones((1, 1, 1, 30), dtype=np.float32)),
        init_f("col_line_w", np.ones((1, 1, 30, 1), dtype=np.float32)),
        init_f("row_win_w", np.ones((1, 1, 4, 1), dtype=np.float32)),
        init_f("col_win_w", np.ones((1, 1, 1, 4), dtype=np.float32)),
        init_i("start_values", np.asarray([s for s, _ in specs])),
        init_i("spacing_values", np.asarray([sp for _, sp in specs])),
        init_i("steps4", np.asarray([0, 1, 2, 3])),
        init_i("row_vec_shape", np.asarray([1, 1, 4, 1])),
        init_i("col_vec_shape", np.asarray([1, 1, 1, 4])),
        init_i("row_repeats", np.asarray([1, 10, 1, 30])),
        init_i("col_repeats", np.asarray([1, 10, 4, 1])),
        init_f("core_w", core_w),
        init_f("four", np.asarray([4.0])),
        init_i("c1_start", np.asarray([1])),
        init_i("c1_end", np.asarray([10])),
        init_f("zero", np.asarray([0.0])),
        init_i("pads", np.asarray([0, 0, 0, 0, 0, 0, 27, 27])),
    ]
    return make_model(
        nodes,
        inits,
        f"{EXP_ID}_compact_dynamic_{'uint8' if output_uint8 else 'float'}",
        output_shape=[1, 10, 30, 30],
        output_type=TensorProto.UINT8 if output_uint8 else TensorProto.FLOAT,
    )


def smoke(raw: bytes) -> str:
    try:
        sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
        y = sess.run(None, {"input": np.zeros((1, 10, 30, 30), dtype=np.float32)})[0]
        return "ok" if tuple(y.shape) == (1, 10, 30, 30) else f"bad shape {tuple(y.shape)}"
    except Exception as exc:
        return f"runtime failed: {str(exc)[:180]}"


def eval_raw(utils: Any, base: BaseTask, variant: str, raw: bytes) -> Row:
    runtime_reason = smoke(raw)
    model = onnx.load_model_from_string(raw)
    ok, static_reason = infer_static_ok(model)
    memory: int | str = ""
    params: int | str = ""
    cost: int | str = ""
    score_reason = ""
    validation_status = "not_run"
    candidate_status = "rejected"
    candidate_reason = runtime_reason if runtime_reason != "ok" else static_reason
    local_delta: float | str = ""
    if runtime_reason == "ok" and ok:
        memory_raw, params_raw, score_reason = score_model(utils, raw, TASK_ID, variant, EXP_DIR)
        if memory_raw is not None and params_raw is not None:
            memory = int(memory_raw)
            params = int(params_raw)
            cost = memory + params
            local_delta = point(cost) - base.points
        ev, _ = evaluate_candidate(
            utils,
            Candidate(TASK_ID, variant, base.route, raw, "generated", "compact dynamic bg selector + GatherElements lattice core"),
            base,
            -1,
            EXP_DIR,
        )
        validation_status = ev.validation_status
        candidate_status = ev.status
        candidate_reason = ev.reason
    return Row(
        variant=variant,
        status="scored" if cost != "" else "rejected",
        node_count=len(model.graph.node),
        memory=memory,
        params=params,
        cost=cost,
        file_bytes=len(raw),
        sha256=sha256(raw),
        static_reason=static_reason,
        score_reason=score_reason,
        runtime_reason=runtime_reason,
        validation_status=validation_status,
        candidate_status=candidate_status,
        candidate_reason=candidate_reason,
        local_delta_if_any=local_delta,
        notes="compact start/spacing dynamic index candidate",
    )


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()[TASK_ID]
    variants = [
        ("compact_dynamic_float", build_candidate(output_uint8=False)),
        ("compact_dynamic_uint8", build_candidate(output_uint8=True)),
    ]
    rows = []
    for variant, raw in variants:
        (EXP_DIR / f"{variant}.onnx").write_bytes(raw)
        rows.append(eval_raw(utils, base, variant, raw))
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])
    improved_rows = [row for row in rows if row.candidate_status == "improved"]
    best_row = min(rows, key=lambda r: float("inf") if r.cost == "" else int(r.cost))
    best_improved = None if not improved_rows else min(improved_rows, key=lambda r: int(r.cost))
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-11",
        "status": "candidate_probe_complete",
        "task_id": TASK_ID,
        "hypothesis": "A compact start/spacing dynamic-index task185 candidate can connect exp329's bg-aware selector to lattice extraction and homogeneous 2x2 core without huge row/col templates.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "current_public_best_lb": CURRENT_PUBLIC_BEST_LB,
        "rows": [asdict(row) for row in rows],
        "best_row": asdict(best_row),
        "accepted_count": len(improved_rows),
        "best_improved": None if best_improved is None else asdict(best_improved),
        "expected_local_delta": 0.0 if best_improved is None else best_improved.local_delta_if_any,
        "expected_public_lb_if_calibrated": CURRENT_PUBLIC_BEST_LB + (0.0 if best_improved is None else float(best_improved.local_delta_if_any)),
        "decision": (
            "Submit after bundle replay if improved and full-arc valid."
            if best_improved is not None
            else "No submit. Use validation/cost failure to decide whether to debug task185 output core or pivot within GridSample queue."
        ),
        "submission_decision": "submit_after_review" if best_improved is not None else "no_submit",
        "leakage_risk": "low: input-only bg detector, window selector, and homogeneous local core; no output lookup or public feedback.",
        "overfitting_risk": "medium: task-specific geometry and dynamic index construction are validated locally but not yet public-calibrated.",
        "runtime_seconds": time.time() - started,
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp329 の dynamic bg axis selector を full candidate に接続する。exp204 の巨大 `row_templates` / `col_templates` を避けるため、`best_row/best_col -> start/spacing -> Tile` で `GatherElements` index を動的生成する。

## 結果

- rows: `{ {r.variant: {'validation': r.validation_status, 'status': r.candidate_status, 'cost': r.cost} for r in rows} }`
- best_row: `{best_row.variant}`
- baseline_cost: `{base.cost}`
- best_cost: `{best_row.cost}`
- best_reason: `{best_row.candidate_reason}`

## 判断

{result['decision']}

## Submission

`{result['submission_decision']}`

## Risk

- leakage risk: {result['leakage_risk']}
- overfitting risk: {result['overfitting_risk']}
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
