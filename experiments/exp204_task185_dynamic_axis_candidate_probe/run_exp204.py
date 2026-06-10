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

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    CandidateEval,
    evaluate_candidate,
    infer_static_ok,
    load_base_tasks,
    load_neurogolf_utils,
    score_model,
    sha256,
)


EXP_ID = "exp204_task185_dynamic_axis_candidate_probe"
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
    validation_status: str
    candidate_status: str
    candidate_reason: str
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


def window_specs() -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    for spacing in (3, 4, 5):
        max_start = 29 - 3 * spacing
        for start in range(max_start + 1):
            specs.append((start, spacing))
    return specs


def row_index_templates() -> np.ndarray:
    templates = np.zeros((len(window_specs()), 1, 10, 4, 30), dtype=np.int64)
    for wi, (start, spacing) in enumerate(window_specs()):
        for i in range(4):
            templates[wi, :, :, i, :] = start + spacing * i
    return templates


def col_index_templates() -> np.ndarray:
    templates = np.zeros((len(window_specs()), 1, 10, 4, 4), dtype=np.int64)
    for wi, (start, spacing) in enumerate(window_specs()):
        for j in range(4):
            templates[wi, :, :, :, j] = start + spacing * j
    return templates


def selector_nodes(score_source: str) -> list[Any]:
    nodes: list[Any] = [
        helper.make_node("ReduceSum", [score_source], ["any_color"], axes=[1], keepdims=1),
        helper.make_node("Conv", ["any_color", "row_line_w"], ["row_line_score"]),
        helper.make_node("Conv", ["any_color", "col_line_w"], ["col_line_score"]),
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
            helper.make_node("Gather", ["row_templates", "best_row"], ["row_idx_with_axis"], axis=0),
            helper.make_node("Gather", ["col_templates", "best_col"], ["col_idx_with_axis"], axis=0),
            helper.make_node("Squeeze", ["row_idx_with_axis"], ["row_idx"], axes=[0]),
            helper.make_node("Squeeze", ["col_idx_with_axis"], ["col_idx"], axes=[0]),
        ]
    )
    return nodes


def common_inits() -> list[Any]:
    return [
        init_f("row_line_w", np.ones((1, 1, 1, 30), dtype=np.float32)),
        init_f("col_line_w", np.ones((1, 1, 30, 1), dtype=np.float32)),
        init_f("row_win_w", np.ones((1, 1, 4, 1), dtype=np.float32)),
        init_f("col_win_w", np.ones((1, 1, 1, 4), dtype=np.float32)),
        init_i("row_templates", row_index_templates()),
        init_i("col_templates", col_index_templates()),
        init_f("w", np.ones((10, 1, 2, 2), dtype=np.float32)),
        init_f("four", np.asarray([4.0], dtype=np.float32)),
        init_f("zero", np.asarray([0.0], dtype=np.float32)),
        init_i("pads", np.asarray([0, 0, 0, 0, 0, 0, 27, 27])),
    ]


def model_dynamic_axis_basic() -> bytes:
    nodes = selector_nodes("input")
    nodes.extend(
        [
            helper.make_node("GatherElements", ["input", "row_idx"], ["selected_rows"], axis=2),
            helper.make_node("GatherElements", ["selected_rows", "col_idx"], ["lattice_4x4"], axis=3),
            helper.make_node("Conv", ["lattice_4x4", "w"], ["sum2"], group=10),
            helper.make_node("Equal", ["sum2", "four"], ["eq4"]),
            helper.make_node("Cast", ["eq4"], ["small"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["small", "pads", "zero"], ["output"], mode="constant"),
        ]
    )
    return make_model(nodes, common_inits(), f"{EXP_ID}_dynamic_axis_basic")


def model_dynamic_axis_nonzero_only() -> bytes:
    # Suppress color 0 in the final core. This is still incomplete for variable
    # background colors, but it tells us whether zero-channel leakage is the first
    # correctness failure.
    channel_mask = np.ones((1, 10, 1, 1), dtype=np.float32)
    channel_mask[:, 0, :, :] = 0.0
    nodes = selector_nodes("input")
    nodes.extend(
        [
            helper.make_node("Mul", ["input", "nonzero_channel_mask"], ["masked_input"]),
            helper.make_node("GatherElements", ["masked_input", "row_idx"], ["selected_rows"], axis=2),
            helper.make_node("GatherElements", ["selected_rows", "col_idx"], ["lattice_4x4"], axis=3),
            helper.make_node("Conv", ["lattice_4x4", "w"], ["sum2"], group=10),
            helper.make_node("Equal", ["sum2", "four"], ["eq4"]),
            helper.make_node("Cast", ["eq4"], ["small"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["small", "pads", "zero"], ["output"], mode="constant"),
        ]
    )
    inits = common_inits() + [init_f("nonzero_channel_mask", channel_mask)]
    return make_model(nodes, inits, f"{EXP_ID}_dynamic_axis_nonzero_only")


def model_dynamic_axis_score_nonzero_core_basic() -> bytes:
    # Use nonzero-channel activity only for selecting the row/col window, but
    # keep the full input for the core so output background channel 0 can still
    # be produced where expected.
    channel_mask = np.ones((1, 10, 1, 1), dtype=np.float32)
    channel_mask[:, 0, :, :] = 0.0
    nodes = [
        helper.make_node("Mul", ["input", "nonzero_channel_mask"], ["score_input"]),
    ]
    nodes.extend(selector_nodes("score_input"))
    nodes.extend(
        [
            helper.make_node("GatherElements", ["input", "row_idx"], ["selected_rows"], axis=2),
            helper.make_node("GatherElements", ["selected_rows", "col_idx"], ["lattice_4x4"], axis=3),
            helper.make_node("Conv", ["lattice_4x4", "w"], ["sum2"], group=10),
            helper.make_node("Equal", ["sum2", "four"], ["eq4"]),
            helper.make_node("Cast", ["eq4"], ["small"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["small", "pads", "zero"], ["output"], mode="constant"),
        ]
    )
    inits = common_inits() + [init_f("nonzero_channel_mask", channel_mask)]
    return make_model(nodes, inits, f"{EXP_ID}_dynamic_axis_score_nonzero_core_basic")


def model_dynamic_axis_score_nonzero_default_bg() -> bytes:
    # Select using nonzero-channel activity, compute homogeneous nonzero color
    # cells, then explicitly fill channel 0 wherever no nonzero color fires.
    channel_mask = np.ones((1, 10, 1, 1), dtype=np.float32)
    channel_mask[:, 0, :, :] = 0.0
    nodes = [
        helper.make_node("Mul", ["input", "nonzero_channel_mask"], ["score_input"]),
    ]
    nodes.extend(selector_nodes("score_input"))
    nodes.extend(
        [
            helper.make_node("GatherElements", ["score_input", "row_idx"], ["selected_rows"], axis=2),
            helper.make_node("GatherElements", ["selected_rows", "col_idx"], ["lattice_4x4"], axis=3),
            helper.make_node("Conv", ["lattice_4x4", "w"], ["sum2"], group=10),
            helper.make_node("Equal", ["sum2", "four"], ["eq4"]),
            helper.make_node("Cast", ["eq4"], ["small_nonzero"], to=TensorProto.FLOAT),
            helper.make_node("Slice", ["small_nonzero", "c1_start", "c1_end", "c_axis"], ["small_tail"]),
            helper.make_node("ReduceMax", ["small_tail"], ["any_tail"], axes=[1], keepdims=1),
            helper.make_node("Sub", ["one", "any_tail"], ["bg_small"]),
            helper.make_node("Concat", ["bg_small", "small_tail"], ["small_out"], axis=1),
            helper.make_node("Pad", ["small_out", "pads", "zero"], ["output"], mode="constant"),
        ]
    )
    inits = common_inits() + [
        init_f("nonzero_channel_mask", channel_mask),
        init_i("c1_start", np.asarray([1])),
        init_i("c1_end", np.asarray([10])),
        init_i("c_axis", np.asarray([1])),
        init_f("one", np.asarray([1.0], dtype=np.float32)),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_dynamic_axis_score_nonzero_default_bg")


def smoke(raw: bytes) -> str:
    try:
        sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
        y = sess.run(None, {"input": np.zeros((1, 10, 30, 30), dtype=np.float32)})[0]
        return "ok" if tuple(y.shape) == (1, 10, 30, 30) else f"bad shape {tuple(y.shape)}"
    except Exception as exc:
        return f"runtime failed: {str(exc)[:180]}"


def score_raw(utils: Any, variant: str, raw: bytes) -> tuple[str, int | str, int | str, int | str]:
    memory, params, reason = score_model(utils, raw, TASK_ID, variant, EXP_DIR)
    if memory is None or params is None:
        return reason, "", "", ""
    return reason, int(memory), int(params), int(memory) + int(params)


def eval_raw(utils: Any, base: BaseTask, variant: str, raw: bytes, notes: str) -> Row:
    runtime_reason = smoke(raw)
    model = onnx.load_model_from_string(raw)
    ok, static_reason = infer_static_ok(model)
    score_reason = ""
    memory: int | str = ""
    params: int | str = ""
    cost: int | str = ""
    validation_status = "not_run"
    candidate_status = "rejected"
    candidate_reason = ""
    if runtime_reason == "ok" and ok:
        score_reason, memory, params, cost = score_raw(utils, variant, raw)
        ev, _ = evaluate_candidate(
            utils,
            Candidate(TASK_ID, variant, base.route, raw, "generated", notes),
            base,
            -1,
            EXP_DIR,
        )
        validation_status = ev.validation_status
        candidate_status = ev.status
        candidate_reason = ev.reason
    else:
        candidate_reason = runtime_reason if runtime_reason != "ok" else static_reason
    return Row(
        variant,
        "scored" if cost != "" else "rejected",
        len(model.graph.node),
        memory,
        params,
        cost,
        len(raw),
        sha256(raw),
        static_reason,
        score_reason,
        runtime_reason,
        validation_status,
        candidate_status,
        candidate_reason,
        notes,
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()[TASK_ID]
    builders = [
        ("dynamic_axis_basic", model_dynamic_axis_basic, "Full dynamic axis selector plus GatherElements lattice core; no bg suppression."),
        ("dynamic_axis_nonzero_only", model_dynamic_axis_nonzero_only, "Same as basic, but suppress color-0 channel before core."),
        (
            "dynamic_axis_score_nonzero_core_basic",
            model_dynamic_axis_score_nonzero_core_basic,
            "Suppress color-0 only for selector scoring; keep full input in core so output can include channel 0 background.",
        ),
        (
            "dynamic_axis_score_nonzero_default_bg",
            model_dynamic_axis_score_nonzero_default_bg,
            "Select/compute nonzero colors, then fill channel 0 wherever no nonzero output color fires.",
        ),
    ]
    rows: list[Row] = []
    eval_rows: list[dict[str, Any]] = []
    for variant, build, notes in builders:
        raw = build()
        (EXP_DIR / f"{variant}.onnx").write_bytes(raw)
        row = eval_raw(utils, base, variant, raw, notes)
        rows.append(row)
        eval_rows.append(asdict(row))

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])

    improved = [row for row in rows if row.candidate_status == "improved"]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "candidate_probe_complete",
        "task_id": TASK_ID,
        "hypothesis": "Dynamic axis selector can be connected to GatherElements extraction and homogeneous core for task185 under baseline cost.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "current_public_best_lb": CURRENT_PUBLIC_BEST_LB,
        "rows": eval_rows,
        "accepted_count": len(improved),
        "best_improved": None if not improved else asdict(min(improved, key=lambda row: int(row.cost))),
        "decision": "If validation fails, inspect first mismatch and add bg/special masking or selector-score masking before another candidate attempt.",
        "submission_decision": "submit_candidate_after_review" if improved else "no_submit",
        "leakage_risk": "low: input-only dynamic geometry, no output lookup.",
        "overfitting_risk": "medium: task-specific geometry and bg handling still being debugged.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes_md = f"""# {EXP_ID}

## 目的

task185で、dilated axis selectorから`GatherElements`用index tensorを作り、4x4 lattice extraction + homogeneous coreまで接続したcorrectness-first candidateを検証する。

## 結果

- variants: `{[row.variant for row in rows]}`
- accepted_count: `{len(improved)}`
- validation_status: `{ {row.variant: row.validation_status for row in rows} }`
- candidate_status: `{ {row.variant: row.candidate_status for row in rows} }`
- cost: `{ {row.variant: row.cost for row in rows} }`

## 判断

full validation passなら提出候補。failなら最初のmismatchから、bg/special maskまたはselector-score maskを追加して次実験へ進む。

## リスク

- leakage risk: low。
- overfitting risk: medium。task-specific geometryを使うが、raw coordinate tableではない。
"""
    (EXP_DIR / "notes.md").write_text(notes_md, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
