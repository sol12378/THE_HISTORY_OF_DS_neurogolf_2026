from __future__ import annotations

import json
import pathlib
import sys
import time
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from phase1_rewrite_utils import (  # noqa: E402
    grid_to_array,
    infer_static_ok,
    load_base_tasks,
    load_neurogolf_utils,
    load_task,
    one_hot_padded,
    score_model,
    sha256,
)


EXP_ID = "exp329_task185_dynamic_bg_axis_selector_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185


def python_bg(arr: np.ndarray) -> int:
    vals = [int(v) for v in arr.ravel() if int(v) != 0]
    return Counter(vals).most_common(1)[0][0]


def grid_lines(arr: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    h, w = arr.shape
    rows = [r for r in range(h) if int((arr[r, :] == bg).sum()) > w * 0.8]
    cols = [c for c in range(w) if int((arr[:, c] == bg).sum()) > h * 0.8]
    return rows, cols


def windows(lines: list[int]) -> list[tuple[int, int, int, int]]:
    out = []
    for i in range(len(lines) - 3):
        win = tuple(lines[i : i + 4])
        delta = [win[j + 1] - win[j] for j in range(3)]
        if len(set(delta)) == 1 and delta[0] in (3, 4, 5):
            out.append(win)
    return out


def axis_scores(arr: np.ndarray, bg: int, row_windows: list[tuple[int, ...]], col_windows: list[tuple[int, ...]]):
    row_stats = []
    col_stats = []
    for rr in row_windows:
        special = (arr[np.array(rr), :] != 0) & (arr[np.array(rr), :] != bg)
        row_stats.append({"win": rr, "score": int(special.sum()), "start": rr[0]})
    for cc in col_windows:
        special = (arr[:, np.array(cc)] != 0) & (arr[:, np.array(cc)] != bg)
        col_stats.append({"win": cc, "score": int(special.sum()), "start": cc[0]})
    return row_stats, col_stats


def choose_axis(stats: list[dict]) -> tuple[int, ...] | None:
    if not stats:
        return None
    return tuple(max(stats, key=lambda s: (s["score"], -s["start"]))["win"])


def window_specs() -> list[tuple[int, int]]:
    specs = []
    for spacing in (3, 4, 5):
        max_start = 29 - 3 * spacing
        for start in range(max_start + 1):
            specs.append((start, spacing))
    return specs


def spec_to_window(spec_index: int) -> tuple[int, int, int, int]:
    start, spacing = window_specs()[spec_index]
    return tuple(start + spacing * i for i in range(4))


def build_selector_probe(*, diagnostics: bool) -> bytes:
    nodes = [
        helper.make_node("ReduceSum", ["input"], ["counts_hw"], axes=[2, 3], keepdims=0),
        helper.make_node("Slice", ["counts_hw", "start1", "end10", "axis1"], ["nonzero_counts"]),
        helper.make_node("ArgMax", ["nonzero_counts"], ["bg0"], axis=1, keepdims=0),
        helper.make_node("Add", ["bg0", "one_i64"], ["bg_channel"]),
        helper.make_node("OneHot", ["bg_channel", "depth", "values"], ["bg_onehot_flat"], axis=1),
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
            helper.make_node("Cast", ["best_row"], ["best_row_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["best_col"], ["best_col_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["bg_channel"], ["bg_channel_f"], to=TensorProto.FLOAT),
            helper.make_node("Add", ["best_row_f", "best_col_f"], ["selector_sum0"]),
            helper.make_node("Add", ["selector_sum0", "bg_channel_f"], ["selector_sum"]),
            helper.make_node("Mul", ["selector_sum", "zero_f"], ["selector_zero"]),
            helper.make_node("Add", ["score_input", "selector_zero"], ["output"]),
        ]
    )
    nonzero_mask = np.ones((1, 10, 1, 1), dtype=np.float32)
    nonzero_mask[:, 0, :, :] = 0.0
    inits = [
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "start1"),
        numpy_helper.from_array(np.asarray([10], dtype=np.int64), "end10"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axis1"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "one_i64"),
        numpy_helper.from_array(np.asarray(10, dtype=np.int64), "depth"),
        numpy_helper.from_array(np.asarray([0.0, 1.0], dtype=np.float32), "values"),
        numpy_helper.from_array(np.asarray([1, 10, 1, 1], dtype=np.int64), "bg_shape"),
        numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), "one_f"),
        numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero_f"),
        numpy_helper.from_array(nonzero_mask, "nonzero_channel_mask"),
        numpy_helper.from_array(np.ones((1, 1, 1, 30), dtype=np.float32), "row_line_w"),
        numpy_helper.from_array(np.ones((1, 1, 30, 1), dtype=np.float32), "col_line_w"),
        numpy_helper.from_array(np.ones((1, 1, 4, 1), dtype=np.float32), "row_win_w"),
        numpy_helper.from_array(np.ones((1, 1, 1, 4), dtype=np.float32), "col_win_w"),
    ]
    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30]),
    ]
    if diagnostics:
        outputs = [
            helper.make_tensor_value_info("best_row", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("best_col", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("bg_channel", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("row_all", TensorProto.FLOAT, [1, len(window_specs())]),
            helper.make_tensor_value_info("col_all", TensorProto.FLOAT, [1, len(window_specs())]),
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30]),
        ]
    graph = helper.make_graph(
        nodes,
        f"{EXP_ID}_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        outputs,
        inits,
    )
    model = helper.make_model(graph, producer_name=EXP_ID, ir_version=10, opset_imports=[helper.make_opsetid("", 11)])
    return model.SerializeToString()


def score_probe(utils, raw: bytes) -> dict[str, object]:
    memory, params, reason = score_model(utils, raw, TASK_ID, "selector_probe", EXP_DIR)
    cost = None if memory is None or params is None else int(memory) + int(params)
    return {"score_reason": reason, "memory": memory, "params": params, "cost": cost}


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()[TASK_ID]
    raw = build_selector_probe(diagnostics=True)
    score_raw = build_selector_probe(diagnostics=False)
    (EXP_DIR / "selector_probe.onnx").write_bytes(raw)
    (EXP_DIR / "selector_score_probe.onnx").write_bytes(score_raw)
    model = onnx.load_model_from_string(raw)
    score_model_proto = onnx.load_model_from_string(score_raw)
    static_ok, static_reason = (False, "")
    static_ok, static_reason = infer_static_ok(model)
    score_static_ok, score_static_reason = infer_static_ok(score_model_proto)

    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    rows = []
    pass_count = 0
    session = ort.InferenceSession(raw, providers=["CPUExecutionProvider"]) if static_ok else None
    for idx, ex in enumerate(examples):
        x = grid_color_arr = grid_to_array(ex["input"])
        bg = python_bg(x)
        line_rows, line_cols = grid_lines(x, bg)
        py_rr = choose_axis(axis_scores(x, bg, windows(line_rows), windows(line_cols))[0])
        py_cc = choose_axis(axis_scores(x, bg, windows(line_rows), windows(line_cols))[1])
        pred_bg = pred_rr = pred_cc = None
        row_ok = col_ok = bg_ok = False
        if session is not None:
            best_row, best_col, bg_channel, *_ = session.run(None, {"input": one_hot_padded(ex["input"])})
            pred_bg = int(bg_channel.reshape(-1)[0])
            pred_rr = spec_to_window(int(best_row.reshape(-1)[0]))
            pred_cc = spec_to_window(int(best_col.reshape(-1)[0]))
            bg_ok = pred_bg == bg
            row_ok = pred_rr == py_rr
            col_ok = pred_cc == py_cc
        ok = bg_ok and row_ok and col_ok
        pass_count += int(ok)
        if not ok and len(rows) < 30:
            rows.append(
                {
                    "idx": idx,
                    "bg": bg,
                    "pred_bg": pred_bg,
                    "py_rr": py_rr,
                    "pred_rr": pred_rr,
                    "py_cc": py_cc,
                    "pred_cc": pred_cc,
                    "bg_ok": bg_ok,
                    "row_ok": row_ok,
                    "col_ok": col_ok,
                }
            )

    score = score_probe(utils, score_raw) if score_static_ok else {"score_reason": score_static_reason, "memory": None, "params": None, "cost": None}
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-11",
        "status": "selector_probe_complete",
        "task_id": TASK_ID,
        "hypothesis": "Connecting the dynamic bg detector to task185's dilated axis selector should reproduce the Python row/col windows while remaining far below baseline cost.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "static_ok": static_ok,
        "static_reason": static_reason,
        "score_static_ok": score_static_ok,
        "score_static_reason": score_static_reason,
        "selector_validation": {
            "status": f"{pass_count}_pass_{len(examples) - pass_count}_fail",
            "pass": pass_count,
            "total": len(examples),
            "first_failures": rows,
        },
        "selector_onnx": {
            **score,
            "file_bytes": len(raw),
            "sha256": sha256(raw),
            "score_file_bytes": len(score_raw),
            "score_sha256": sha256(score_raw),
        },
        "decision": (
            "Selector is ready to connect to compact extraction/core."
            if pass_count == len(examples) and score.get("cost") is not None and score["cost"] < base.cost
            else "Do not build the full task185 candidate yet; selector correctness or cost still needs fixing."
        ),
        "submission_decision": "no_submit: selector subgraph probe only",
        "leakage_risk": "low: input-only bg and geometry selector; no output lookup or public feedback.",
        "overfitting_risk": "medium-low: selector rule is task-specific but checked across all local examples.",
        "runtime_seconds": time.time() - started,
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp328 の dynamic bg detector を task185 dilated axis selector に接続し、Python selector と全例一致するか、official cost が baseline 未満かを確認する。

## 結果

- selector validation: `{result['selector_validation']['status']}`
- selector cost: `{result['selector_onnx']['cost']}`
- memory/params: `{result['selector_onnx']['memory']}` / `{result['selector_onnx']['params']}`
- static: `{static_reason}`

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
