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
    score_model,
    sha256,
)


EXP_ID = "exp328_task185_dynamic_bg_detector_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185


def python_bg(x: np.ndarray) -> int:
    vals = [int(v) for v in x.ravel() if int(v) != 0]
    return Counter(vals).most_common(1)[0][0]


def build_bg_detector_probe() -> bytes:
    nodes = [
        helper.make_node("ReduceSum", ["input"], ["counts_hw"], axes=[2, 3], keepdims=0),
        helper.make_node("Slice", ["counts_hw", "start1", "end10", "axis1"], ["nonzero_counts"]),
        helper.make_node("ArgMax", ["nonzero_counts"], ["bg0"], axis=1, keepdims=0),
        helper.make_node("Add", ["bg0", "one_i64"], ["bg_channel"]),
        helper.make_node("OneHot", ["bg_channel", "depth", "values"], ["bg_onehot_flat"], axis=1),
        helper.make_node("Reshape", ["bg_onehot_flat", "shape"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        f"{EXP_ID}_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 1, 1])],
        [
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "start1"),
            numpy_helper.from_array(np.asarray([10], dtype=np.int64), "end10"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axis1"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "one_i64"),
            numpy_helper.from_array(np.asarray(10, dtype=np.int64), "depth"),
            numpy_helper.from_array(np.asarray([0.0, 1.0], dtype=np.float32), "values"),
            numpy_helper.from_array(np.asarray([1, 10, 1, 1], dtype=np.int64), "shape"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name=EXP_ID,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 11)],
    )
    return model.SerializeToString()


def validate_detector(raw: bytes) -> dict[str, object]:
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    session = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    pass_count = 0
    mismatches = []
    bg_hist: Counter[int] = Counter()
    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        bg = python_bg(x)
        bg_hist[bg] += 1
        onehot = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(ex["input"]):
            for c, color in enumerate(row):
                onehot[0, int(color), r, c] = 1.0
        out = session.run(None, {"input": onehot})[0]
        pred = int(np.argmax(out.reshape(10)))
        if pred == bg:
            pass_count += 1
        elif len(mismatches) < 20:
            mismatches.append({"idx": idx, "expected_bg": bg, "pred_bg": pred})
    return {
        "status": f"{pass_count}_pass_{len(examples) - pass_count}_fail",
        "pass": pass_count,
        "total": len(examples),
        "bg_hist": dict(sorted(bg_hist.items())),
        "mismatches": mismatches,
    }


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()[TASK_ID]
    raw = build_bg_detector_probe()
    (EXP_DIR / "bg_detector_probe.onnx").write_bytes(raw)
    model = onnx.load_model_from_string(raw)
    static_ok, static_reason = infer_static_ok(model)
    detector_validation = validate_detector(raw) if static_ok else {"status": "not_run", "reason": static_reason}
    memory, params, score_reason = score_model(utils, raw, TASK_ID, "bg_detector_probe", EXP_DIR)
    cost = None if memory is None or params is None else int(memory) + int(params)
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-11",
        "status": "detector_probe_complete",
        "task_id": TASK_ID,
        "hypothesis": "task185 dynamic background is the nonzero modal input color and can be detected with a small ReduceSum+ArgMax subgraph.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "detector_validation": detector_validation,
        "detector_onnx": {
            "static_ok": static_ok,
            "static_reason": static_reason,
            "score_reason": score_reason,
            "memory": memory,
            "params": params,
            "cost": cost,
            "file_bytes": len(raw),
            "sha256": sha256(raw),
        },
        "decision": (
            "Dynamic bg detector is ready for the next task185 candidate."
            if detector_validation.get("pass") == detector_validation.get("total") and cost is not None and cost < 10000
            else "Do not build another task185 full candidate until bg detector correctness/cost is fixed."
        ),
        "submission_decision": "no_submit: detector subgraph probe only",
        "leakage_risk": "low: input-only modal-color detector; no output lookup or public feedback.",
        "overfitting_risk": "low-to-medium: modal-bg assumption is task-specific but validated over all local examples.",
        "runtime_seconds": time.time() - started,
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task185 の残課題である dynamic background detector を、入力の非ゼロ最頻色として検証し、ONNX subgraph の official cost を測る。

## 結果

- detector validation: `{result['detector_validation']['status']}`
- detector cost: `{result['detector_onnx']['cost']}`
- memory/params: `{result['detector_onnx']['memory']}` / `{result['detector_onnx']['params']}`
- bg_hist: `{result['detector_validation'].get('bg_hist')}`

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
