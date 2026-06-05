from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
EXP_DIR = ROOT / "experiments" / "exp001_baseline"
TASK_NUM = 87
TASK_ID = f"task{TASK_NUM:03d}"
ONNX_PATH = EXP_DIR / f"{TASK_ID}.onnx"
SUBMISSION_PATH = EXP_DIR / "submission.zip"
TRACE_PREFIX = str(EXP_DIR / TASK_ID)


def load_neurogolf_utils():
    path = DATA_DIR / "neurogolf_utils" / "neurogolf_utils.py"
    spec = importlib.util.spec_from_file_location("neurogolf_utils", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["neurogolf_utils"] = module
    spec.loader.exec_module(module)
    return module


def build_rot180_3x3_model() -> onnx.ModelProto:
    grid_shape = [1, 10, 30, 30]
    reverse_first_three = np.array([2, 1, 0, *range(3, 30)], dtype=np.int64)

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, grid_shape)
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, grid_shape)
    row_idx = helper.make_tensor(
        "row_idx", TensorProto.INT64, [30], reverse_first_three.tolist()
    )
    col_idx = helper.make_tensor(
        "col_idx", TensorProto.INT64, [30], reverse_first_three.tolist()
    )
    gather_rows = helper.make_node(
        "Gather", ["input", "row_idx"], ["row_reversed"], axis=2
    )
    gather_cols = helper.make_node(
        "Gather", ["row_reversed", "col_idx"], ["output"], axis=3
    )
    graph = helper.make_graph(
        [gather_rows, gather_cols],
        "task087_rot180_3x3",
        [x],
        [y],
        [row_idx, col_idx],
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def verify_subset(utils, session: ort.InferenceSession, examples: list[dict]) -> tuple[int, int]:
    passed = 0
    failed = 0
    for example in examples:
        benchmark = utils.convert_to_numpy(example)
        if benchmark is None:
            continue
        result = utils.run_network(session, benchmark["input"])
        if np.array_equal(result, benchmark["output"]):
            passed += 1
        else:
            failed += 1
    return passed, failed


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    task = json.loads((DATA_DIR / f"{TASK_ID}.json").read_text(encoding="utf-8"))

    model = build_rot180_3x3_model()
    onnx.save(model, ONNX_PATH)

    sanitized = utils.sanitize_model(onnx.load(ONNX_PATH))
    if sanitized is None:
        raise RuntimeError("sanitize_model failed")

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = TRACE_PREFIX
    session = ort.InferenceSession(sanitized.SerializeToString(), options)

    arc_agi_pass, arc_agi_fail = verify_subset(
        utils, session, task["train"] + task["test"]
    )
    arc_gen_pass, arc_gen_fail = verify_subset(utils, session, task["arc-gen"])
    trace_path = session.end_profiling()
    memory, params = utils.score_network(sanitized, trace_path)
    if memory is None or params is None:
        raise RuntimeError("score_network failed")

    points = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
    result = {
        "task_id": TASK_ID,
        "transform": "rot180 over fixed 3x3 grids",
        "onnx_path": str(ONNX_PATH.relative_to(ROOT)),
        "submission_path": str(SUBMISSION_PATH.relative_to(ROOT)),
        "arc_agi_pass": arc_agi_pass,
        "arc_agi_fail": arc_agi_fail,
        "arc_gen_pass": arc_gen_pass,
        "arc_gen_fail": arc_gen_fail,
        "memory_bytes": memory,
        "params": params,
        "estimated_points": points,
        "filesize_bytes": ONNX_PATH.stat().st_size,
        "leakage_risk": "低: public train/test/arc-genに対する固定3x3 rot180の手書きONNX。private validationで同一変換かは未保証。",
        "overfitting_risk": "中: task087の全公開例に一致する固定変換であり、汎化検証ではない。",
    }
    (EXP_DIR / "task087_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with zipfile.ZipFile(SUBMISSION_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(ONNX_PATH, arcname=f"{TASK_ID}.onnx")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
