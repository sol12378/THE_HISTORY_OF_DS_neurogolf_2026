from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import load_neurogolf_utils, score_model, sha256


EXP_DIR = ROOT / "experiments" / "exp158_dtype_memory_cost_probe"
TASK_ID = 87
INPUT_SHAPE = [1, 10, 30, 30]


def make_model(case: str) -> bytes:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, INPUT_SHAPE)
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, INPUT_SHAPE)
    nodes = []
    initializers = []

    if case == "identity":
        nodes.append(helper.make_node("Identity", ["input"], ["output"]))
    elif case == "fp32_add_zero":
        zero = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero")
        initializers.append(zero)
        nodes.append(helper.make_node("Add", ["input", "zero"], ["output"]))
    elif case == "fp32_add_zero_then_identity":
        zero = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero")
        initializers.append(zero)
        nodes.append(helper.make_node("Add", ["input", "zero"], ["mid_fp32"]))
        nodes.append(helper.make_node("Identity", ["mid_fp32"], ["output"]))
    elif case == "fp32_mul_one_then_add_zero":
        one = numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), "one")
        zero = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero")
        initializers.extend([one, zero])
        nodes.append(helper.make_node("Mul", ["input", "one"], ["mid_fp32"]))
        nodes.append(helper.make_node("Add", ["mid_fp32", "zero"], ["output"]))
    elif case == "fp16_roundtrip":
        nodes.append(helper.make_node("Cast", ["input"], ["mid_fp16"], to=TensorProto.FLOAT16))
        nodes.append(helper.make_node("Cast", ["mid_fp16"], ["output"], to=TensorProto.FLOAT))
    elif case == "bool_roundtrip":
        nodes.append(helper.make_node("Cast", ["input"], ["mid_bool"], to=TensorProto.BOOL))
        nodes.append(helper.make_node("Cast", ["mid_bool"], ["output"], to=TensorProto.FLOAT))
    elif case == "uint8_roundtrip":
        nodes.append(helper.make_node("Cast", ["input"], ["mid_u8"], to=TensorProto.UINT8))
        nodes.append(helper.make_node("Cast", ["mid_u8"], ["output"], to=TensorProto.FLOAT))
    elif case == "int32_roundtrip":
        nodes.append(helper.make_node("Cast", ["input"], ["mid_i32"], to=TensorProto.INT32))
        nodes.append(helper.make_node("Cast", ["mid_i32"], ["output"], to=TensorProto.FLOAT))
    elif case == "fp16_add_zero_roundtrip":
        zero16 = numpy_helper.from_array(np.asarray([0.0], dtype=np.float16), "zero16")
        initializers.append(zero16)
        nodes.append(helper.make_node("Cast", ["input"], ["mid_fp16"], to=TensorProto.FLOAT16))
        nodes.append(helper.make_node("Add", ["mid_fp16", "zero16"], ["mid2_fp16"]))
        nodes.append(helper.make_node("Cast", ["mid2_fp16"], ["output"], to=TensorProto.FLOAT))
    elif case == "bool_where_roundtrip":
        zero = numpy_helper.from_array(np.zeros(INPUT_SHAPE, dtype=np.float32), "zero_full")
        initializers.append(zero)
        nodes.append(helper.make_node("Cast", ["input"], ["cond_bool"], to=TensorProto.BOOL))
        nodes.append(helper.make_node("Where", ["cond_bool", "input", "zero_full"], ["output"]))
    else:
        raise ValueError(case)

    graph = helper.make_graph(nodes, f"exp158_{case}", [input_info], [output_info], initializers)
    model = helper.make_model(graph, producer_name="exp158_dtype_memory_cost_probe", ir_version=10, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    cases = [
        "identity",
        "fp32_add_zero",
        "fp32_add_zero_then_identity",
        "fp32_mul_one_then_add_zero",
        "fp16_roundtrip",
        "bool_roundtrip",
        "uint8_roundtrip",
        "int32_roundtrip",
        "fp16_add_zero_roundtrip",
        "bool_where_roundtrip",
    ]
    rows = []
    for case in cases:
        raw = make_model(case)
        path = EXP_DIR / f"{case}.onnx"
        path.write_bytes(raw)
        memory, params, reason = score_model(utils, raw, TASK_ID, case, EXP_DIR)
        cost = int(memory + params) if memory is not None and params is not None else None
        rows.append(
            {
                "case": case,
                "memory": memory if memory is not None else "",
                "params": params if params is not None else "",
                "cost": cost if cost is not None else "",
                "score_reason": reason,
                "bytes": len(raw),
                "sha256": sha256(raw),
                "path": str(path.relative_to(ROOT)),
            }
        )

    out_csv = EXP_DIR / "dtype_cost_probe.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result: dict[str, Any] = {
        "exp_id": "exp158_dtype_memory_cost_probe",
        "date": "2026-06-10",
        "status": "cost_probe_complete",
        "purpose": "Measure whether fp16/bool/uint8/int32 full-grid intermediates reduce official memory cost compared with fp32.",
        "task_id_for_trace": TASK_ID,
        "rows": rows,
        "outputs": {"dtype_cost_probe": str(out_csv.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "decision": decide(rows),
        "leakage_risk": "low: synthetic cost-only probe; no task labels used.",
        "overfitting_risk": "low: measures score_network cost physics, not public LB behavior.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def decide(rows: list[dict[str, Any]]) -> str:
    costs = {row["case"]: row["cost"] for row in rows if row["cost"] != ""}
    fp32 = max(costs.get("fp32_add_zero", 0), costs.get("fp32_add_zero_then_identity", 0), costs.get("fp32_mul_one_then_add_zero", 0))
    fp16 = costs.get("fp16_roundtrip")
    bool_cost = costs.get("bool_roundtrip")
    u8 = costs.get("uint8_roundtrip")
    if fp32 is None:
        return "score_failed_for_fp32_control"
    gains = []
    for name, cost in [("fp16", fp16), ("bool", bool_cost), ("uint8", u8)]:
        if cost is not None and cost < fp32:
            gains.append(f"{name}:{fp32}->{cost}")
    if gains:
        return "dtype_memory_reduction_observed; prioritize targeted post-pass probes: " + ", ".join(gains)
    return "no_dtype_memory_reduction_observed_for_simple_roundtrip; do not prioritize broad dtype post-pass yet"


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp158_dtype_memory_cost_probe",
        "",
        "## Hypothesis",
        "",
        "full-grid中間をfp16/bool/uint8へ縮小できれば、公式memory costがdtype幅に比例して下がる可能性がある。",
        "",
        "## Result",
        "",
        f"- status: `{result['status']}`",
        f"- decision: {result['decision']}",
        "",
        "| case | memory | params | cost | reason |",
        "|---|---:|---:|---:|---|",
    ]
    for row in result["rows"]:
        lines.append(f"| {row['case']} | {row['memory']} | {row['params']} | {row['cost']} | {row['score_reason']} |")
    lines.extend(["", "## Leakage / Overfitting Risk", "", str(result["leakage_risk"]), str(result["overfitting_risk"])])
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
