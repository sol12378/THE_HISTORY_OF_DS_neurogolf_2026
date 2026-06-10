from __future__ import annotations

import json
import pathlib
import sys
from datetime import date

import onnx
from onnx import TensorProto, helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import load_neurogolf_utils, score_model  # noqa: E402


EXP_ID = "exp266_dtype_memory_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 87


def make_cast_chain(dtype: int, label: str) -> bytes:
    nodes = [
        helper.make_node("Cast", ["input"], [f"{label}_mid"], to=dtype),
        helper.make_node("Identity", [f"{label}_mid"], [f"{label}_id"]),
        helper.make_node("Cast", [f"{label}_id"], ["output"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes,
        f"dtype_probe_{label}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, None, None, None])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, None, None, None])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def make_arith_chain(dtype: int, label: str) -> bytes:
    zero = helper.make_tensor(f"{label}_zero", dtype, [1], [0])
    nodes = [
        helper.make_node("Cast", ["input"], [f"{label}_mid"], to=dtype),
        helper.make_node("Add", [f"{label}_mid", f"{label}_zero"], [f"{label}_add"]),
        helper.make_node("Identity", [f"{label}_add"], [f"{label}_id"]),
        helper.make_node("Cast", [f"{label}_id"], ["output"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes,
        f"dtype_probe_{label}_arith",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, None, None, None])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, None, None, None])],
        [zero],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def make_bool_chain() -> bytes:
    nodes = [
        helper.make_node("Cast", ["input"], ["bool_mid"], to=TensorProto.BOOL),
        helper.make_node("Not", ["bool_mid"], ["bool_not"]),
        helper.make_node("Not", ["bool_not"], ["bool_id"]),
        helper.make_node("Cast", ["bool_id"], ["output"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes,
        "dtype_probe_bool",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, None, None, None])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, None, None, None])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()

    candidates = [
        ("cast_fp32", make_cast_chain(TensorProto.FLOAT, "fp32")),
        ("cast_fp16", make_cast_chain(TensorProto.FLOAT16, "fp16")),
        ("cast_int64", make_cast_chain(TensorProto.INT64, "int64")),
        ("cast_uint8", make_cast_chain(TensorProto.UINT8, "uint8")),
        ("bool_notnot", make_bool_chain()),
        ("arith_fp32", make_arith_chain(TensorProto.FLOAT, "fp32")),
        ("arith_fp16", make_arith_chain(TensorProto.FLOAT16, "fp16")),
        ("arith_int64", make_arith_chain(TensorProto.INT64, "int64")),
        ("arith_uint8", make_arith_chain(TensorProto.UINT8, "uint8")),
    ]

    rows = []
    for name, raw in candidates:
        path = EXP_DIR / f"{name}.onnx"
        path.write_bytes(raw)
        memory, params, reason = score_model(utils, raw, TASK_ID, name, EXP_DIR)
        rows.append(
            {
                "name": name,
                "bytes": len(raw),
                "memory": memory,
                "params": params,
                "cost": None if memory is None or params is None else memory + params,
                "reason": reason,
            }
        )

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "rows": rows,
        "interpretation": "dtype memory accounting probe; no submission.",
        "submission_decision": "no_submit",
        "leakage_risk": "none: synthetic score probe only.",
        "overfitting_risk": "none.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes_lines = ["# exp266_dtype_memory_probe", "", "## 目的", "", "同一構造の中間tensor dtypeだけを変え、official costがdtype比例で下がるか確認する。", "", "## 結果", ""]
    for row in rows:
        notes_lines.append(f"- {row['name']}: memory `{row['memory']}`, params `{row['params']}`, cost `{row['cost']}`, reason `{row['reason']}`")
    notes_lines.extend(["", "## 判断", "", "このprobe結果をもとに、dtype縮小post-passのgo/no-goを決める。"])
    (EXP_DIR / "notes.md").write_text("\n".join(notes_lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
