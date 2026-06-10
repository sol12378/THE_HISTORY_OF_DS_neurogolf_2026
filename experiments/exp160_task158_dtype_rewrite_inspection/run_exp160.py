from __future__ import annotations

import csv
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXP_DIR = ROOT / "experiments" / "exp160_task158_dtype_rewrite_inspection"
BASE_ZIP = ROOT / "experiments" / "exp152_task023_b035_repair_probe" / "submission.zip"
TASK_ID = 158
FULL_GRID_DIMS = {tuple([1, 10, 30, 30]), tuple([1, 1, 30, 30])}


def type_name(elem_type: int | None) -> str:
    if elem_type is None:
        return ""
    return TensorProto.DataType.Name(elem_type)


def dims_of(value_info: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return tuple(dims)


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE_ZIP) as zf:
        raw = zf.read(f"task{TASK_ID:03d}.onnx")
    model = onnx.shape_inference.infer_shapes(onnx.load_model_from_string(raw), strict_mode=False)
    (EXP_DIR / "task158.onnx").write_bytes(raw)

    info: dict[str, tuple[int | None, tuple[int | str, ...]]] = {}
    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        tensor_type = value_info.type.tensor_type
        if tensor_type.HasField("elem_type"):
            info[value_info.name] = (int(tensor_type.elem_type), dims_of(value_info))
    for init in model.graph.initializer:
        info[init.name] = (int(init.data_type), tuple(int(d) for d in init.dims))

    consumers: dict[str, list[onnx.NodeProto]] = {}
    producers: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
        for name in node.output:
            producers[name] = node

    rows = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "Cast":
            continue
        to_attr = next((attr.i for attr in node.attribute if attr.name == "to"), None)
        out = node.output[0]
        out_type, out_dims = info.get(out, (None, ()))
        in_type, in_dims = info.get(node.input[0], (None, ()))
        if to_attr != TensorProto.FLOAT or tuple(out_dims) not in FULL_GRID_DIMS:
            continue
        users = consumers.get(out, [])
        in_producer = producers.get(node.input[0])
        rows.append(
            {
                "node_index": idx,
                "input": node.input[0],
                "input_type": type_name(in_type),
                "input_dims": str(in_dims),
                "input_producer": in_producer.op_type if in_producer is not None else "input_or_initializer",
                "output": out,
                "output_type": type_name(out_type),
                "output_dims": str(out_dims),
                "user_count": len(users),
                "users": " ".join(user.op_type for user in users),
                "rewrite_hint": hint(in_type, users),
            }
        )

    out_csv = EXP_DIR / "task158_cast_to_float_fullgrid.csv"
    fieldnames = list(rows[0].keys()) if rows else ["node_index"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    hints = {}
    for row in rows:
        hints[row["rewrite_hint"]] = hints.get(row["rewrite_hint"], 0) + 1
    result: dict[str, Any] = {
        "exp_id": "exp160_task158_dtype_rewrite_inspection",
        "date": "2026-06-10",
        "status": "inspection_complete",
        "task_id": TASK_ID,
        "cast_to_float_fullgrid_count": len(rows),
        "rewrite_hint_counts": hints,
        "sample_rows": rows[:20],
        "outputs": {"cast_audit": str(out_csv.relative_to(ROOT)), "task_model": str((EXP_DIR / "task158.onnx").relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "decision": decide(hints),
        "leakage_risk": "low: graph structure inspection only.",
        "overfitting_risk": "low-to-medium: rewrite hints may not be valid transformations.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def hint(input_type: int | None, users: list[onnx.NodeProto]) -> str:
    user_ops = {user.op_type for user in users}
    if input_type in {TensorProto.BOOL, TensorProto.UINT8} and user_ops <= {"Where"}:
        return "possible_keep_bool_as_where_condition"
    if input_type in {TensorProto.BOOL, TensorProto.UINT8} and user_ops & {"MatMul", "Mul", "Add", "ReduceSum", "Sum"}:
        return "numeric_consumers_need_float"
    if input_type in {TensorProto.BOOL, TensorProto.UINT8}:
        return "boolish_to_float_mixed_consumers"
    return "non_boolish_to_float"


def decide(hints: dict[str, int]) -> str:
    if hints.get("possible_keep_bool_as_where_condition", 0):
        return "try local rewrite for casts used only as Where condition"
    if hints.get("numeric_consumers_need_float", 0):
        return "task158 casts feed numeric consumers; direct dtype removal is unlikely safe"
    return "no obvious safe dtype rewrite from Cast audit"


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp160_task158_dtype_rewrite_inspection",
        "",
        "## 目的",
        "",
        "exp159で候補になったtask158について、full-grid Cast-to-FLOATの消費先を確認し、dtype post-pass rewriteが安全そうか判断する。",
        "",
        "## 結果",
        "",
        f"- cast_to_float_fullgrid_count: `{result['cast_to_float_fullgrid_count']}`",
        f"- rewrite_hint_counts: `{result['rewrite_hint_counts']}`",
        f"- decision: {result['decision']}",
        "",
        "## リスク",
        "",
        str(result["leakage_risk"]),
        str(result["overfitting_risk"]),
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
