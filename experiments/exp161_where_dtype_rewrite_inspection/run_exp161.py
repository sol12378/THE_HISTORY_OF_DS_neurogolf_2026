from __future__ import annotations

import csv
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXP_DIR = ROOT / "experiments" / "exp161_where_dtype_rewrite_inspection"
BASE_ZIP = ROOT / "experiments" / "exp152_task023_b035_repair_probe" / "submission.zip"
TASKS = [206, 338, 328, 366]
FULL_GRID_DIMS = {tuple([1, 10, 30, 30]), tuple([1, 1, 30, 30])}


def type_name(elem_type: int | None) -> str:
    return "" if elem_type is None else TensorProto.DataType.Name(elem_type)


def dims_of(value_info: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        dims.append(int(dim.dim_value) if dim.dim_value else dim.dim_param or "?")
    return tuple(dims)


def inspect_task(task_id: int, raw: bytes) -> list[dict[str, Any]]:
    model = onnx.shape_inference.infer_shapes(onnx.load_model_from_string(raw), strict_mode=False)
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
        if node.op_type == "Cast":
            to_attr = next((attr.i for attr in node.attribute if attr.name == "to"), None)
            out = node.output[0]
            out_type, out_dims = info.get(out, (None, ()))
            in_type, in_dims = info.get(node.input[0], (None, ()))
            if to_attr == TensorProto.FLOAT and tuple(out_dims) in FULL_GRID_DIMS:
                users = consumers.get(out, [])
                rows.append(
                    {
                        "task_id": task_id,
                        "kind": "cast_to_float_fullgrid",
                        "node_index": idx,
                        "input": node.input[0],
                        "input_type": type_name(in_type),
                        "input_dims": str(in_dims),
                        "input_producer": producers[node.input[0]].op_type if node.input[0] in producers else "input_or_initializer",
                        "output": out,
                        "output_type": type_name(out_type),
                        "output_dims": str(out_dims),
                        "users": " ".join(user.op_type for user in users),
                        "user_count": len(users),
                        "hint": hint(in_type, users),
                    }
                )
        if node.op_type == "Where":
            out = node.output[0]
            out_type, out_dims = info.get(out, (None, ()))
            if tuple(out_dims) in FULL_GRID_DIMS:
                cond_type, cond_dims = info.get(node.input[0], (None, ()))
                x_type, x_dims = info.get(node.input[1], (None, ()))
                y_type, y_dims = info.get(node.input[2], (None, ()))
                rows.append(
                    {
                        "task_id": task_id,
                        "kind": "where_fullgrid",
                        "node_index": idx,
                        "input": node.input[0],
                        "input_type": type_name(cond_type),
                        "input_dims": str(cond_dims),
                        "input_producer": producers[node.input[0]].op_type if node.input[0] in producers else "input_or_initializer",
                        "output": out,
                        "output_type": type_name(out_type),
                        "output_dims": str(out_dims),
                        "users": " ".join(user.op_type for user in consumers.get(out, [])),
                        "user_count": len(consumers.get(out, [])),
                        "hint": f"where_cond={type_name(cond_type)} x={type_name(x_type)}{x_dims} y={type_name(y_type)}{y_dims}",
                    }
                )
    return rows


def hint(input_type: int | None, users: list[onnx.NodeProto]) -> str:
    user_ops = {user.op_type for user in users}
    if input_type in {TensorProto.BOOL, TensorProto.UINT8} and user_ops <= {"Where"}:
        return "possible_keep_bool_as_where_condition"
    if input_type in {TensorProto.BOOL, TensorProto.UINT8} and user_ops & {"Sum", "ReduceSum", "Mul", "Add", "Sub", "MatMul"}:
        return "numeric_consumers_need_float"
    if input_type in {TensorProto.BOOL, TensorProto.UINT8}:
        return "boolish_to_float_mixed_consumers"
    return "non_boolish_to_float"


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(BASE_ZIP) as zf:
        for task_id in TASKS:
            raw = zf.read(f"task{task_id:03d}.onnx")
            (EXP_DIR / f"task{task_id:03d}.onnx").write_bytes(raw)
            rows.extend(inspect_task(task_id, raw))

    out_csv = EXP_DIR / "where_dtype_inspection.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_task = {}
    for task_id in TASKS:
        task_rows = [r for r in rows if r["task_id"] == task_id]
        by_task[f"{task_id:03d}"] = dict(Counter(r["hint"] for r in task_rows))
    result: dict[str, Any] = {
        "exp_id": "exp161_where_dtype_rewrite_inspection",
        "date": "2026-06-10",
        "status": "inspection_complete",
        "tasks": TASKS,
        "hint_counts_by_task": by_task,
        "promising_rows": [r for r in rows if r["hint"] == "possible_keep_bool_as_where_condition"][:30],
        "outputs": {"inspection": str(out_csv.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "decision": decide(rows),
        "leakage_risk": "low: graph structure inspection only.",
        "overfitting_risk": "low-to-medium: rewrite hints may not be valid transformations.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def decide(rows: list[dict[str, Any]]) -> str:
    promising = [r for r in rows if r["hint"] == "possible_keep_bool_as_where_condition"]
    if promising:
        tasks = sorted({int(r["task_id"]) for r in promising})
        return "inspect possible Where-condition cast rewrites for tasks " + " ".join(f"{t:03d}" for t in tasks)
    return "no obvious safe Cast-to-FLOAT removal; dtype lane should target generated/lowering graphs rather than existing artifact surgery"


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp161_where_dtype_rewrite_inspection",
        "",
        "## 目的",
        "",
        "Whereが多いdtype候補taskについて、boolish-to-FLOAT CastがWhere条件としてだけ使われる安全rewrite候補かを監査する。",
        "",
        "## 結果",
        "",
        f"- tasks: `{result['tasks']}`",
        f"- hint_counts_by_task: `{result['hint_counts_by_task']}`",
        f"- promising_rows: `{len(result['promising_rows'])}`",
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
