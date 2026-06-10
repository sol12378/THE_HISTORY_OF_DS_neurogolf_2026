from __future__ import annotations

import csv
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXP_DIR = ROOT / "experiments" / "exp159_dtype_postpass_candidate_audit"
BASE_EXP = ROOT / "experiments" / "exp152_task023_b035_repair_probe"
BASE_ZIP = BASE_EXP / "submission.zip"
FULL_GRID_DIMS = {tuple([1, 10, 30, 30]), tuple([1, 1, 30, 30])}
BOOLISH_OPS = {"Greater", "Less", "Equal", "And", "Or", "Not", "Xor"}


def type_name(elem_type: int | None) -> str:
    if elem_type is None:
        return ""
    return TensorProto.DataType.Name(elem_type)


def dims_of(value_info: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    dims = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return tuple(dims)


def collect_value_types(model: onnx.ModelProto) -> dict[str, tuple[int | None, tuple[int | str, ...]]]:
    info: dict[str, tuple[int | None, tuple[int | str, ...]]] = {}
    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("elem_type"):
            continue
        info[value_info.name] = (int(tensor_type.elem_type), dims_of(value_info))
    for init in model.graph.initializer:
        info[init.name] = (int(init.data_type), tuple(int(d) for d in init.dims))
    return info


def audit_model(task_id: int, raw: bytes) -> dict[str, Any]:
    try:
        model = onnx.load_model_from_string(raw)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        reason = "ok"
    except Exception as exc:
        try:
            model = onnx.load_model_from_string(raw)
            inferred = model
            reason = f"shape_inference_failed:{str(exc)[:120]}"
        except Exception as exc2:
            return {"task_id": task_id, "audit_status": "parse_failed", "reason": str(exc2)[:160]}

    value_types = collect_value_types(inferred)
    producer: dict[str, str] = {}
    for node in inferred.graph.node:
        for out in node.output:
            producer[out] = node.op_type

    op_counts = Counter(node.op_type for node in inferred.graph.node)
    full_grid_by_type = Counter()
    full_grid_by_producer = Counter()
    boolish_full_grid = 0
    float_full_grid_from_boolish = 0
    cast_to_float_full_grid = 0
    cast_from_boolish_to_float_full_grid = 0
    where_full_grid = 0

    for name, (elem_type, dims) in value_types.items():
        if tuple(dims) not in FULL_GRID_DIMS:
            continue
        dtype = type_name(elem_type)
        op = producer.get(name, "input_or_initializer")
        full_grid_by_type[dtype] += 1
        full_grid_by_producer[op] += 1
        if dtype == "BOOL" or op in BOOLISH_OPS:
            boolish_full_grid += 1
        if dtype == "FLOAT" and op in BOOLISH_OPS:
            float_full_grid_from_boolish += 1
        if dtype == "FLOAT" and op == "Cast":
            cast_to_float_full_grid += 1

    for node in inferred.graph.node:
        if node.op_type == "Where":
            for out in node.output:
                _, dims = value_types.get(out, (None, ()))
                if tuple(dims) in FULL_GRID_DIMS:
                    where_full_grid += 1
        if node.op_type == "Cast":
            to_attr = next((attr.i for attr in node.attribute if attr.name == "to"), None)
            out_dtype, out_dims = value_types.get(node.output[0], (None, ()))
            in_dtype, _ = value_types.get(node.input[0], (None, ()))
            if tuple(out_dims) in FULL_GRID_DIMS and to_attr == TensorProto.FLOAT and in_dtype in {TensorProto.BOOL, TensorProto.UINT8, TensorProto.INT32, TensorProto.INT64}:
                cast_from_boolish_to_float_full_grid += 1

    heuristic_saving = 0
    heuristic_saving += max(0, cast_from_boolish_to_float_full_grid) * 27000
    heuristic_saving += max(0, boolish_full_grid) * 27000
    heuristic_saving += max(0, where_full_grid) * 9000

    return {
        "task_id": task_id,
        "audit_status": "ok",
        "reason": reason,
        "node_count": len(inferred.graph.node),
        "op_counts": " ".join(f"{op}:{n}" for op, n in op_counts.most_common(12)),
        "full_grid_float": full_grid_by_type.get("FLOAT", 0),
        "full_grid_bool": full_grid_by_type.get("BOOL", 0),
        "full_grid_uint8": full_grid_by_type.get("UINT8", 0),
        "full_grid_int32": full_grid_by_type.get("INT32", 0),
        "boolish_full_grid": boolish_full_grid,
        "float_full_grid_from_boolish": float_full_grid_from_boolish,
        "cast_to_float_full_grid": cast_to_float_full_grid,
        "cast_from_boolish_to_float_full_grid": cast_from_boolish_to_float_full_grid,
        "where_full_grid": where_full_grid,
        "top_full_grid_producers": " ".join(f"{op}:{n}" for op, n in full_grid_by_producer.most_common(10)),
        "heuristic_saving": heuristic_saving,
    }


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(BASE_ZIP) as zf:
        for name in sorted(zf.namelist()):
            task_id = int(Path(name).stem.replace("task", ""))
            rows.append(audit_model(task_id, zf.read(name)))
    rows.sort(key=lambda row: (-int(row.get("heuristic_saving") or 0), -int(row.get("full_grid_float") or 0), int(row["task_id"])))

    out_csv = EXP_DIR / "dtype_postpass_candidate_audit.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result: dict[str, Any] = {
        "exp_id": "exp159_dtype_postpass_candidate_audit",
        "date": "2026-06-10",
        "status": "candidate_audit_complete",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "audited_tasks": len(rows),
        "top_candidates": rows[:20],
        "outputs": {"candidate_audit": str(out_csv.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "decision": "Inspect top candidates for local graph rewrites that keep boolean/mask intermediates as BOOL/UINT8 and avoid fp32 materialization.",
        "leakage_risk": "low: graph structure audit only.",
        "overfitting_risk": "low-to-medium: cost-structure heuristic may not translate to valid rewrites.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp159_dtype_postpass_candidate_audit",
        "",
        "## Hypothesis",
        "",
        "exp152 current best bundleには、bool/mask系full-grid中間をFLOATとしてmaterializeしているtaskがあり、dtype post-pass候補を構造監査で絞れる。",
        "",
        "## Result",
        "",
        f"- status: `{result['status']}`",
        f"- audited_tasks: `{result['audited_tasks']}`",
        f"- decision: {result['decision']}",
        "",
        "## Top Candidates",
        "",
        "| task | heuristic_saving | full_grid_float | boolish_full_grid | cast_boolish_to_float | where_full_grid | ops |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["top_candidates"][:12]:
        lines.append(
            f"| {int(row['task_id']):03d} | {row.get('heuristic_saving', '')} | {row.get('full_grid_float', '')} | {row.get('boolish_full_grid', '')} | {row.get('cast_from_boolish_to_float_full_grid', '')} | {row.get('where_full_grid', '')} | {row.get('op_counts', '')} |"
        )
    lines.extend(["", "## Leakage / Overfitting Risk", "", str(result["leakage_risk"]), str(result["overfitting_risk"])])
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
