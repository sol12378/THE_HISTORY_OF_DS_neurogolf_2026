from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import time
import zipfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "exp005_top_cost_rewrite_strict"
INPUT_EXP = ROOT / "experiments" / "exp004_public_blend_relaxed_static"
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
BANNED_OPS = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
MAX_ONNX_BYTES = int(1.44 * 1024 * 1024)


@dataclass
class RewriteResult:
    task_id: int
    source: str
    baseline_cost: int
    new_cost: int
    baseline_points: float
    new_points: float
    removed_initializers: int
    baseline_file_bytes: int
    new_file_bytes: int
    status: str
    reason: str


def load_neurogolf_utils() -> Any:
    path = DATA_DIR / "neurogolf_utils" / "neurogolf_utils.py"
    spec = importlib.util.spec_from_file_location("neurogolf_utils_exp005", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["neurogolf_utils_exp005"] = module
    spec.loader.exec_module(module)
    return module


def task_examples(task_id: int, arc_gen_sample: int = 2) -> list[dict[str, Any]]:
    task = json.loads((DATA_DIR / f"task{task_id:03d}.json").read_text(encoding="utf-8"))
    return task["train"] + task["test"] + task["arc-gen"][:arc_gen_sample]


def validate_sample(utils: Any, raw: bytes, task_id: int) -> tuple[bool, str]:
    try:
        session = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    except Exception as exc:
        return False, f"ort load failed: {str(exc)[:160]}"
    for idx, example in enumerate(task_examples(task_id)):
        benchmark = utils.convert_to_numpy(example)
        if benchmark is None:
            continue
        try:
            output = utils.run_network(session, benchmark["input"])
        except Exception as exc:
            return False, f"runtime example {idx}: {str(exc)[:160]}"
        if not np.array_equal(output, benchmark["output"]):
            return False, f"mismatch example {idx}"
    return True, "ok"


def infer_static_ok(model: onnx.ModelProto) -> tuple[bool, str]:
    if model.functions:
        return False, "functions are not allowed"
    for node in model.graph.node:
        op = node.op_type.upper()
        if op in BANNED_OPS or "SEQUENCE" in op:
            return False, f"banned op {node.op_type}"
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return False, "subgraphs are not allowed"
    try:
        onnx.checker.check_model(model, full_check=True)
        graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    except Exception as exc:
        return False, f"shape/check failed: {str(exc)[:160]}"
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        if not value.type.HasField("tensor_type"):
            continue
        shape = value.type.tensor_type.shape
        if not shape:
            return False, f"missing shape {value.name}"
        for dim in shape.dim:
            if dim.HasField("dim_param"):
                return False, f"dynamic shape {value.name}"
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                return False, f"bad dim {value.name}"
    return True, "ok"


def remove_unused_initializers(model: onnx.ModelProto) -> int:
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [init for init in model.graph.initializer if init.name in used]
    removed = len(model.graph.initializer) - len(kept)
    if removed:
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
    return removed


def score_model(utils: Any, raw: bytes, task_id: int, source: str) -> tuple[int | None, int | None, str]:
    try:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.profile_file_prefix = str(EXP_DIR / f"profile_task{task_id:03d}_{source}_rewrite")
        session = ort.InferenceSession(raw, options, providers=["CPUExecutionProvider"])
        benchmark = utils.convert_to_numpy(task_examples(task_id, 1)[0])
        if benchmark is not None:
            utils.run_network(session, benchmark["input"])
        trace_path = session.end_profiling()
        memory, params = utils.score_network(onnx.load_model_from_string(raw), trace_path)
        pathlib.Path(trace_path).unlink(missing_ok=True)
    except Exception as exc:
        return None, None, f"score failed: {str(exc)[:160]}"
    if memory is None or params is None:
        return None, None, "score returned none"
    return int(memory), int(params), "ok"


def point(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def load_selected_rows() -> dict[int, dict[str, str]]:
    with (INPUT_EXP / "selected_manifest.csv").open(encoding="utf-8", newline="") as f:
        return {int(row["task_id"]): row for row in csv.DictReader(f)}


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    selected_rows = load_selected_rows()
    results: list[RewriteResult] = []
    final_raw: dict[int, bytes] = {}
    changed_tasks: list[int] = []

    with zipfile.ZipFile(INPUT_EXP / "submission.zip") as zf:
        for idx, name in enumerate(sorted(zf.namelist()), start=1):
            if idx % 25 == 0:
                print(f"Rewriting {idx}/400", flush=True)
            task_id = int(pathlib.Path(name).stem.replace("task", ""))
            row = selected_rows[task_id]
            baseline_raw = zf.read(name)
            baseline_cost = int(float(row["cost"]))
            model = onnx.load_model_from_string(baseline_raw)
            removed = remove_unused_initializers(model)
            if removed == 0:
                final_raw[task_id] = baseline_raw
                results.append(
                    RewriteResult(
                        task_id,
                        row["source_label"],
                        baseline_cost,
                        baseline_cost,
                        float(row["local_points"]),
                        float(row["local_points"]),
                        0,
                        len(baseline_raw),
                        len(baseline_raw),
                        "unchanged",
                        "no unused initializers",
                    )
                )
                continue
            ok, reason = infer_static_ok(model)
            if not ok:
                final_raw[task_id] = baseline_raw
                status = "rejected"
                results.append(
                    RewriteResult(task_id, row["source_label"], baseline_cost, baseline_cost, float(row["local_points"]), float(row["local_points"]), removed, len(baseline_raw), len(baseline_raw), status, reason)
                )
                continue
            sanitized = utils.sanitize_model(model)
            if sanitized is None:
                final_raw[task_id] = baseline_raw
                results.append(
                    RewriteResult(task_id, row["source_label"], baseline_cost, baseline_cost, float(row["local_points"]), float(row["local_points"]), removed, len(baseline_raw), len(baseline_raw), "rejected", "sanitize failed")
                )
                continue
            raw = sanitized.SerializeToString()
            if len(raw) > MAX_ONNX_BYTES:
                final_raw[task_id] = baseline_raw
                results.append(
                    RewriteResult(task_id, row["source_label"], baseline_cost, baseline_cost, float(row["local_points"]), float(row["local_points"]), removed, len(baseline_raw), len(raw), "rejected", "file too large")
                )
                continue
            ok, reason = validate_sample(utils, raw, task_id)
            if not ok:
                final_raw[task_id] = baseline_raw
                results.append(
                    RewriteResult(task_id, row["source_label"], baseline_cost, baseline_cost, float(row["local_points"]), float(row["local_points"]), removed, len(baseline_raw), len(raw), "rejected", reason)
                )
                continue
            memory, params, reason = score_model(utils, raw, task_id, row["source_label"])
            if memory is None or params is None:
                final_raw[task_id] = baseline_raw
                results.append(
                    RewriteResult(task_id, row["source_label"], baseline_cost, baseline_cost, float(row["local_points"]), float(row["local_points"]), removed, len(baseline_raw), len(raw), "rejected", reason)
                )
                continue
            new_cost = memory + params
            if new_cost < baseline_cost:
                final_raw[task_id] = raw
                changed_tasks.append(task_id)
                status = "improved"
            else:
                final_raw[task_id] = baseline_raw
                status = "no_cost_gain"
            results.append(
                RewriteResult(
                    task_id,
                    row["source_label"],
                    baseline_cost,
                    new_cost if status == "improved" else baseline_cost,
                    float(row["local_points"]),
                    point(new_cost) if status == "improved" else float(row["local_points"]),
                    removed,
                    len(baseline_raw),
                    len(raw),
                    status,
                    reason,
                )
            )

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task_id, raw in sorted(final_raw.items()):
            zf.writestr(f"task{task_id:03d}.onnx", raw)

    with (EXP_DIR / "rewrite_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RewriteResult.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([r.__dict__ for r in results])

    baseline_score = sum(r.baseline_points for r in results)
    new_score = sum(r.new_points for r in results)
    result = {
        "exp_id": "exp005_top_cost_rewrite_strict",
        "status": "rewrite_improved" if changed_tasks else "analysis_complete_no_rewrite_gain",
        "date": "2026-06-05",
        "input_exp": "exp004_public_blend_relaxed_static",
        "rewrite": "remove unused initializers from selected strict-valid ONNX models",
        "baseline_local_estimate": baseline_score,
        "new_local_estimate": new_score,
        "delta": new_score - baseline_score,
        "changed_task_count": len(changed_tasks),
        "changed_tasks": changed_tasks,
        "submission_zip": str(OUTPUT_ZIP.relative_to(ROOT)),
        "submission_zip_bytes": OUTPUT_ZIP.stat().st_size,
        "top_rewrite_candidates": [
            {"task_id": 366, "priority": 1, "reason": "突出したcost。mask cascade/template rewrite候補。"},
            {"task_id": 173, "priority": 2, "reason": "memory支配。broadcast/Tile/Where削減候補。"},
            {"task_id": 54, "priority": 3, "reason": "Compress reject候補をstrict互換に置換する余地。"},
            {"task_id": 77, "priority": 4, "reason": "memory支配かつCompress reject候補が多い。"},
            {"task_id": 286, "priority": 5, "reason": "Min/MaxPool chain短縮候補。"},
        ],
        "hash": hashlib.sha256(OUTPUT_ZIP.read_bytes()).hexdigest(),
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
