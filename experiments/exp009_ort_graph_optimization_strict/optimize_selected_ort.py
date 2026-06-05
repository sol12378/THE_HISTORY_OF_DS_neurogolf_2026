from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "exp009_ort_graph_optimization_strict"
INPUT_EXP = ROOT / "experiments" / "exp005_top_cost_rewrite_strict"
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
BANNED_OPS = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
MAX_ONNX_BYTES = int(1.44 * 1024 * 1024)


@dataclass
class Row:
    task_id: int
    baseline_cost: int
    new_cost: int
    baseline_points: float
    new_points: float
    baseline_bytes: int
    new_bytes: int
    optimizer_level: str
    status: str
    reason: str


def point(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def load_utils() -> Any:
    path = DATA_DIR / "neurogolf_utils" / "neurogolf_utils.py"
    spec = importlib.util.spec_from_file_location("neurogolf_utils_exp009", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["neurogolf_utils_exp009"] = module
    spec.loader.exec_module(module)
    return module


def task_examples(task_id: int, arc_gen_sample: int = 2) -> list[dict[str, Any]]:
    data = json.loads((DATA_DIR / f"task{task_id:03d}.json").read_text(encoding="utf-8"))
    return data["train"] + data["test"] + data["arc-gen"][:arc_gen_sample]


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
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return False, f"custom domain {opset.domain}"
    try:
        onnx.checker.check_model(model, full_check=True)
        graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    except Exception as exc:
        return False, f"shape/check failed: {str(exc)[:160]}"
    seen: set[str] = set()
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value.name in seen:
            return False, f"duplicate value_info {value.name}"
        seen.add(value.name)
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
            out = utils.run_network(session, benchmark["input"])
        except Exception as exc:
            return False, f"runtime example {idx}: {str(exc)[:160]}"
        if not np.array_equal(out, benchmark["output"]):
            return False, f"mismatch example {idx}"
    return True, "ok"


def score(utils: Any, raw: bytes, task_id: int, label: str) -> tuple[int | None, str]:
    try:
        model = onnx.load_model_from_string(raw)
        sanitized = utils.sanitize_model(model)
        if sanitized is None:
            return None, "sanitize failed"
        raw = sanitized.SerializeToString()
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.profile_file_prefix = str(EXP_DIR / f"profile_task{task_id:03d}_{label}")
        session = ort.InferenceSession(raw, options, providers=["CPUExecutionProvider"])
        benchmark = utils.convert_to_numpy(task_examples(task_id, 1)[0])
        if benchmark is not None:
            utils.run_network(session, benchmark["input"])
        trace = session.end_profiling()
        memory, params = utils.score_network(sanitized, trace)
        pathlib.Path(trace).unlink(missing_ok=True)
    except Exception as exc:
        return None, f"score failed: {str(exc)[:160]}"
    if memory is None or params is None:
        return None, "cost failed"
    return int(memory) + int(params), "ok"


def optimized_raw(raw: bytes, level: ort.GraphOptimizationLevel, task_id: int, label: str) -> tuple[bytes | None, str]:
    with tempfile.TemporaryDirectory(dir=EXP_DIR) as td:
        out = pathlib.Path(td) / f"task{task_id:03d}_{label}.onnx"
        options = ort.SessionOptions()
        options.optimized_model_filepath = str(out)
        options.graph_optimization_level = level
        try:
            ort.InferenceSession(raw, options, providers=["CPUExecutionProvider"])
        except Exception as exc:
            return None, f"opt session failed: {str(exc)[:160]}"
        if not out.exists():
            return None, "optimized model not written"
        return out.read_bytes(), "ok"


def load_baseline_costs() -> dict[int, tuple[int, float]]:
    with (INPUT_EXP / "rewrite_manifest.csv").open(encoding="utf-8", newline="") as f:
        return {int(row["task_id"]): (int(float(row["new_cost"])), float(row["new_points"])) for row in csv.DictReader(f)}


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_utils()
    costs = load_baseline_costs()
    final: dict[int, bytes] = {}
    rows: list[Row] = []
    levels = [
        ("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
        ("extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
    ]
    with zipfile.ZipFile(INPUT_EXP / "submission.zip") as zf:
        names = sorted(zf.namelist())
        for idx, name in enumerate(names, start=1):
            if idx % 25 == 0:
                print(f"Optimizing {idx}/{len(names)}", flush=True)
            task_id = int(pathlib.Path(name).stem.replace("task", ""))
            baseline = zf.read(name)
            base_cost, base_points = costs[task_id]
            best_raw = baseline
            best_cost = base_cost
            best_level = "none"
            best_reason = "no candidate improved"
            status = "unchanged"
            for level_name, level in levels:
                cand_raw, reason = optimized_raw(baseline, level, task_id, level_name)
                if cand_raw is None:
                    continue
                try:
                    model = onnx.load_model_from_string(cand_raw)
                except Exception as exc:
                    best_reason = f"parse failed {level_name}: {str(exc)[:80]}"
                    continue
                ok, reason = infer_static_ok(model)
                if not ok:
                    best_reason = f"{level_name}: {reason}"
                    continue
                sanitized = utils.sanitize_model(model)
                if sanitized is None:
                    best_reason = f"{level_name}: sanitize failed"
                    continue
                cand_raw = sanitized.SerializeToString()
                if len(cand_raw) > MAX_ONNX_BYTES:
                    best_reason = f"{level_name}: file too large"
                    continue
                ok, reason = validate_sample(utils, cand_raw, task_id)
                if not ok:
                    best_reason = f"{level_name}: {reason}"
                    continue
                cand_cost, reason = score(utils, cand_raw, task_id, level_name)
                if cand_cost is None:
                    best_reason = f"{level_name}: {reason}"
                    continue
                if cand_cost < best_cost:
                    best_raw = cand_raw
                    best_cost = cand_cost
                    best_level = level_name
                    best_reason = "ok"
                    status = "improved"
            final[task_id] = best_raw
            rows.append(
                Row(
                    task_id=task_id,
                    baseline_cost=base_cost,
                    new_cost=best_cost,
                    baseline_points=base_points,
                    new_points=point(best_cost),
                    baseline_bytes=len(baseline),
                    new_bytes=len(best_raw),
                    optimizer_level=best_level,
                    status=status,
                    reason=best_reason,
                )
            )
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task_id, raw in sorted(final.items()):
            zf.writestr(f"task{task_id:03d}.onnx", raw)
    with (EXP_DIR / "optimization_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([r.__dict__ for r in rows])
    result = {
        "exp_id": "exp009_ort_graph_optimization_strict",
        "date": "2026-06-05",
        "input_exp": "exp005_top_cost_rewrite_strict",
        "baseline_local_estimate": sum(r.baseline_points for r in rows),
        "new_local_estimate": sum(r.new_points for r in rows),
        "delta": sum(r.new_points for r in rows) - sum(r.baseline_points for r in rows),
        "changed_task_count": sum(1 for r in rows if r.status == "improved"),
        "changed_tasks": [r.task_id for r in rows if r.status == "improved"],
        "status": "optimization_improved" if any(r.status == "improved" for r in rows) else "no_optimization_gain",
        "submission_zip": str(OUTPUT_ZIP.relative_to(ROOT)),
        "submission_zip_bytes": OUTPUT_ZIP.stat().st_size,
        "hash": hashlib.sha256(OUTPUT_ZIP.read_bytes()).hexdigest(),
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
