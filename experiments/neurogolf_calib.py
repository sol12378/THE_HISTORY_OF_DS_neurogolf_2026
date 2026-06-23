"""Calibrated local scorer for NeuroGolf — P0-1 fix.

Problem (empirically verified, see diag_p0.py): the OFFICIAL scorer
`neurogolf_utils.calculate_memory` begins with

    onnx.checker.check_model(model, full_check=True)            # line 195
    onnx.shape_inference.infer_shapes(model, strict_mode=True)  # line 196

With the locally-installed onnx==1.22.0 these raise
`[ShapeInferenceError] Attribute pads must not contain negative values`
for 7 union artifacts that use **negative pads on Conv / ConvTranspose**
(tasks 45,127,135,146,149,240,384). The Kaggle scoring backend (a different
onnx build) tolerates these and scores the artifacts normally, which is why the
local rescore is 7023.58 while the real LB is 7117.01 (gap == the 7 sentinels).

Fix: reproduce Kaggle's lenient behaviour locally by running the *unchanged*
official memory/params logic, but with non-strict shape checks. We do NOT touch
the artifacts (that would change their cost and break calibration) and we do NOT
edit the official `neurogolf_utils.py` (kept pristine as the reference scorer).

Safety design: `score_model_calibrated` first runs the official scorer verbatim,
so the 393 already-scoreable tasks are byte-for-byte unchanged. Only when the
official scorer fails *specifically* with a negative-pads / shape-inference error
do we fall back to the lenient recompute. Genuinely invalid models (banned ops,
etc.) still fail.
"""
from __future__ import annotations

import contextlib
import gc
import json
import math
import pathlib
import zipfile
from typing import Any

import numpy as np
import onnx
import onnx.checker
import onnx.shape_inference
import onnxruntime as ort

import phase1_rewrite_utils as P

# error fingerprints that mean "over-strict local rejection, Kaggle would score it"
_LENIENT_TRIGGERS = ("must not contain negative", "ShapeInferenceError", "shape inference")

# onnxruntime profiler dtype-string -> numpy itemsize (bytes)
_TRACE_DTYPE_BYTES = {
    "float": 4, "double": 8, "float16": 2, "bfloat16": 2,
    "int8": 1, "uint8": 1, "bool": 1, "int16": 2, "uint16": 2,
    "int32": 4, "uint32": 4, "int64": 8, "uint64": 8,
}

# the 7 union artifacts that use negative pads on Conv/ConvTranspose. The local
# onnx==1.22.0 strict shape-inference rejects them; Kaggle's onnx tolerates them.
# We score these via ORT execution (trace) AND freeze them from improvement until
# their true Kaggle cost is calibrated by a submission (P1-2) — a locally-cheaper
# replacement could be WORSE on Kaggle if Kaggle scored the negative-pad original
# favourably.
NEGATIVE_PAD_TASKS = frozenset({45, 127, 135, 146, 149, 240, 384})


@contextlib.contextmanager
def lenient_onnx_shape_checks():
    """Temporarily make check_model non-full and infer_shapes non-strict, so the
    official calculate_memory does not reject negative-pads Conv/ConvTranspose.
    Always restored (so infer_static_ok keeps its strict gate intact)."""
    orig_check = onnx.checker.check_model
    orig_infer = onnx.shape_inference.infer_shapes

    def lenient_check(model, *args, **kwargs):
        kwargs.pop("full_check", None)
        try:
            return orig_check(model, *args, full_check=False, **kwargs)
        except Exception:
            # check_model is only a gate here; the real measurement is in
            # calculate_memory. Swallow structural-inference complaints.
            return None

    def lenient_infer(model, *args, **kwargs):
        kwargs.pop("strict_mode", None)
        return orig_infer(model, *args, strict_mode=False, **kwargs)

    onnx.checker.check_model = lenient_check
    onnx.shape_inference.infer_shapes = lenient_infer
    try:
        yield
    finally:
        onnx.checker.check_model = orig_check
        onnx.shape_inference.infer_shapes = orig_infer


def _memory_from_trace(model: onnx.ModelProto, trace_path: str) -> int:
    """Reproduce the official calculate_memory's intent — sum of intermediate
    activation tensor sizes — but source every node-output shape from the ORT
    execution trace (output_type_shape) instead of static shape inference, which
    fails locally on negative pads. Faithful because ORT runs the exact graph the
    scorer measures; graph input/output and initializers are excluded (initializers
    are counted as params by calculate_params, exactly as the official scorer does)."""
    node_outputs: dict[str, list[str]] = {}
    for node in model.graph.node:
        key = node.name if node.name else (node.output[0] if node.output else "")
        node_outputs[key] = list(node.output)
    trace = json.loads(pathlib.Path(trace_path).read_text())
    mem: dict[str, int] = {}
    for event in trace:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        args = event["args"]
        if "output_type_shape" not in args:
            continue
        name = event.get("name", "").replace("_kernel_time", "")
        if name not in node_outputs:
            continue
        for i, shape_dict in enumerate(args["output_type_shape"]):
            if i >= len(node_outputs[name]):
                continue
            out_name = node_outputs[name][i]
            if not out_name or out_name in ("input", "output"):
                continue
            total, dtype = 0, "float"
            for dtype_str, dims in shape_dict.items():
                dtype = dtype_str
                total += math.prod(dims) if dims else 1
            itemsize = _TRACE_DTYPE_BYTES.get(dtype, 4)
            mem[out_name] = max(mem.get(out_name, 0), total * itemsize)
    return sum(mem.values())


def _score_via_trace(
    utils: Any, raw: bytes, task_id: int, template_name: str, exp_dir: pathlib.Path
) -> tuple[int | None, int | None, str]:
    """Execution-based score: memory from the ORT profile trace + params from the
    official calculate_params (no shape inference). Used only for negative-pad
    artifacts the official scorer cannot measure."""
    trace_path = ""
    try:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.profile_file_prefix = str(exp_dir / f"calib_task{task_id:03d}_{template_name}")
        session = ort.InferenceSession(raw, options, providers=["CPUExecutionProvider"])
        benchmark = utils.convert_to_numpy(P.examples_for(P.load_task(task_id), 1)[0])
        if benchmark is not None:
            utils.run_network(session, benchmark["input"])
        trace_path = session.end_profiling()
        model = onnx.load_model_from_string(raw)
        memory = _memory_from_trace(model, trace_path)
        params = utils.calculate_params(model)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"trace score failed: {str(exc)[:180]}"
    finally:
        if trace_path:
            pathlib.Path(trace_path).unlink(missing_ok=True)
    if params is None:
        return None, None, "trace score: params none"
    return int(memory), int(params), "ok(trace)"


def score_model_calibrated(
    utils: Any, raw: bytes, task_id: int, template_name: str, exp_dir: pathlib.Path
) -> tuple[int | None, int | None, str]:
    """Drop-in replacement for P.score_model that also measures the negative-pads
    artifacts Kaggle scores but local onnx strict shape-inference rejects.

    Design: run the OFFICIAL scorer first, so all 393 already-scoreable tasks are
    byte-for-byte unchanged (and stay consistent with evaluate_candidate, which uses
    the official score_model). Only on a negative-pads / shape-inference rejection do
    we fall back to execution-based (trace) scoring."""
    memory, params, reason = P.score_model(utils, raw, task_id, template_name, exp_dir)
    if memory is not None:
        return memory, params, reason
    if not any(trig in reason for trig in _LENIENT_TRIGGERS):
        return memory, params, reason  # genuine failure (banned op, broken graph, ...)
    return _score_via_trace(utils, raw, task_id, template_name, exp_dir)


def _read_zip(zip_path: pathlib.Path) -> dict[int, bytes]:
    out: dict[int, bytes] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            out[int(pathlib.Path(name).stem.replace("task", ""))] = zf.read(name)
    return out


def rebuild_baseline_calibrated(
    utils: Any,
    union_zip: pathlib.Path,
    out_ledger: pathlib.Path,
    log=print,
    resume: bool = True,
) -> dict[str, Any]:
    """Rescore all 400 union artifacts with the calibrated scorer and write
    baseline_ledger.json. Crash-resilient: writes progress to <ledger>.partial after
    every task, so an ORT segfault can be resumed by re-running (resume=True)."""
    raws = _read_zip(union_zip)
    partial = out_ledger.with_suffix(".partial.json")
    points: dict[int, float] = {}
    costs: dict[int, int] = {}
    if resume and partial.exists():
        data = json.loads(partial.read_text(encoding="utf-8"))
        points = {int(k): float(v) for k, v in data["points"].items()}
        costs = {int(k): int(v) for k, v in data["costs"].items()}
        log(f"resuming from partial: {len(points)}/400 already scored")
    for i, tid in enumerate(sorted(raws), 1):
        if tid in costs:
            continue
        m, p, reason = score_model_calibrated(utils, raws[tid], tid, "baseline", out_ledger.parent)
        if m is None:
            log(f"  WARN task{tid:03d} calibrated score failed: {reason}; sentinel cost=1e9")
            points[tid], costs[tid] = 1.0, 10**9
        else:
            costs[tid] = m + p
            points[tid] = P.point(costs[tid])
            if "lenient" in reason:
                log(f"  task{tid:03d} recovered via lenient: cost={m + p} points={points[tid]:.3f}")
        partial.write_text(json.dumps({"points": points, "costs": costs}), encoding="utf-8")
        gc.collect()  # release ORT session/profiler resources between tasks
        if i % 50 == 0:
            log(f"  baseline {i}/400 ... running total {sum(points.values()):.2f}")
    total = sum(points.values())
    sentinels = sorted(t for t, c in costs.items() if c >= 10**9)
    out_ledger.write_text(
        json.dumps({"points": points, "costs": costs, "total": total}), encoding="utf-8"
    )
    partial.unlink(missing_ok=True)
    log(f"calibrated baseline total = {total:.2f}  sentinels={len(sentinels)} {sentinels}")
    return {"total": total, "sentinels": sentinels, "points": points, "costs": costs}
