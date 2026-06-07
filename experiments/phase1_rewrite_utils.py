from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import zipfile
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
BASE_EXP = ROOT / "experiments" / "exp012_template_factory_core"
STRICT_EXP = ROOT / "experiments" / "exp005_top_cost_rewrite_strict"
ROUTE_EXP = ROOT / "experiments" / "exp011_gpu_route_classifier"
BANNED_OPS = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
MAX_ONNX_BYTES = int(1.44 * 1024 * 1024)


@dataclass(frozen=True)
class BaseTask:
    task_id: int
    cost: int
    points: float
    source: str
    template_name: str
    route: str
    raw: bytes


@dataclass(frozen=True)
class Candidate:
    task_id: int
    template_name: str
    route: str
    raw: bytes | None
    status: str
    reason: str


@dataclass
class CandidateEval:
    task_id: int
    route: str
    template_name: str
    baseline_cost: int
    candidate_cost: int | str
    baseline_points: float
    candidate_points: float | str
    file_bytes: int | str
    validation_status: str
    status: str
    reason: str
    sha256: str


def load_neurogolf_utils() -> Any:
    path = DATA_DIR / "neurogolf_utils" / "neurogolf_utils.py"
    spec = importlib.util.spec_from_file_location("neurogolf_utils_phase1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["neurogolf_utils_phase1"] = module
    spec.loader.exec_module(module)
    return module


def load_task(task_id: int) -> dict[str, Any]:
    return json.loads((DATA_DIR / f"task{task_id:03d}.json").read_text(encoding="utf-8"))


def examples_for(task: dict[str, Any], arc_gen_sample: int) -> list[dict[str, Any]]:
    arc = task["arc-gen"] if arc_gen_sample < 0 else task["arc-gen"][:arc_gen_sample]
    return task["train"] + task["test"] + arc


def grid_to_array(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def one_hot_padded(grid: list[list[int]], size: int = 30) -> np.ndarray:
    arr = np.zeros((1, 10, size, size), dtype=np.float32)
    for r, row in enumerate(grid[:size]):
        for c, color in enumerate(row[:size]):
            if 0 <= int(color) < 10:
                arr[0, int(color), r, c] = 1.0
    return arr


def one_hot_flat(grid: list[list[int]]) -> np.ndarray:
    return one_hot_padded(grid).reshape(-1)


def point(cost: int | float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def make_model(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    producer: str,
    opset_version: int = 10,
) -> bytes:
    graph = helper.make_graph(
        nodes,
        f"{producer}_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name=producer,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", opset_version)],
    )
    return model.SerializeToString()


def load_routes() -> dict[int, str]:
    routes: dict[int, str] = {}
    with (ROUTE_EXP / "route_predictions.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            routes[int(row["task_id"])] = row["pred_route"]
    return routes


def load_base_tasks(base_exp: pathlib.Path = BASE_EXP) -> dict[int, BaseTask]:
    routes = load_routes()
    rows: dict[int, dict[str, str]] = {}
    with (base_exp / "selected_manifest.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["task_id"])] = row
    with zipfile.ZipFile(base_exp / "submission.zip") as zf:
        raws = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}
    out: dict[int, BaseTask] = {}
    for task_id, row in rows.items():
        out[task_id] = BaseTask(
            task_id=task_id,
            cost=int(float(row["cost"])),
            points=float(row["local_points"]),
            source=row["source"],
            template_name=row["template_name"],
            route=routes.get(task_id, ""),
            raw=raws[task_id],
        )
    return out


def top_task_ids(base: dict[int, BaseTask], top_k: int) -> list[int]:
    return [x.task_id for x in sorted(base.values(), key=lambda item: (-item.cost, item.task_id))[:top_k]]


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
        return False, f"shape/check failed: {str(exc)[:180]}"
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


def validate_examples(utils: Any, raw: bytes, task_id: int, arc_gen_sample: int) -> tuple[bool, str, int, int]:
    task = load_task(task_id)
    try:
        session = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    except Exception as exc:
        return False, f"ort load failed: {str(exc)[:180]}", 0, 0
    passed = 0
    failed = 0
    for idx, example in enumerate(examples_for(task, arc_gen_sample)):
        benchmark = utils.convert_to_numpy(example)
        if benchmark is None:
            continue
        try:
            output = utils.run_network(session, benchmark["input"])
        except Exception as exc:
            return False, f"runtime example {idx}: {str(exc)[:180]}", passed, failed + 1
        if np.array_equal(output, benchmark["output"]):
            passed += 1
        else:
            failed += 1
            return False, f"mismatch example {idx}", passed, failed
    return True, "ok", passed, failed


def score_model(utils: Any, raw: bytes, task_id: int, template_name: str, exp_dir: pathlib.Path) -> tuple[int | None, int | None, str]:
    trace_path = ""
    try:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.profile_file_prefix = str(exp_dir / f"profile_task{task_id:03d}_{template_name}")
        session = ort.InferenceSession(raw, options, providers=["CPUExecutionProvider"])
        benchmark = utils.convert_to_numpy(examples_for(load_task(task_id), 1)[0])
        if benchmark is not None:
            utils.run_network(session, benchmark["input"])
        trace_path = session.end_profiling()
        memory, params = utils.score_network(onnx.load_model_from_string(raw), trace_path)
    except Exception as exc:
        return None, None, f"score failed: {str(exc)[:180]}"
    finally:
        if trace_path:
            pathlib.Path(trace_path).unlink(missing_ok=True)
    if memory is None or params is None:
        return None, None, "score returned none"
    return int(memory), int(params), "ok"


def evaluate_candidate(
    utils: Any,
    candidate: Candidate,
    base: BaseTask,
    arc_gen_sample: int,
    exp_dir: pathlib.Path,
) -> tuple[CandidateEval, bytes | None]:
    if candidate.raw is None:
        return (
            CandidateEval(base.task_id, candidate.route, candidate.template_name, base.cost, "", base.points, "", "", "not_run", candidate.status, candidate.reason, ""),
            None,
        )
    raw = candidate.raw
    if len(raw) > MAX_ONNX_BYTES:
        return (
            CandidateEval(base.task_id, candidate.route, candidate.template_name, base.cost, "", base.points, "", len(raw), "not_run", "rejected", "file too large", sha256(raw)),
            None,
        )
    try:
        model = onnx.load_model_from_string(raw)
    except Exception as exc:
        return (
            CandidateEval(base.task_id, candidate.route, candidate.template_name, base.cost, "", base.points, "", len(raw), "not_run", "rejected", f"parse failed: {str(exc)[:180]}", sha256(raw)),
            None,
        )
    ok, reason = infer_static_ok(model)
    if not ok:
        return (
            CandidateEval(base.task_id, candidate.route, candidate.template_name, base.cost, "", base.points, "", len(raw), "not_run", "rejected", reason, sha256(raw)),
            None,
        )
    sanitized = utils.sanitize_model(model)
    if sanitized is None:
        return (
            CandidateEval(base.task_id, candidate.route, candidate.template_name, base.cost, "", base.points, "", len(raw), "not_run", "rejected", "sanitize failed", sha256(raw)),
            None,
        )
    sanitized_raw = sanitized.SerializeToString()
    ok, reason, passed, failed = validate_examples(utils, sanitized_raw, base.task_id, arc_gen_sample)
    validation_status = f"{passed}_pass_{failed}_fail"
    if not ok:
        return (
            CandidateEval(base.task_id, candidate.route, candidate.template_name, base.cost, "", base.points, "", len(sanitized_raw), validation_status, "rejected", reason, sha256(sanitized_raw)),
            None,
        )
    memory, params, reason = score_model(utils, sanitized_raw, base.task_id, candidate.template_name, exp_dir)
    if memory is None or params is None:
        return (
            CandidateEval(base.task_id, candidate.route, candidate.template_name, base.cost, "", base.points, "", len(sanitized_raw), validation_status, "rejected", reason, sha256(sanitized_raw)),
            None,
        )
    cost = memory + params
    status = "improved" if cost < base.cost else "no_cost_gain"
    return (
        CandidateEval(
            base.task_id,
            candidate.route,
            candidate.template_name,
            base.cost,
            cost,
            base.points,
            point(cost),
            len(sanitized_raw),
            validation_status,
            status,
            reason if status == "improved" else "candidate cost is not lower than baseline",
            sha256(sanitized_raw),
        ),
        sanitized_raw if status == "improved" else None,
    )


def candidate_fields() -> list[str]:
    return list(CandidateEval.__dataclass_fields__.keys())


def write_zip(path: pathlib.Path, raws: dict[int, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task_id, raw in sorted(raws.items()):
            zf.writestr(f"task{task_id:03d}.onnx", raw)


def zip_sanity(path: pathlib.Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        names = sorted(zf.namelist())
    return {
        "count": len(names),
        "names_ok": len(names) == 400 and names[0] == "task001.onnx" and names[-1] == "task400.onnx",
        "first": names[:3],
        "last": names[-3:],
        "bytes": path.stat().st_size,
        "sha256": sha256(path.read_bytes()),
    }


def build_conv_color_map_candidate(task_id: int, examples: list[dict[str, Any]], route: str) -> Candidate:
    mapping: dict[int, int] = {}
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if x.shape != y.shape:
            return Candidate(task_id, "conv_color_map", route, None, "skipped", "shape mismatch")
        for src, dst in zip(x.ravel(), y.ravel()):
            src_i, dst_i = int(src), int(dst)
            if src_i in mapping and mapping[src_i] != dst_i:
                return Candidate(task_id, "conv_color_map", route, None, "skipped", "conflicting color map")
            mapping[src_i] = dst_i
    weight = np.zeros((10, 10, 1, 1), dtype=np.float32)
    for src in range(10):
        dst = mapping.get(src, src)
        weight[dst, src, 0, 0] = 1.0
    raw = make_model([helper.make_node("Conv", ["input", "weight"], ["output"])], [numpy_helper.from_array(weight, "weight")], "phase1_conv_color_map")
    return Candidate(task_id, "conv_color_map", route, raw, "generated", f"mapping={mapping}")


def transform_grid(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return arr.copy()
    if name == "flip_h":
        return np.flip(arr, axis=1).copy()
    if name == "flip_v":
        return np.flip(arr, axis=0).copy()
    if name == "rot180":
        return np.flip(np.flip(arr, axis=0), axis=1).copy()
    if name == "transpose":
        return arr.T.copy()
    if name == "rot90_cw":
        return np.rot90(arr, k=3).copy()
    if name == "rot90_ccw":
        return np.rot90(arr, k=1).copy()
    if name == "anti_transpose":
        return np.flip(np.flip(arr.T, axis=0), axis=1).copy()
    raise ValueError(f"unknown transform {name}")


def build_global_transform_color_map_candidate(task_id: int, examples: list[dict[str, Any]], route: str) -> Candidate:
    if not examples:
        return Candidate(task_id, "global_transform_color_map", route, None, "skipped", "no examples")
    input_shapes = {grid_to_array(ex["input"]).shape for ex in examples}
    output_shapes = {grid_to_array(ex["output"]).shape for ex in examples}
    if len(input_shapes) != 1 or len(output_shapes) != 1:
        return Candidate(task_id, "global_transform_color_map", route, None, "skipped", "variable shapes")
    ih, iw = next(iter(input_shapes))
    oh, ow = next(iter(output_shapes))

    transforms = ["flip_h", "flip_v", "rot180", "transpose", "rot90_cw", "rot90_ccw", "anti_transpose"]
    for transform in transforms:
        mapped: dict[int, int] = {}
        ok = True
        for ex in examples:
            x = transform_grid(grid_to_array(ex["input"]), transform)
            y = grid_to_array(ex["output"])
            if x.shape != y.shape:
                ok = False
                break
            for src, dst in zip(x.ravel(), y.ravel()):
                src_i, dst_i = int(src), int(dst)
                if src_i in mapped and mapped[src_i] != dst_i:
                    ok = False
                    break
                mapped[src_i] = dst_i
            if not ok:
                break
        if not ok:
            continue

        weight = np.zeros((10, 10, 1, 1), dtype=np.float32)
        for src in range(10):
            dst = mapped.get(src, src)
            weight[dst, src, 0, 0] = 1.0

        initializers = [
            numpy_helper.from_array(np.asarray([0, 0, 0, 0], dtype=np.int64), "starts"),
            numpy_helper.from_array(np.asarray([1, 10, ih, iw], dtype=np.int64), "ends"),
            numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "axes"),
            numpy_helper.from_array(weight, "weight"),
        ]
        nodes = [helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["x0"])]
        current = "x0"
        if transform == "flip_h":
            initializers.append(numpy_helper.from_array(np.arange(iw - 1, -1, -1, dtype=np.int64), "idx_w"))
            nodes.append(helper.make_node("Gather", [current, "idx_w"], ["x1"], axis=3))
            current = "x1"
        elif transform == "flip_v":
            initializers.append(numpy_helper.from_array(np.arange(ih - 1, -1, -1, dtype=np.int64), "idx_h"))
            nodes.append(helper.make_node("Gather", [current, "idx_h"], ["x1"], axis=2))
            current = "x1"
        elif transform == "rot180":
            initializers.extend(
                [
                    numpy_helper.from_array(np.arange(ih - 1, -1, -1, dtype=np.int64), "idx_h"),
                    numpy_helper.from_array(np.arange(iw - 1, -1, -1, dtype=np.int64), "idx_w"),
                ]
            )
            nodes.append(helper.make_node("Gather", [current, "idx_h"], ["x1"], axis=2))
            nodes.append(helper.make_node("Gather", ["x1", "idx_w"], ["x2"], axis=3))
            current = "x2"
        elif transform == "transpose":
            nodes.append(helper.make_node("Transpose", [current], ["x1"], perm=[0, 1, 3, 2]))
            current = "x1"
        elif transform == "rot90_cw":
            initializers.append(numpy_helper.from_array(np.arange(ih - 1, -1, -1, dtype=np.int64), "idx_h"))
            nodes.append(helper.make_node("Transpose", [current], ["x1"], perm=[0, 1, 3, 2]))
            nodes.append(helper.make_node("Gather", ["x1", "idx_h"], ["x2"], axis=3))
            current = "x2"
        elif transform == "rot90_ccw":
            initializers.append(numpy_helper.from_array(np.arange(iw - 1, -1, -1, dtype=np.int64), "idx_w"))
            nodes.append(helper.make_node("Transpose", [current], ["x1"], perm=[0, 1, 3, 2]))
            nodes.append(helper.make_node("Gather", ["x1", "idx_w"], ["x2"], axis=2))
            current = "x2"
        elif transform == "anti_transpose":
            initializers.extend(
                [
                    numpy_helper.from_array(np.arange(iw - 1, -1, -1, dtype=np.int64), "idx_w"),
                    numpy_helper.from_array(np.arange(ih - 1, -1, -1, dtype=np.int64), "idx_h"),
                ]
            )
            nodes.append(helper.make_node("Transpose", [current], ["x1"], perm=[0, 1, 3, 2]))
            nodes.append(helper.make_node("Gather", ["x1", "idx_w"], ["x2"], axis=2))
            nodes.append(helper.make_node("Gather", ["x2", "idx_h"], ["x3"], axis=3))
            current = "x3"

        nodes.append(helper.make_node("Conv", [current, "weight"], ["colored"]))
        nodes.append(helper.make_node("Pad", ["colored"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30 - oh, 30 - ow], value=0.0))
        raw = make_model(nodes, initializers, "phase1_global_transform_color_map")
        return Candidate(task_id, f"global_{transform}_color_map", route, raw, "generated", f"transform={transform},mapping={mapped}")
    return Candidate(task_id, "global_transform_color_map", route, None, "skipped", "no global transform fits")


def build_identity_candidate(task_id: int, examples: list[dict[str, Any]], route: str) -> Candidate:
    for ex in examples:
        if not np.array_equal(grid_to_array(ex["input"]), grid_to_array(ex["output"])):
            return Candidate(task_id, "direct_identity", route, None, "skipped", "not identity")
    raw = make_model([helper.make_node("Identity", ["input"], ["output"])], [], "phase1_direct_identity")
    return Candidate(task_id, "direct_identity", route, raw, "generated", "all fit examples are identity")


def build_constant_sparse_output_candidate(task_id: int, examples: list[dict[str, Any]], route: str) -> Candidate:
    if not examples:
        return Candidate(task_id, "constant_sparse_output", route, None, "skipped", "no examples")
    outputs = [one_hot_padded(ex["output"]) for ex in examples]
    first = outputs[0]
    if any(not np.array_equal(first, out) for out in outputs[1:]):
        return Candidate(task_id, "constant_sparse_output", route, None, "skipped", "non-constant output")

    nz = np.argwhere(first > 0.5).astype(np.int64)
    if len(nz) == 0:
        return Candidate(task_id, "constant_sparse_output", route, None, "skipped", "empty output")
    updates = np.ones((len(nz),), dtype=np.float32)
    raw = make_model(
        [
            helper.make_node("ConstantOfShape", ["shape"], ["zeros"], value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])),
            helper.make_node("ScatterND", ["zeros", "indices", "updates"], ["output"]),
        ],
        [
            numpy_helper.from_array(np.asarray([1, 10, 30, 30], dtype=np.int64), "shape"),
            numpy_helper.from_array(nz, "indices"),
            numpy_helper.from_array(updates, "updates"),
        ],
        "phase1_constant_sparse_output",
        opset_version=11,
    )
    return Candidate(task_id, "constant_sparse_output", route, raw, "generated", f"active_cells={len(nz)}")


def build_fixed_mask_where_candidate(task_id: int, examples: list[dict[str, Any]], route: str) -> Candidate:
    cond = np.zeros((1, 10, 30, 30), dtype=bool)
    fill = np.zeros((1, 10, 30, 30), dtype=np.float32)
    expected: dict[tuple[int, int], int] = {}
    any_edit = False
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if x.shape != y.shape:
            return Candidate(task_id, "fixed_mask_where", route, None, "skipped", "shape mismatch")
        for r in range(y.shape[0]):
            for c in range(y.shape[1]):
                if int(x[r, c]) == int(y[r, c]):
                    continue
                key = (r, c)
                color = int(y[r, c])
                if key in expected and expected[key] != color:
                    return Candidate(task_id, "fixed_mask_where", route, None, "skipped", "fixed edit conflict")
                expected[key] = color
                any_edit = True
    if not any_edit:
        return Candidate(task_id, "fixed_mask_where", route, None, "skipped", "no edits")
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        for r in range(y.shape[0]):
            for c in range(y.shape[1]):
                key = (r, c)
                if key in expected:
                    if int(y[r, c]) != expected[key]:
                        return Candidate(task_id, "fixed_mask_where", route, None, "skipped", "edit target mismatch")
                elif int(x[r, c]) != int(y[r, c]):
                    return Candidate(task_id, "fixed_mask_where", route, None, "skipped", "non-fixed edit exists")
    for (r, c), color in expected.items():
        cond[0, :, r, c] = True
        fill[0, color, r, c] = 1.0
    raw = make_model(
        [helper.make_node("Where", ["cond", "fill", "input"], ["output"])],
        [numpy_helper.from_array(cond, "cond"), numpy_helper.from_array(fill, "fill")],
        "phase1_fixed_mask_where",
        opset_version=11,
    )
    return Candidate(task_id, "fixed_mask_where", route, raw, "generated", f"fixed_cells={len(expected)}")


def changed_target_colors(examples: list[dict[str, Any]]) -> list[int]:
    colors: set[int] = set()
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if x.shape != y.shape:
            continue
        for src, dst in zip(x.ravel(), y.ravel()):
            if int(src) != int(dst):
                colors.add(int(dst))
    return sorted(colors)


def dedupe_examples(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]] | None, str]:
    seen: dict[bytes, np.ndarray] = {}
    deduped: list[dict[str, Any]] = []
    for ex in examples:
        inp = one_hot_flat(ex["input"])
        out = one_hot_flat(ex["output"])
        key = inp.astype(np.uint8).tobytes()
        old = seen.get(key)
        if old is not None:
            if not np.array_equal(old, out):
                return None, "duplicate input has conflicting output"
            continue
        seen[key] = out
        deduped.append(ex)
    return deduped, "ok"


def flat_index_to_4d(index: int) -> tuple[int, int, int, int]:
    channel_area = 30 * 30
    channel = index // channel_area
    rem = index % channel_area
    row = rem // 30
    col = rem % 30
    return 0, int(channel), int(row), int(col)


def choose_signature_features(inputs: np.ndarray, max_features: int = 128) -> tuple[list[int] | None, str]:
    if inputs.shape[0] <= 1:
        nz = np.flatnonzero(inputs[0] > 0.5)
        return [int(nz[0] if len(nz) else 0)], "single example"
    candidates = np.flatnonzero(np.any(inputs != inputs[0:1], axis=0))
    groups: list[list[int]] = [list(range(inputs.shape[0]))]
    selected: list[int] = []
    selected_set: set[int] = set()
    while any(len(group) > 1 for group in groups):
        best_feature: int | None = None
        best_gain = 0
        for feature in candidates:
            feature_i = int(feature)
            if feature_i in selected_set:
                continue
            gain = 0
            for group in groups:
                if len(group) <= 1:
                    continue
                vals = inputs[group, feature_i]
                ones = int(np.sum(vals > 0.5))
                zeros = len(group) - ones
                gain += len(group) - max(ones, zeros)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_i
        if best_feature is None or best_gain <= 0:
            return None, "could not distinguish all known inputs"
        selected.append(best_feature)
        selected_set.add(best_feature)
        next_groups: list[list[int]] = []
        for group in groups:
            if len(group) <= 1:
                next_groups.append(group)
                continue
            zero_group = [idx for idx in group if inputs[idx, best_feature] < 0.5]
            one_group = [idx for idx in group if inputs[idx, best_feature] >= 0.5]
            if zero_group:
                next_groups.append(zero_group)
            if one_group:
                next_groups.append(one_group)
        groups = next_groups
        if len(selected) >= max_features and any(len(group) > 1 for group in groups):
            return None, f"signature needs more than {max_features} features"
    return selected, f"features={len(selected)}"


def build_signature_scatternd_lookup_candidate(task_id: int, examples: list[dict[str, Any]], route: str) -> Candidate:
    deduped, reason = dedupe_examples(examples)
    if deduped is None:
        return Candidate(task_id, "signature_scatternd_lookup", route, None, "skipped", reason)
    if not deduped:
        return Candidate(task_id, "signature_scatternd_lookup", route, None, "skipped", "no examples")
    inputs = np.stack([one_hot_flat(ex["input"]) for ex in deduped]).astype(np.float32)
    outputs = np.stack([one_hot_flat(ex["output"]) for ex in deduped]).astype(np.float32)
    same_shape = all(grid_to_array(ex["input"]).shape == grid_to_array(ex["output"]).shape for ex in deduped)
    features, reason = choose_signature_features(inputs)
    if features is None:
        return Candidate(task_id, "signature_scatternd_lookup", route, None, "skipped", reason)

    edits: list[tuple[np.ndarray, np.ndarray]] = []
    for input_flat, output_flat in zip(inputs, outputs):
        if same_shape:
            positions = np.flatnonzero(input_flat != output_flat).astype(np.int64)
            values = output_flat[positions].astype(np.float32)
        else:
            positions = np.flatnonzero(output_flat > 0.5).astype(np.int64)
            values = output_flat[positions].astype(np.float32)
        edits.append((positions, values))
    max_edits = max(len(pos) for pos, _ in edits)
    if max_edits == 0:
        return Candidate(task_id, "signature_scatternd_lookup", route, None, "skipped", "no edits")
    if max_edits > 1200:
        return Candidate(task_id, "signature_scatternd_lookup", route, None, "skipped", f"too many edits: {max_edits}")

    index_rows = np.zeros((len(deduped), max_edits * 4), dtype=np.float32)
    value_rows = np.zeros((len(deduped), max_edits), dtype=np.float32)
    for row, (positions, values) in enumerate(edits):
        if len(positions) == 0:
            positions = np.asarray([0], dtype=np.int64)
            values = np.asarray([float(inputs[row, 0]) if same_shape else 0.0], dtype=np.float32)
        if len(positions) < max_edits:
            pad = max_edits - len(positions)
            positions = np.concatenate([positions, np.repeat(positions[0], pad).astype(np.int64)])
            values = np.concatenate([values, np.repeat(values[0], pad).astype(np.float32)])
        coords = np.asarray([flat_index_to_4d(int(pos)) for pos in positions], dtype=np.float32).reshape(-1)
        index_rows[row] = coords
        value_rows[row] = values.astype(np.float32)

    feature_coords = np.asarray([flat_index_to_4d(int(idx)) for idx in features], dtype=np.int64)
    signature_values = inputs[:, features].astype(np.float32)
    initializers = [
        numpy_helper.from_array(feature_coords, "feature_coords"),
        numpy_helper.from_array(signature_values, "signature_values"),
        numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), "match_threshold"),
        numpy_helper.from_array(index_rows, "update_indices_flat"),
        numpy_helper.from_array(value_rows, "update_values"),
        numpy_helper.from_array(np.asarray([max_edits, 4], dtype=np.int64), "indices_shape"),
    ]
    nodes = [
        helper.make_node("GatherND", ["input", "feature_coords"], ["selected_features"]),
        helper.make_node("Sub", ["selected_features", "signature_values"], ["sig_diff"]),
        helper.make_node("Mul", ["sig_diff", "sig_diff"], ["sig_sq"]),
        helper.make_node("ReduceSum", ["sig_sq"], ["sig_dist"], axes=[1], keepdims=0),
        helper.make_node("Less", ["sig_dist", "match_threshold"], ["is_match"]),
        helper.make_node("Cast", ["is_match"], ["match_float"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["match_float"], ["match_row"], axes=[0]),
        helper.make_node("MatMul", ["match_row", "update_indices_flat"], ["selected_indices_flat"]),
        helper.make_node("Reshape", ["selected_indices_flat", "indices_shape"], ["selected_indices_f"]),
        helper.make_node("Cast", ["selected_indices_f"], ["selected_indices"], to=TensorProto.INT64),
        helper.make_node("MatMul", ["match_row", "update_values"], ["selected_values_row"]),
        helper.make_node("Squeeze", ["selected_values_row"], ["selected_values"], axes=[0]),
    ]
    data_input = "input"
    if not same_shape:
        initializers.append(numpy_helper.from_array(np.zeros((1, 10, 30, 30), dtype=np.float32), "zero_tensor"))
        data_input = "zero_tensor"
    nodes.append(helper.make_node("ScatterND", [data_input, "selected_indices", "selected_values"], ["output"]))
    raw = make_model(nodes, initializers, "phase1_signature_scatternd_lookup", opset_version=11)
    return Candidate(
        task_id,
        "signature_scatternd_lookup",
        route,
        raw,
        "generated",
        f"known_examples={len(deduped)},features={len(features)},max_edits={max_edits},same_shape={same_shape}",
    )


def build_neighbor_fill_candidate(
    task_id: int,
    examples: list[dict[str, Any]],
    route: str,
    target_color: int,
    kernel_name: str,
    threshold: int,
) -> Candidate:
    kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
    if kernel_name == "cross":
        kernel[0, 0, 0, 1] = 1
        kernel[0, 0, 1, 0] = 1
        kernel[0, 0, 1, 2] = 1
        kernel[0, 0, 2, 1] = 1
    elif kernel_name == "square":
        kernel[0, 0, :, :] = 1
        kernel[0, 0, 1, 1] = 0
    elif kernel_name == "hline":
        kernel[0, 0, 1, 0] = 1
        kernel[0, 0, 1, 2] = 1
    elif kernel_name == "vline":
        kernel[0, 0, 0, 1] = 1
        kernel[0, 0, 2, 1] = 1
    else:
        return Candidate(task_id, f"neighbor_fill_{kernel_name}_{target_color}_{threshold}", route, None, "skipped", "unknown kernel")

    def apply_rule(inp: np.ndarray) -> np.ndarray:
        out = inp.copy()
        padded = np.pad((inp == target_color).astype(np.int64), 1)
        count = np.zeros_like(inp)
        for rr in range(3):
            for cc in range(3):
                if kernel[0, 0, rr, cc]:
                    count += padded[rr : rr + inp.shape[0], cc : cc + inp.shape[1]]
        out[(inp == 0) & (count >= threshold)] = target_color
        return out

    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if x.shape != y.shape:
            return Candidate(task_id, f"neighbor_fill_{kernel_name}_{target_color}_{threshold}", route, None, "skipped", "shape mismatch")
        if not np.array_equal(apply_rule(x), y):
            return Candidate(task_id, f"neighbor_fill_{kernel_name}_{target_color}_{threshold}", route, None, "skipped", "rule does not fit")

    one = np.asarray([target_color], dtype=np.int64)
    bg = np.asarray([0], dtype=np.int64)
    thresh = np.asarray([float(threshold - 0.5)], dtype=np.float32)
    fill = np.zeros((1, 10, 1, 1), dtype=np.float32)
    fill[0, target_color, 0, 0] = 1.0
    raw = make_model(
        [
            helper.make_node("Gather", ["input", "target_idx"], ["target_ch"], axis=1),
            helper.make_node("Gather", ["input", "bg_idx"], ["bg_ch"], axis=1),
            helper.make_node("Conv", ["target_ch", "kernel"], ["neighbor_count"], pads=[1, 1, 1, 1]),
            helper.make_node("Greater", ["neighbor_count", "threshold"], ["enough"]),
            helper.make_node("Greater", ["bg_ch", "threshold_zero"], ["is_bg"]),
            helper.make_node("And", ["enough", "is_bg"], ["paint_mask"]),
            helper.make_node("Where", ["paint_mask", "fill", "input"], ["output"]),
        ],
        [
            numpy_helper.from_array(one, "target_idx"),
            numpy_helper.from_array(bg, "bg_idx"),
            numpy_helper.from_array(kernel, "kernel"),
            numpy_helper.from_array(thresh, "threshold"),
            numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), "threshold_zero"),
            numpy_helper.from_array(fill, "fill"),
        ],
        f"phase1_neighbor_fill_{kernel_name}",
        opset_version=11,
    )
    return Candidate(task_id, f"neighbor_fill_{kernel_name}_{target_color}_{threshold}", route, raw, "generated", "background neighbor fill")


def build_iterative_neighbor_fill_candidate(
    task_id: int,
    examples: list[dict[str, Any]],
    route: str,
    target_color: int,
    kernel_name: str,
    threshold: int,
    steps: int,
) -> Candidate:
    kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
    if kernel_name == "cross":
        kernel[0, 0, 0, 1] = 1
        kernel[0, 0, 1, 0] = 1
        kernel[0, 0, 1, 2] = 1
        kernel[0, 0, 2, 1] = 1
    elif kernel_name == "square":
        kernel[0, 0, :, :] = 1
        kernel[0, 0, 1, 1] = 0
    elif kernel_name == "hline":
        kernel[0, 0, 1, 0] = 1
        kernel[0, 0, 1, 2] = 1
    elif kernel_name == "vline":
        kernel[0, 0, 0, 1] = 1
        kernel[0, 0, 2, 1] = 1
    else:
        return Candidate(task_id, f"iter_neighbor_fill_{kernel_name}_{target_color}_{threshold}_{steps}", route, None, "skipped", "unknown kernel")

    def apply_rule(inp: np.ndarray) -> np.ndarray:
        out = inp.copy()
        for _ in range(steps):
            padded = np.pad((out == target_color).astype(np.int64), 1)
            count = np.zeros_like(out)
            for rr in range(3):
                for cc in range(3):
                    if kernel[0, 0, rr, cc]:
                        count += padded[rr : rr + out.shape[0], cc : cc + out.shape[1]]
            out[(out == 0) & (count >= threshold)] = target_color
        return out

    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if x.shape != y.shape:
            return Candidate(task_id, f"iter_neighbor_fill_{kernel_name}_{target_color}_{threshold}_{steps}", route, None, "skipped", "shape mismatch")
        if not np.array_equal(apply_rule(x), y):
            return Candidate(task_id, f"iter_neighbor_fill_{kernel_name}_{target_color}_{threshold}_{steps}", route, None, "skipped", "rule does not fit")

    target_idx = np.asarray([target_color], dtype=np.int64)
    bg_idx = np.asarray([0], dtype=np.int64)
    thresh = np.asarray([float(threshold - 0.5)], dtype=np.float32)
    fill = np.zeros((1, 10, 1, 1), dtype=np.float32)
    fill[0, target_color, 0, 0] = 1.0
    nodes: list[onnx.NodeProto] = []
    current = "input"
    for step in range(steps):
        nodes.extend(
            [
                helper.make_node("Gather", [current, "target_idx"], [f"target_ch_{step}"], axis=1),
                helper.make_node("Gather", [current, "bg_idx"], [f"bg_ch_{step}"], axis=1),
                helper.make_node("Conv", [f"target_ch_{step}", "kernel"], [f"neighbor_count_{step}"], pads=[1, 1, 1, 1]),
                helper.make_node("Greater", [f"neighbor_count_{step}", "threshold"], [f"enough_{step}"]),
                helper.make_node("Greater", [f"bg_ch_{step}", "threshold_zero"], [f"is_bg_{step}"]),
                helper.make_node("And", [f"enough_{step}", f"is_bg_{step}"], [f"paint_mask_{step}"]),
                helper.make_node("Where", [f"paint_mask_{step}", "fill", current], [f"filled_{step}"]),
            ]
        )
        current = f"filled_{step}"
    nodes.append(helper.make_node("Identity", [current], ["output"]))
    raw = make_model(
        nodes,
        [
            numpy_helper.from_array(target_idx, "target_idx"),
            numpy_helper.from_array(bg_idx, "bg_idx"),
            numpy_helper.from_array(kernel, "kernel"),
            numpy_helper.from_array(thresh, "threshold"),
            numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), "threshold_zero"),
            numpy_helper.from_array(fill, "fill"),
        ],
        f"phase1_iter_neighbor_fill_{kernel_name}_{steps}",
        opset_version=11,
    )
    return Candidate(
        task_id,
        f"iter_neighbor_fill_{kernel_name}_{target_color}_{threshold}_{steps}",
        route,
        raw,
        "generated",
        f"background iterative neighbor fill steps={steps}",
    )


def flood_masks_from_boundary(inp: np.ndarray, background_color: int = 0) -> tuple[np.ndarray, np.ndarray]:
    background = inp == background_color
    exterior = np.zeros_like(background, dtype=bool)
    height, width = inp.shape
    stack: list[tuple[int, int]] = []
    for r in range(height):
        for c in [0, width - 1]:
            if background[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    for c in range(width):
        for r in [0, height - 1]:
            if background[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and background[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))
    interior = background & ~exterior
    return exterior, interior


def build_boundary_flood_fill_candidate(
    task_id: int,
    examples: list[dict[str, Any]],
    route: str,
    steps: int,
    background_color: int = 0,
) -> Candidate:
    ext_colors: set[int] = set()
    int_colors: set[int] = set()
    name = f"boundary_flood_fill_bg{background_color}_{steps}"
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if x.shape != y.shape:
            return Candidate(task_id, name, route, None, "skipped", "shape mismatch")
        exterior, interior = flood_masks_from_boundary(x, background_color)
        if not np.any(exterior) or not np.any(interior):
            return Candidate(task_id, name, route, None, "skipped", "missing exterior or interior")
        if not np.array_equal(y[~(x == background_color)], x[~(x == background_color)]):
            return Candidate(task_id, name, route, None, "skipped", "non-background cells changed")
        ext_colors.update(int(v) for v in np.unique(y[exterior]))
        int_colors.update(int(v) for v in np.unique(y[interior]))
        if len(ext_colors) > 1 or len(int_colors) > 1:
            return Candidate(task_id, name, route, None, "skipped", "fill colors are not constant")
        expected = x.copy()
        expected[exterior] = next(iter(ext_colors))
        expected[interior] = next(iter(int_colors))
        if not np.array_equal(expected, y):
            return Candidate(task_id, name, route, None, "skipped", "rule does not fit")

    if not ext_colors or not int_colors:
        return Candidate(task_id, name, route, None, "skipped", "no fill colors")
    exterior_color = next(iter(ext_colors))
    interior_color = next(iter(int_colors))
    if exterior_color == background_color and interior_color == background_color:
        return Candidate(task_id, name, route, None, "skipped", "no output fill")

    cross = np.zeros((1, 1, 3, 3), dtype=np.float32)
    cross[0, 0, 0, 1] = 1
    cross[0, 0, 1, 0] = 1
    cross[0, 0, 1, 2] = 1
    cross[0, 0, 2, 1] = 1
    border = np.zeros((1, 1, 30, 30), dtype=bool)
    border[:, :, 0, :] = True
    border[:, :, -1, :] = True
    border[:, :, :, 0] = True
    border[:, :, :, -1] = True
    ext_fill = np.zeros((1, 10, 1, 1), dtype=np.float32)
    int_fill = np.zeros((1, 10, 1, 1), dtype=np.float32)
    ext_fill[0, exterior_color, 0, 0] = 1.0
    int_fill[0, interior_color, 0, 0] = 1.0

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceSum", ["input"], ["valid_sum"], axes=[1], keepdims=1),
        helper.make_node("Greater", ["valid_sum", "half"], ["valid"]),
        helper.make_node("Not", ["valid"], ["invalid"]),
        helper.make_node("Gather", ["input", "bg_idx"], ["bg_ch"], axis=1),
        helper.make_node("Greater", ["bg_ch", "half"], ["bg"]),
        helper.make_node("And", ["valid", "bg"], ["bg_valid"]),
        helper.make_node("Cast", ["invalid"], ["invalid_float"], to=TensorProto.FLOAT),
        helper.make_node("Conv", ["invalid_float", "cross_kernel"], ["invalid_neighbor_count"], pads=[1, 1, 1, 1]),
        helper.make_node("Greater", ["invalid_neighbor_count", "half"], ["near_invalid"]),
        helper.make_node("Or", ["near_invalid", "border_mask"], ["near_outside"]),
        helper.make_node("And", ["bg_valid", "near_outside"], ["exterior_0"]),
    ]
    current = "exterior_0"
    for step in range(steps):
        nodes.extend(
            [
                helper.make_node("Cast", [current], [f"exterior_float_{step}"], to=TensorProto.FLOAT),
                helper.make_node("Conv", [f"exterior_float_{step}", "cross_kernel"], [f"reach_count_{step}"], pads=[1, 1, 1, 1]),
                helper.make_node("Greater", [f"reach_count_{step}", "half"], [f"can_reach_{step}"]),
                helper.make_node("And", ["bg_valid", f"can_reach_{step}"], [f"new_exterior_{step}"]),
                helper.make_node("Or", [current, f"new_exterior_{step}"], [f"exterior_{step + 1}"]),
            ]
        )
        current = f"exterior_{step + 1}"
    nodes.extend(
        [
            helper.make_node("Not", [current], ["not_exterior"]),
            helper.make_node("And", ["bg_valid", "not_exterior"], ["interior"]),
            helper.make_node("Where", [current, "exterior_fill", "input"], ["with_exterior"]),
            helper.make_node("Where", ["interior", "interior_fill", "with_exterior"], ["output"]),
        ]
    )
    raw = make_model(
        nodes,
        [
            numpy_helper.from_array(np.asarray([background_color], dtype=np.int64), "bg_idx"),
            numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), "half"),
            numpy_helper.from_array(cross, "cross_kernel"),
            numpy_helper.from_array(border, "border_mask"),
            numpy_helper.from_array(ext_fill, "exterior_fill"),
            numpy_helper.from_array(int_fill, "interior_fill"),
        ],
        f"phase1_boundary_flood_fill_{steps}",
        opset_version=11,
    )
    return Candidate(
        task_id,
        name,
        route,
        raw,
        "generated",
        f"bg={background_color},exterior={exterior_color},interior={interior_color},steps={steps}",
    )


def periodic_shift_rule(inp: np.ndarray, period: int) -> np.ndarray:
    out = np.zeros_like(inp)
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            out[r, c] = inp[r % 2, (c + 1) % period]
    return out


def build_periodic_shift_fill_candidate(task_id: int, examples: list[dict[str, Any]], route: str) -> Candidate:
    name = "periodic_shift_fill_p2_p3"
    used_periods: set[int] = set()
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if x.shape != y.shape:
            return Candidate(task_id, name, route, None, "skipped", "shape mismatch")
        fit_periods = [period for period in [2, 3] if x.shape[1] >= period and np.array_equal(periodic_shift_rule(x, period), y)]
        if not fit_periods:
            return Candidate(task_id, name, route, None, "skipped", "periodic shift rule does not fit")
        used_periods.add(fit_periods[0])

    masks: dict[str, np.ndarray] = {}
    for period in [2, 3]:
        for sr in [0, 1]:
            for sc in range(period):
                masks[f"p{period}_r{sr}_c{sc}"] = np.zeros((1, 1, 30, 30), dtype=bool)
    for r in range(30):
        for c in range(30):
            sr = r % 2
            masks[f"p2_r{sr}_c{(c + 1) % 2}"][0, 0, r, c] = True
            masks[f"p3_r{sr}_c{(c + 1) % 3}"][0, 0, r, c] = True

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceSum", ["input"], ["valid_sum"], axes=[1], keepdims=1),
        helper.make_node("Greater", ["valid_sum", "half"], ["valid"]),
        helper.make_node("Slice", ["input", "start00", "end00", "axes"], ["color00"]),
        helper.make_node("Slice", ["input", "start01", "end01", "axes"], ["color01"]),
        helper.make_node("Slice", ["input", "start02", "end02", "axes"], ["color02"]),
        helper.make_node("Slice", ["input", "start10", "end10", "axes"], ["color10"]),
        helper.make_node("Slice", ["input", "start11", "end11", "axes"], ["color11"]),
        helper.make_node("Slice", ["input", "start12", "end12", "axes"], ["color12"]),
        helper.make_node("Sub", ["color00", "color02"], ["p2_diff"]),
        helper.make_node("Mul", ["p2_diff", "p2_diff"], ["p2_sq"]),
        helper.make_node("ReduceSum", ["p2_sq"], ["p2_dist"], axes=[0, 1, 2, 3], keepdims=0),
        helper.make_node("Less", ["p2_dist", "half"], ["is_p2"]),
        helper.make_node("Not", ["is_p2"], ["is_p3"]),
    ]
    current = "input"
    for label in ["00", "01", "02", "10", "11", "12"]:
        sr, sc = int(label[0]), int(label[1])
        mask_inputs: list[str] = []
        if sc < 2:
            nodes.extend(
                [
                    helper.make_node("And", ["is_p2", f"p2_r{sr}_c{sc}"], [f"p2_{label}_raw"]),
                    helper.make_node("And", [f"p2_{label}_raw", "valid"], [f"p2_{label}"]),
                ]
            )
            mask_inputs.append(f"p2_{label}")
        nodes.extend(
            [
                helper.make_node("And", ["is_p3", f"p3_r{sr}_c{sc}"], [f"p3_{label}_raw"]),
                helper.make_node("And", [f"p3_{label}_raw", "valid"], [f"p3_{label}"]),
            ]
        )
        mask_inputs.append(f"p3_{label}")
        if len(mask_inputs) == 2:
            nodes.append(helper.make_node("Or", mask_inputs, [f"mask_{label}"]))
            mask_name = f"mask_{label}"
        else:
            mask_name = mask_inputs[0]
        out_name = "output" if label == "12" else f"filled_{label}"
        nodes.append(helper.make_node("Where", [mask_name, f"color{label}", current], [out_name]))
        current = out_name
    raw = make_model(
        nodes,
        [
            numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), "half"),
            numpy_helper.from_array(np.asarray([0, 0, 0, 0], dtype=np.int64), "start00"),
            numpy_helper.from_array(np.asarray([1, 10, 1, 1], dtype=np.int64), "end00"),
            numpy_helper.from_array(np.asarray([0, 0, 0, 1], dtype=np.int64), "start01"),
            numpy_helper.from_array(np.asarray([1, 10, 1, 2], dtype=np.int64), "end01"),
            numpy_helper.from_array(np.asarray([0, 0, 0, 2], dtype=np.int64), "start02"),
            numpy_helper.from_array(np.asarray([1, 10, 1, 3], dtype=np.int64), "end02"),
            numpy_helper.from_array(np.asarray([0, 0, 1, 0], dtype=np.int64), "start10"),
            numpy_helper.from_array(np.asarray([1, 10, 2, 1], dtype=np.int64), "end10"),
            numpy_helper.from_array(np.asarray([0, 0, 1, 1], dtype=np.int64), "start11"),
            numpy_helper.from_array(np.asarray([1, 10, 2, 2], dtype=np.int64), "end11"),
            numpy_helper.from_array(np.asarray([0, 0, 1, 2], dtype=np.int64), "start12"),
            numpy_helper.from_array(np.asarray([1, 10, 2, 3], dtype=np.int64), "end12"),
            numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "axes"),
            *(numpy_helper.from_array(value, key) for key, value in masks.items()),
        ],
        "phase1_periodic_shift_fill_p2_p3",
        opset_version=11,
    )
    return Candidate(task_id, name, route, raw, "generated", f"periods={sorted(used_periods)}")


def diagonal_shift_tile_rule(inp: np.ndarray) -> np.ndarray:
    if inp.shape != (1, 5):
        raise ValueError("expected 1x5 input")
    count = int(np.count_nonzero(inp))
    size = 5 * count
    out = np.zeros((size, size), dtype=np.int64)
    for r in range(size):
        for c in range(size):
            idx = c + r - (size - 1)
            if 0 <= idx < 5:
                out[r, c] = inp[0, idx]
    return out


def _pad_rows(rows: list[list[int]], width: int) -> np.ndarray:
    if not rows:
        rows = [[0, 0, 0, 0]]
    padded = [row[:] for row in rows]
    while len(padded) < width:
        padded.append(padded[-1][:])
    return np.asarray(padded, dtype=np.float32)


def build_diagonal_shift_tile_candidate(task_id: int, examples: list[dict[str, Any]], route: str) -> Candidate:
    name = "diagonal_shift_tile_scatternd"
    used_counts: set[int] = set()
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if x.shape != (1, 5):
            return Candidate(task_id, name, route, None, "skipped", "input is not 1x5")
        try:
            pred = diagonal_shift_tile_rule(x)
        except ValueError as exc:
            return Candidate(task_id, name, route, None, "skipped", str(exc))
        if not np.array_equal(pred, y):
            return Candidate(task_id, name, route, None, "skipped", "diagonal shift rule does not fit")
        used_counts.add(int(np.count_nonzero(x)))

    count_values = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
    max_bg = 25 * 25
    max_formula = (5 * 25 - 10) * 10
    bg_rows: list[np.ndarray] = []
    formula_index_rows: list[np.ndarray] = []
    formula_source_rows: list[np.ndarray] = []
    for count in range(1, 6):
        size = 5 * count
        bg_indices: list[list[int]] = []
        for r in range(size):
            for c in range(size):
                bg_indices.append([0, 0, r, c])
        formula_indices: list[list[int]] = []
        source_indices: list[list[int]] = []
        for r in range(size):
            for j in range(5):
                c = size - 1 - r + j
                if not (0 <= c < size):
                    continue
                for channel in range(10):
                    formula_indices.append([0, channel, r, c])
                    source_indices.append([0, channel, 0, j])
        bg_rows.append(_pad_rows(bg_indices, max_bg).reshape(-1))
        formula_index_rows.append(_pad_rows(formula_indices, max_formula).reshape(-1))
        formula_source_rows.append(_pad_rows(source_indices, max_formula).reshape(-1))

    nodes = [
        helper.make_node("GatherND", ["input", "first_five_bg_coords"], ["first_five_bg"]),
        helper.make_node("ReduceSum", ["first_five_bg"], ["zero_count"], axes=[0], keepdims=0),
        helper.make_node("Sub", ["five", "zero_count"], ["nonzero_count"]),
        helper.make_node("Sub", ["nonzero_count", "count_values"], ["count_diff"]),
        helper.make_node("Mul", ["count_diff", "count_diff"], ["count_sq"]),
        helper.make_node("Less", ["count_sq", "half"], ["count_match_bool"]),
        helper.make_node("Cast", ["count_match_bool"], ["count_match"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["count_match"], ["count_match_row"], axes=[0]),
        helper.make_node("MatMul", ["count_match_row", "bg_index_rows"], ["selected_bg_flat_row"]),
        helper.make_node("Squeeze", ["selected_bg_flat_row"], ["selected_bg_flat"], axes=[0]),
        helper.make_node("Reshape", ["selected_bg_flat", "bg_shape"], ["selected_bg_f"]),
        helper.make_node("Cast", ["selected_bg_f"], ["selected_bg"], to=TensorProto.INT64),
        helper.make_node("ConstantOfShape", ["tensor_shape"], ["zero_tensor"], value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])),
        helper.make_node("ScatterND", ["zero_tensor", "selected_bg", "bg_updates"], ["bg_base"]),
        helper.make_node("MatMul", ["count_match_row", "formula_index_rows"], ["selected_formula_flat_row"]),
        helper.make_node("Squeeze", ["selected_formula_flat_row"], ["selected_formula_flat"], axes=[0]),
        helper.make_node("Reshape", ["selected_formula_flat", "formula_shape"], ["selected_formula_f"]),
        helper.make_node("Cast", ["selected_formula_f"], ["selected_formula"], to=TensorProto.INT64),
        helper.make_node("MatMul", ["count_match_row", "formula_source_rows"], ["selected_source_flat_row"]),
        helper.make_node("Squeeze", ["selected_source_flat_row"], ["selected_source_flat"], axes=[0]),
        helper.make_node("Reshape", ["selected_source_flat", "formula_shape"], ["selected_source_f"]),
        helper.make_node("Cast", ["selected_source_f"], ["selected_source"], to=TensorProto.INT64),
        helper.make_node("GatherND", ["input", "selected_source"], ["selected_values"]),
        helper.make_node("ScatterND", ["bg_base", "selected_formula", "selected_values"], ["output"]),
    ]
    raw = make_model(
        nodes,
        [
            numpy_helper.from_array(np.asarray([[0, 0, 0, c] for c in range(5)], dtype=np.int64), "first_five_bg_coords"),
            numpy_helper.from_array(np.asarray([5.0], dtype=np.float32), "five"),
            numpy_helper.from_array(count_values, "count_values"),
            numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), "half"),
            numpy_helper.from_array(np.stack(bg_rows).astype(np.float32), "bg_index_rows"),
            numpy_helper.from_array(np.stack(formula_index_rows).astype(np.float32), "formula_index_rows"),
            numpy_helper.from_array(np.stack(formula_source_rows).astype(np.float32), "formula_source_rows"),
            numpy_helper.from_array(np.asarray([max_bg, 4], dtype=np.int64), "bg_shape"),
            numpy_helper.from_array(np.asarray([max_formula, 4], dtype=np.int64), "formula_shape"),
            numpy_helper.from_array(np.asarray([1, 10, 30, 30], dtype=np.int64), "tensor_shape"),
            numpy_helper.from_array(np.ones((max_bg,), dtype=np.float32), "bg_updates"),
        ],
        "phase1_diagonal_shift_tile_scatternd",
        opset_version=11,
    )
    return Candidate(task_id, name, route, raw, "generated", f"counts={sorted(used_counts)}")


def ring_depth_reverse_rule(inp: np.ndarray) -> np.ndarray:
    if inp.ndim != 2 or inp.shape[0] != inp.shape[1] or inp.shape[0] % 2 != 0:
        raise ValueError("input is not an even square")
    n = inp.shape[0]
    depth_count = n // 2
    out = inp.copy()
    for r in range(n):
        for c in range(n):
            depth = min(r, c, n - 1 - r, n - 1 - c)
            rev = depth_count - 1 - depth
            out[r, c] = inp[rev, rev]
    return out


def build_ring_depth_reverse_color_map_candidate(task_id: int, examples: list[dict[str, Any]], route: str) -> Candidate:
    name = "ring_depth_reverse_color_map"
    allowed_sizes = [6, 8, 10, 12, 14, 16, 18]
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if x.shape[0] not in allowed_sizes:
            return Candidate(task_id, name, route, None, "skipped", f"unsupported size {x.shape}")
        try:
            pred = ring_depth_reverse_rule(x)
        except ValueError as exc:
            return Candidate(task_id, name, route, None, "skipped", str(exc))
        if not np.array_equal(pred, y):
            return Candidate(task_id, name, route, None, "skipped", "ring reverse rule does not fit")

    sentinel_cols = np.asarray(allowed_sizes, dtype=np.int64)
    signature_values = np.zeros((len(allowed_sizes), len(allowed_sizes)), dtype=np.float32)
    source_coords = np.zeros((len(allowed_sizes), 9, 3), dtype=np.int64)
    target_coords = np.zeros((len(allowed_sizes), 9, 3), dtype=np.int64)
    active_mask = np.zeros((len(allowed_sizes), 9, 1, 1), dtype=np.float32)
    for row, size in enumerate(allowed_sizes):
        signature_values[row] = (sentinel_cols < size).astype(np.float32)
        depth_count = size // 2
        for depth in range(9):
            source_coords[row, depth] = [0, depth, depth]
            target_depth = max(0, depth_count - 1 - depth)
            target_coords[row, depth] = [0, target_depth, target_depth]
            if depth < depth_count:
                active_mask[row, depth, 0, 0] = 1.0

    bg_identity = np.zeros((1, 10, 10), dtype=np.float32)
    bg_identity[0, 0, 0] = 1.0
    nodes = [
        helper.make_node("Transpose", ["input"], ["input_nhwc"], perm=[0, 2, 3, 1]),
        helper.make_node("Gather", ["input_nhwc", "row_zero"], ["row0_nhwc"], axis=1),
        helper.make_node("Gather", ["row0_nhwc", "sentinel_cols"], ["sentinel_vectors"], axis=2),
        helper.make_node("ReduceSum", ["sentinel_vectors"], ["sentinel_occupancy"], axes=[3], keepdims=0),
        helper.make_node("Squeeze", ["sentinel_occupancy"], ["sentinel_occupancy_flat"], axes=[0, 1]),
        helper.make_node("Sub", ["sentinel_occupancy_flat", "signature_values"], ["size_diff"]),
        helper.make_node("Mul", ["size_diff", "size_diff"], ["size_sq"]),
        helper.make_node("ReduceSum", ["size_sq"], ["size_dist"], axes=[1], keepdims=0),
        helper.make_node("Less", ["size_dist", "match_threshold"], ["size_match_bool"]),
        helper.make_node("Cast", ["size_match_bool"], ["size_match"], to=TensorProto.FLOAT),
        helper.make_node("GatherND", ["input_nhwc", "source_coords"], ["source_vectors"]),
        helper.make_node("GatherND", ["input_nhwc", "target_coords"], ["target_vectors"]),
        helper.make_node("Unsqueeze", ["target_vectors"], ["target_col"], axes=[3]),
        helper.make_node("Unsqueeze", ["source_vectors"], ["source_row"], axes=[2]),
        helper.make_node("Mul", ["target_col", "source_row"], ["ring_outer"]),
        helper.make_node("Mul", ["ring_outer", "active_mask"], ["ring_outer_active"]),
        helper.make_node("ReduceSum", ["ring_outer_active"], ["mapping_by_size"], axes=[1], keepdims=0),
        helper.make_node("Add", ["mapping_by_size", "bg_identity"], ["mapping_with_bg"]),
        helper.make_node("Reshape", ["mapping_with_bg", "mapping_rows_shape"], ["mapping_rows"]),
        helper.make_node("Unsqueeze", ["size_match"], ["size_match_row"], axes=[0]),
        helper.make_node("MatMul", ["size_match_row", "mapping_rows"], ["selected_mapping_flat"]),
        helper.make_node("Reshape", ["selected_mapping_flat", "mapping_shape"], ["selected_mapping"]),
        helper.make_node("Transpose", ["selected_mapping"], ["selected_mapping_t"], perm=[1, 0]),
        helper.make_node("Reshape", ["input_nhwc", "flat_input_shape"], ["input_flat_nhwc"]),
        helper.make_node("MatMul", ["input_flat_nhwc", "selected_mapping_t"], ["output_flat_nhwc"]),
        helper.make_node("Reshape", ["output_flat_nhwc", "nhwc_shape"], ["output_nhwc"]),
        helper.make_node("Transpose", ["output_nhwc"], ["output"], perm=[0, 3, 1, 2]),
    ]
    raw = make_model(
        nodes,
        [
            numpy_helper.from_array(sentinel_cols, "sentinel_cols"),
            numpy_helper.from_array(np.asarray([0], dtype=np.int64), "row_zero"),
            numpy_helper.from_array(signature_values, "signature_values"),
            numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), "match_threshold"),
            numpy_helper.from_array(source_coords, "source_coords"),
            numpy_helper.from_array(target_coords, "target_coords"),
            numpy_helper.from_array(active_mask, "active_mask"),
            numpy_helper.from_array(bg_identity, "bg_identity"),
            numpy_helper.from_array(np.asarray([len(allowed_sizes), 100], dtype=np.int64), "mapping_rows_shape"),
            numpy_helper.from_array(np.asarray([10, 10], dtype=np.int64), "mapping_shape"),
            numpy_helper.from_array(np.asarray([900, 10], dtype=np.int64), "flat_input_shape"),
            numpy_helper.from_array(np.asarray([1, 30, 30, 10], dtype=np.int64), "nhwc_shape"),
        ],
        "phase1_ring_depth_reverse_color_map",
        opset_version=11,
    )
    return Candidate(task_id, name, route, raw, "generated", f"sizes={allowed_sizes}")


def tile_prefix_blocks_rule(inp: np.ndarray) -> np.ndarray:
    if inp.shape != (3, 3):
        raise ValueError("input is not 3x3")
    zero_count = int(np.sum(inp == 0))
    active_blocks = 9 - zero_count
    size = 3 * zero_count
    out = np.zeros((size, size), dtype=inp.dtype)
    if size == 0:
        return out
    for block in range(active_blocks):
        br = block // max(1, zero_count)
        bc = block % max(1, zero_count)
        if br >= zero_count:
            break
        out[br * 3 : br * 3 + 3, bc * 3 : bc * 3 + 3] = inp
    return out


def build_tile_prefix_blocks_candidate(task_id: int, examples: list[dict[str, Any]], route: str) -> Candidate:
    name = "tile_prefix_blocks"
    used_zero_counts: set[int] = set()
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        try:
            pred = tile_prefix_blocks_rule(x)
        except ValueError as exc:
            return Candidate(task_id, name, route, None, "skipped", str(exc))
        if not np.array_equal(pred, y):
            return Candidate(task_id, name, route, None, "skipped", "tile prefix rule does not fit")
        used_zero_counts.add(int(np.sum(x == 0)))

    count_values = np.arange(10, dtype=np.float32)
    mask_rows = np.zeros((10, 30 * 30), dtype=np.float32)
    bg_rows = np.zeros((10, 30 * 30), dtype=np.float32)
    bg_channel = np.zeros((1, 10, 1, 1), dtype=np.float32)
    bg_channel[0, 0, 0, 0] = 1.0
    for zero_count in range(10):
        active_blocks = 9 - zero_count
        if zero_count <= 0:
            continue
        mask = np.zeros((30, 30), dtype=np.float32)
        for block in range(active_blocks):
            br = block // zero_count
            bc = block % zero_count
            if br >= zero_count:
                break
            mask[br * 3 : br * 3 + 3, bc * 3 : bc * 3 + 3] = 1.0
        valid = np.zeros((30, 30), dtype=np.float32)
        valid[: zero_count * 3, : zero_count * 3] = 1.0
        mask_rows[zero_count] = mask.reshape(-1)
        bg_rows[zero_count] = (valid - mask).reshape(-1)

    nodes = [
        helper.make_node("Slice", ["input", "crop_starts", "crop_ends", "crop_axes"], ["tile_source"]),
        helper.make_node("Tile", ["tile_source", "tile_repeats"], ["tiled"]),
        helper.make_node("GatherND", ["input", "zero_coords"], ["zero_channel_values"]),
        helper.make_node("ReduceSum", ["zero_channel_values"], ["zero_count"], axes=[0], keepdims=0),
        helper.make_node("Sub", ["zero_count", "count_values"], ["count_diff"]),
        helper.make_node("Mul", ["count_diff", "count_diff"], ["count_sq"]),
        helper.make_node("Less", ["count_sq", "half"], ["count_match_bool"]),
        helper.make_node("Cast", ["count_match_bool"], ["count_match"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["count_match"], ["count_match_row"], axes=[0]),
        helper.make_node("MatMul", ["count_match_row", "mask_rows"], ["selected_mask_flat"]),
        helper.make_node("Reshape", ["selected_mask_flat", "mask_shape"], ["selected_mask"]),
        helper.make_node("Mul", ["tiled", "selected_mask"], ["active_tiles"]),
        helper.make_node("MatMul", ["count_match_row", "bg_rows"], ["selected_bg_flat"]),
        helper.make_node("Reshape", ["selected_bg_flat", "mask_shape"], ["selected_bg"]),
        helper.make_node("Mul", ["bg_channel", "selected_bg"], ["background"]),
        helper.make_node("Add", ["active_tiles", "background"], ["output"]),
    ]
    raw = make_model(
        nodes,
        [
            numpy_helper.from_array(np.asarray([0, 0, 0, 0], dtype=np.int64), "crop_starts"),
            numpy_helper.from_array(np.asarray([1, 10, 3, 3], dtype=np.int64), "crop_ends"),
            numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "crop_axes"),
            numpy_helper.from_array(np.asarray([1, 1, 10, 10], dtype=np.int64), "tile_repeats"),
            numpy_helper.from_array(np.asarray([[0, 0, r, c] for r in range(3) for c in range(3)], dtype=np.int64), "zero_coords"),
            numpy_helper.from_array(count_values, "count_values"),
            numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), "half"),
            numpy_helper.from_array(mask_rows, "mask_rows"),
            numpy_helper.from_array(bg_rows, "bg_rows"),
            numpy_helper.from_array(bg_channel, "bg_channel"),
            numpy_helper.from_array(np.asarray([1, 1, 30, 30], dtype=np.int64), "mask_shape"),
        ],
        "phase1_tile_prefix_blocks",
        opset_version=11,
    )
    return Candidate(task_id, name, route, raw, "generated", f"zero_counts={sorted(used_zero_counts)}")


def build_tile_prefix_blocks_sparse_candidate(task_id: int, examples: list[dict[str, Any]], route: str) -> Candidate:
    name = "tile_prefix_blocks_sparse_argmax"
    allowed_counts = [3, 4, 5, 6, 7]
    used_zero_counts: set[int] = set()
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        try:
            pred = tile_prefix_blocks_rule(x)
        except ValueError as exc:
            return Candidate(task_id, name, route, None, "skipped", str(exc))
        zero_count = int(np.sum(x == 0))
        if zero_count not in allowed_counts:
            return Candidate(task_id, name, route, None, "skipped", f"unsupported zero_count={zero_count}")
        if not np.array_equal(pred, y):
            return Candidate(task_id, name, route, None, "skipped", "tile prefix rule does not fit")
        used_zero_counts.add(zero_count)

    max_active = max((9 - z) * 9 for z in allowed_counts)
    max_bg = max(z * z * 9 - (9 - z) * 9 for z in allowed_counts)
    active_nrc_rows: list[np.ndarray] = []
    source_rows: list[np.ndarray] = []
    bg_rows: list[np.ndarray] = []
    for zero_count in allowed_counts:
        active_blocks = 9 - zero_count
        active_nrc: list[list[int]] = []
        source_coords: list[list[int]] = []
        active_cells: set[tuple[int, int]] = set()
        for block in range(active_blocks):
            br = block // zero_count
            bc = block % zero_count
            for rr in range(3):
                for cc in range(3):
                    out_r = br * 3 + rr
                    out_c = bc * 3 + cc
                    active_cells.add((out_r, out_c))
                    active_nrc.append([0, out_r, out_c])
                    source_coords.append([0, rr, cc])
        bg_indices: list[list[int]] = []
        for r in range(zero_count * 3):
            for c in range(zero_count * 3):
                if (r, c) not in active_cells:
                    bg_indices.append([0, 0, r, c])
        active_nrc_rows.append(_pad_rows(active_nrc, max_active).reshape(-1))
        source_rows.append(_pad_rows(source_coords, max_active).reshape(-1))
        bg_rows.append(_pad_rows(bg_indices, max_bg).reshape(-1))

    nodes = [
        helper.make_node("Transpose", ["input"], ["input_nhwc"], perm=[0, 2, 3, 1]),
        helper.make_node("GatherND", ["input", "zero_coords"], ["zero_channel_values"]),
        helper.make_node("ReduceSum", ["zero_channel_values"], ["zero_count"], axes=[0], keepdims=0),
        helper.make_node("Sub", ["zero_count", "count_values"], ["count_diff"]),
        helper.make_node("Mul", ["count_diff", "count_diff"], ["count_sq"]),
        helper.make_node("Less", ["count_sq", "half"], ["count_match_bool"]),
        helper.make_node("Cast", ["count_match_bool"], ["count_match"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["count_match"], ["count_match_row"], axes=[0]),
        helper.make_node("MatMul", ["count_match_row", "bg_index_rows"], ["selected_bg_flat_row"]),
        helper.make_node("Squeeze", ["selected_bg_flat_row"], ["selected_bg_flat"], axes=[0]),
        helper.make_node("Reshape", ["selected_bg_flat", "bg_shape"], ["selected_bg_f"]),
        helper.make_node("Cast", ["selected_bg_f"], ["selected_bg"], to=TensorProto.INT64),
        helper.make_node("ConstantOfShape", ["tensor_shape"], ["zero_tensor"], value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])),
        helper.make_node("ScatterND", ["zero_tensor", "selected_bg", "bg_updates"], ["bg_base"]),
        helper.make_node("MatMul", ["count_match_row", "active_nrc_rows"], ["selected_active_nrc_flat_row"]),
        helper.make_node("Squeeze", ["selected_active_nrc_flat_row"], ["selected_active_nrc_flat"], axes=[0]),
        helper.make_node("Reshape", ["selected_active_nrc_flat", "active_nrc_shape"], ["selected_active_nrc_f"]),
        helper.make_node("Cast", ["selected_active_nrc_f"], ["selected_active_nrc"], to=TensorProto.INT64),
        helper.make_node("MatMul", ["count_match_row", "source_rows"], ["selected_source_flat_row"]),
        helper.make_node("Squeeze", ["selected_source_flat_row"], ["selected_source_flat"], axes=[0]),
        helper.make_node("Reshape", ["selected_source_flat", "active_nrc_shape"], ["selected_source_f"]),
        helper.make_node("Cast", ["selected_source_f"], ["selected_source"], to=TensorProto.INT64),
        helper.make_node("GatherND", ["input_nhwc", "selected_source"], ["source_vectors"]),
        helper.make_node("ArgMax", ["source_vectors"], ["active_channels"], axis=1, keepdims=0),
        helper.make_node("Unsqueeze", ["active_channels"], ["active_channel_col"], axes=[1]),
        helper.make_node("Slice", ["selected_active_nrc", "n_start", "n_end", "col_axes"], ["active_n_col"]),
        helper.make_node("Slice", ["selected_active_nrc", "rc_start", "rc_end", "col_axes"], ["active_rc_cols"]),
        helper.make_node("Concat", ["active_n_col", "active_channel_col", "active_rc_cols"], ["active_indices"], axis=1),
        helper.make_node("ScatterND", ["bg_base", "active_indices", "active_updates"], ["output"]),
    ]
    raw = make_model(
        nodes,
        [
            numpy_helper.from_array(np.asarray([[0, 0, r, c] for r in range(3) for c in range(3)], dtype=np.int64), "zero_coords"),
            numpy_helper.from_array(np.asarray(allowed_counts, dtype=np.float32), "count_values"),
            numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), "half"),
            numpy_helper.from_array(np.stack(bg_rows).astype(np.float32), "bg_index_rows"),
            numpy_helper.from_array(np.stack(active_nrc_rows).astype(np.float32), "active_nrc_rows"),
            numpy_helper.from_array(np.stack(source_rows).astype(np.float32), "source_rows"),
            numpy_helper.from_array(np.asarray([max_bg, 4], dtype=np.int64), "bg_shape"),
            numpy_helper.from_array(np.asarray([max_active, 3], dtype=np.int64), "active_nrc_shape"),
            numpy_helper.from_array(np.asarray([1, 10, 30, 30], dtype=np.int64), "tensor_shape"),
            numpy_helper.from_array(np.ones((max_bg,), dtype=np.float32), "bg_updates"),
            numpy_helper.from_array(np.ones((max_active,), dtype=np.float32), "active_updates"),
            numpy_helper.from_array(np.asarray([0], dtype=np.int64), "n_start"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "n_end"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "rc_start"),
            numpy_helper.from_array(np.asarray([3], dtype=np.int64), "rc_end"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "col_axes"),
        ],
        "phase1_tile_prefix_blocks_sparse_argmax",
        opset_version=11,
    )
    return Candidate(task_id, name, route, raw, "generated", f"zero_counts={sorted(used_zero_counts)}")


def build_common_crop_candidate(task_id: int, examples: list[dict[str, Any]], route: str) -> Candidate:
    candidates: set[tuple[int, int, int, int]] | None = None
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        oh, ow = y.shape
        ih, iw = x.shape
        if oh > ih or ow > iw:
            return Candidate(task_id, "common_fixed_crop", route, None, "skipped", "output larger than input")
        local: set[tuple[int, int, int, int]] = set()
        for top in range(ih - oh + 1):
            for left in range(iw - ow + 1):
                if np.array_equal(x[top : top + oh, left : left + ow], y):
                    local.add((top, left, oh, ow))
        if not local:
            return Candidate(task_id, "common_fixed_crop", route, None, "skipped", "no crop fits")
        candidates = local if candidates is None else candidates.intersection(local)
        if not candidates:
            return Candidate(task_id, "common_fixed_crop", route, None, "skipped", "no common crop")
    if not candidates:
        return Candidate(task_id, "common_fixed_crop", route, None, "skipped", "no examples")
    top, left, oh, ow = sorted(candidates)[0]
    raw = make_model(
        [
            helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["cropped"]),
            helper.make_node("Pad", ["cropped"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30 - oh, 30 - ow], value=0.0),
        ],
        [
            numpy_helper.from_array(np.asarray([0, 0, top, left], dtype=np.int64), "starts"),
            numpy_helper.from_array(np.asarray([1, 10, top + oh, left + ow], dtype=np.int64), "ends"),
            numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "axes"),
        ],
        "phase1_common_fixed_crop",
    )
    return Candidate(task_id, "common_fixed_crop", route, raw, "generated", f"top={top},left={left},h={oh},w={ow}")


def crop_color_map_for(examples: list[dict[str, Any]], top: int, left: int, oh: int, ow: int) -> tuple[dict[int, int] | None, str]:
    mapped: dict[int, int] = {}
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if y.shape != (oh, ow):
            return None, "output shape mismatch"
        if top + oh > x.shape[0] or left + ow > x.shape[1]:
            return None, "crop outside input"
        crop = x[top : top + oh, left : left + ow]
        for src, dst in zip(crop.ravel(), y.ravel()):
            src_i, dst_i = int(src), int(dst)
            if src_i in mapped and mapped[src_i] != dst_i:
                return None, "conflicting crop color map"
            mapped[src_i] = dst_i
    return mapped, "ok"


def build_common_crop_color_map_candidate(task_id: int, examples: list[dict[str, Any]], route: str) -> Candidate:
    candidates: set[tuple[int, int, int, int]] | None = None
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        oh, ow = y.shape
        ih, iw = x.shape
        if oh > ih or ow > iw:
            return Candidate(task_id, "common_fixed_crop_color_map", route, None, "skipped", "output larger than input")
        local: set[tuple[int, int, int, int]] = set()
        for top in range(ih - oh + 1):
            for left in range(iw - ow + 1):
                mapped, _ = crop_color_map_for([ex], top, left, oh, ow)
                if mapped is not None:
                    local.add((top, left, oh, ow))
        if not local:
            return Candidate(task_id, "common_fixed_crop_color_map", route, None, "skipped", "no crop color map fits")
        candidates = local if candidates is None else candidates.intersection(local)
        if not candidates:
            return Candidate(task_id, "common_fixed_crop_color_map", route, None, "skipped", "no common crop color map")
    if not candidates:
        return Candidate(task_id, "common_fixed_crop_color_map", route, None, "skipped", "no examples")

    for top, left, oh, ow in sorted(candidates):
        mapped, reason = crop_color_map_for(examples, top, left, oh, ow)
        if mapped is None:
            continue
        weight = np.zeros((10, 10, 1, 1), dtype=np.float32)
        for src in range(10):
            dst = mapped.get(src, src)
            weight[dst, src, 0, 0] = 1.0
        raw = make_model(
            [
                helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["cropped"]),
                helper.make_node("Conv", ["cropped", "weight"], ["colored"]),
                helper.make_node("Pad", ["colored"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30 - oh, 30 - ow], value=0.0),
            ],
            [
                numpy_helper.from_array(np.asarray([0, 0, top, left], dtype=np.int64), "starts"),
                numpy_helper.from_array(np.asarray([1, 10, top + oh, left + ow], dtype=np.int64), "ends"),
                numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "axes"),
                numpy_helper.from_array(weight, "weight"),
            ],
            "phase1_common_fixed_crop_color_map",
        )
        return Candidate(
            task_id,
            "common_fixed_crop_color_map",
            route,
            raw,
            "generated",
            f"top={top},left={left},h={oh},w={ow},mapping={mapped}",
        )
    return Candidate(task_id, "common_fixed_crop_color_map", route, None, "skipped", reason)


def sparse_candidates(task_id: int, examples: list[dict[str, Any]], route: str) -> list[Candidate]:
    candidates = [
        build_constant_sparse_output_candidate(task_id, examples, route),
        build_global_transform_color_map_candidate(task_id, examples, route),
        build_periodic_shift_fill_candidate(task_id, examples, route),
        build_ring_depth_reverse_color_map_candidate(task_id, examples, route),
        build_boundary_flood_fill_candidate(task_id, examples, route, 12),
        build_boundary_flood_fill_candidate(task_id, examples, route, 18),
        build_boundary_flood_fill_candidate(task_id, examples, route, 24),
        build_boundary_flood_fill_candidate(task_id, examples, route, 30),
        build_signature_scatternd_lookup_candidate(task_id, examples, route),
        build_identity_candidate(task_id, examples, route),
        build_conv_color_map_candidate(task_id, examples, route),
        build_fixed_mask_where_candidate(task_id, examples, route),
    ]
    for color in changed_target_colors(examples):
        for kernel in ["hline", "vline", "cross", "square"]:
            max_threshold = 2 if kernel in {"hline", "vline"} else (4 if kernel == "cross" else 8)
            for threshold in range(1, max_threshold + 1):
                candidates.append(build_neighbor_fill_candidate(task_id, examples, route, color, kernel, threshold))
                for steps in [2, 3, 5, 8]:
                    candidates.append(build_iterative_neighbor_fill_candidate(task_id, examples, route, color, kernel, threshold, steps))
    return candidates


def crop_candidates(task_id: int, examples: list[dict[str, Any]], route: str) -> list[Candidate]:
    return [
        build_constant_sparse_output_candidate(task_id, examples, route),
        build_global_transform_color_map_candidate(task_id, examples, route),
        build_common_crop_candidate(task_id, examples, route),
        build_common_crop_color_map_candidate(task_id, examples, route),
        build_tile_prefix_blocks_sparse_candidate(task_id, examples, route),
        build_tile_prefix_blocks_candidate(task_id, examples, route),
        build_diagonal_shift_tile_candidate(task_id, examples, route),
        build_signature_scatternd_lookup_candidate(task_id, examples, route),
        build_identity_candidate(task_id, examples, route),
    ]
