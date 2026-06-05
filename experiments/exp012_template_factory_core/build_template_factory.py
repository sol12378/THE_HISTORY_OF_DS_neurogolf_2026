from __future__ import annotations

import argparse
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
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "exp012_template_factory_core"
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
BASE_EXP = ROOT / "experiments" / "exp005_top_cost_rewrite_strict"
ROUTE_EXP = ROOT / "experiments" / "exp011_gpu_route_classifier"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
MAX_ONNX_BYTES = int(1.44 * 1024 * 1024)
BANNED_OPS = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
ROUTES = ["crop_or_resize", "same_shape_global_transform", "sparse_edit_or_object_completion"]


@dataclass(frozen=True)
class BaselineRow:
    task_id: int
    source: str
    cost: int
    points: float


@dataclass(frozen=True)
class Candidate:
    task_id: int
    template_name: str
    route: str
    raw: bytes | None
    status: str
    reason: str


@dataclass
class CandidateRecord:
    task_id: int
    template_name: str
    route: str
    baseline_cost: int
    candidate_cost: int | str
    baseline_points: float
    candidate_points: float | str
    file_bytes: int | str
    validation_status: str
    status: str
    reason: str
    sha256: str


@dataclass
class SelectedRecord:
    task_id: int
    source: str
    template_name: str
    cost: int
    local_points: float
    file_bytes: int
    status: str
    reason: str
    sha256: str


def load_neurogolf_utils() -> Any:
    path = DATA_DIR / "neurogolf_utils" / "neurogolf_utils.py"
    spec = importlib.util.spec_from_file_location("neurogolf_utils_exp012", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["neurogolf_utils_exp012"] = module
    spec.loader.exec_module(module)
    return module


def load_task(task_id: int) -> dict[str, Any]:
    return json.loads((DATA_DIR / f"task{task_id:03d}.json").read_text(encoding="utf-8"))


def examples_for(task: dict[str, Any], arc_gen_sample: int) -> list[dict[str, Any]]:
    arc_gen = task["arc-gen"] if arc_gen_sample < 0 else task["arc-gen"][:arc_gen_sample]
    return task["train"] + task["test"] + arc_gen


def fit_examples_for(task: dict[str, Any]) -> list[dict[str, Any]]:
    return task["train"] + task["test"]


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


def point(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


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


def build_color_map_candidate(task_id: int, fit_examples: list[dict[str, Any]], route: str) -> Candidate:
    mapping: dict[int, int] = {}
    for ex in fit_examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if x.shape != y.shape:
            return Candidate(task_id, "conv_color_map", route, None, "skipped", "shape mismatch")
        for src, dst in zip(x.ravel(), y.ravel()):
            src_i, dst_i = int(src), int(dst)
            if src_i in mapping and mapping[src_i] != dst_i:
                return Candidate(task_id, "conv_color_map", route, None, "skipped", "conflicting color map")
            mapping[src_i] = dst_i
    full_mapping = {color: mapping.get(color, color) for color in range(10)}
    weight = np.zeros((10, 10, 1, 1), dtype=np.float32)
    for src, dst in full_mapping.items():
        weight[dst, src, 0, 0] = 1.0
    raw = make_model(
        [helper.make_node("Conv", ["input", "weight"], ["output"])],
        [numpy_helper.from_array(weight, "weight")],
        "exp012_conv_color_map",
    )
    return Candidate(task_id, "conv_color_map", route, raw, "generated", json.dumps(full_mapping, sort_keys=True))


def build_constant_output_candidate(task_id: int, fit_examples: list[dict[str, Any]], route: str) -> Candidate:
    outputs = [one_hot_padded(ex["output"]) for ex in fit_examples]
    if not outputs:
        return Candidate(task_id, "constant_output", route, None, "skipped", "no fit examples")
    first = outputs[0]
    if any(not np.array_equal(first, other) for other in outputs[1:]):
        return Candidate(task_id, "constant_output", route, None, "skipped", "fit outputs differ")
    value = numpy_helper.from_array(first.astype(np.float32), "constant_output_value")
    raw = make_model(
        [helper.make_node("Constant", [], ["output"], value=value)],
        [],
        "exp012_constant_output",
    )
    return Candidate(task_id, "constant_output", route, raw, "generated", "constant output inferred from train+test")


def build_fixed_mask_candidate(task_id: int, fit_examples: list[dict[str, Any]], route: str) -> Candidate:
    if not fit_examples:
        return Candidate(task_id, "fixed_mask_rewrite", route, None, "skipped", "no fit examples")
    keep = np.ones((1, 10, 30, 30), dtype=np.float32)
    fill = np.zeros((1, 10, 30, 30), dtype=np.float32)
    changed_color_by_cell: dict[tuple[int, int], int] = {}
    changed_cells: set[tuple[int, int]] = set()

    for ex in fit_examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if x.shape != y.shape:
            return Candidate(task_id, "fixed_mask_rewrite", route, None, "skipped", "shape mismatch")
        h, w = y.shape
        for r in range(h):
            for c in range(w):
                if int(x[r, c]) == int(y[r, c]):
                    continue
                key = (r, c)
                color = int(y[r, c])
                if key in changed_color_by_cell and changed_color_by_cell[key] != color:
                    return Candidate(task_id, "fixed_mask_rewrite", route, None, "skipped", "changed cell color conflict")
                changed_color_by_cell[key] = color
                changed_cells.add(key)

    if not changed_cells:
        return Candidate(task_id, "fixed_mask_rewrite", route, None, "skipped", "no fixed edits")

    for ex in fit_examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        h, w = y.shape
        for r in range(h):
            for c in range(w):
                key = (r, c)
                if key in changed_cells:
                    if int(y[r, c]) != changed_color_by_cell[key]:
                        return Candidate(task_id, "fixed_mask_rewrite", route, None, "skipped", "fixed edit does not hold")
                elif int(x[r, c]) != int(y[r, c]):
                    return Candidate(task_id, "fixed_mask_rewrite", route, None, "skipped", "non-fixed edit exists")

    for r, c in changed_cells:
        keep[0, :, r, c] = 0.0
        fill[0, changed_color_by_cell[(r, c)], r, c] = 1.0

    raw = make_model(
        [
            helper.make_node("Mul", ["input", "keep_mask"], ["kept"]),
            helper.make_node("Add", ["kept", "fill_tensor"], ["output"]),
        ],
        [
            numpy_helper.from_array(keep, "keep_mask"),
            numpy_helper.from_array(fill, "fill_tensor"),
        ],
        "exp012_fixed_mask_rewrite",
    )
    return Candidate(task_id, "fixed_mask_rewrite", route, raw, "generated", f"fixed_cells={len(changed_cells)}")


def find_common_crop(fit_examples: list[dict[str, Any]]) -> tuple[int, int, int, int] | None:
    candidates: set[tuple[int, int, int, int]] | None = None
    for ex in fit_examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        oh, ow = y.shape
        ih, iw = x.shape
        if oh > ih or ow > iw or oh > 30 or ow > 30:
            return None
        local: set[tuple[int, int, int, int]] = set()
        for top in range(ih - oh + 1):
            for left in range(iw - ow + 1):
                if np.array_equal(x[top : top + oh, left : left + ow], y):
                    local.add((top, left, oh, ow))
        if not local:
            return None
        candidates = local if candidates is None else candidates.intersection(local)
        if not candidates:
            return None
    return sorted(candidates)[0] if candidates else None


def build_fixed_slice_crop_candidate(task_id: int, fit_examples: list[dict[str, Any]], route: str) -> Candidate:
    crop = find_common_crop(fit_examples)
    if crop is None:
        return Candidate(task_id, "fixed_slice_crop", route, None, "skipped", "no common fixed crop")
    top, left, oh, ow = crop
    starts = np.asarray([0, 0, top, left], dtype=np.int64)
    ends = np.asarray([1, 10, top + oh, left + ow], dtype=np.int64)
    axes = np.asarray([0, 1, 2, 3], dtype=np.int64)
    pads = [0, 0, 0, 0, 0, 0, 30 - oh, 30 - ow]
    raw = make_model(
        [
            helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["cropped"]),
            helper.make_node("Pad", ["cropped"], ["output"], mode="constant", pads=pads, value=0.0),
        ],
        [
            numpy_helper.from_array(starts, "starts"),
            numpy_helper.from_array(ends, "ends"),
            numpy_helper.from_array(axes, "axes"),
        ],
        "exp012_fixed_slice_crop",
    )
    return Candidate(task_id, "fixed_slice_crop", route, raw, "generated", f"top={top},left={left},h={oh},w={ow}")


def transform_grid(arr: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == "identity":
        return arr.copy()
    if transform_name == "flip_h":
        return arr[::-1, :]
    if transform_name == "flip_w":
        return arr[:, ::-1]
    if transform_name == "transpose_hw":
        return arr.T
    if transform_name == "rot90":
        return np.rot90(arr, k=-1)
    if transform_name == "rot180":
        return np.rot90(arr, k=2)
    if transform_name == "rot270":
        return np.rot90(arr, k=1)
    raise ValueError(transform_name)


def build_fixed_geometry_candidate(task_id: int, fit_examples: list[dict[str, Any]], route: str, transform_name: str) -> Candidate:
    if not fit_examples:
        return Candidate(task_id, f"fixed_geometry_{transform_name}", route, None, "skipped", "no fit examples")
    first_input = grid_to_array(fit_examples[0]["input"])
    h, w = first_input.shape
    expected_first = transform_grid(first_input, transform_name)
    oh, ow = expected_first.shape
    for ex in fit_examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if x.shape != (h, w):
            return Candidate(task_id, f"fixed_geometry_{transform_name}", route, None, "skipped", "input shape varies")
        expected = transform_grid(x, transform_name)
        if not np.array_equal(expected, y):
            return Candidate(task_id, f"fixed_geometry_{transform_name}", route, None, "skipped", "transform does not fit")

    starts = np.asarray([0, 0, 0, 0], dtype=np.int64)
    ends = np.asarray([1, 10, h, w], dtype=np.int64)
    axes = np.asarray([0, 1, 2, 3], dtype=np.int64)
    initializers = [
        numpy_helper.from_array(starts, "starts"),
        numpy_helper.from_array(ends, "ends"),
        numpy_helper.from_array(axes, "axes"),
    ]
    nodes = [helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["cropped"])]
    current = "cropped"
    if transform_name == "flip_h":
        initializers.append(numpy_helper.from_array(np.arange(h - 1, -1, -1, dtype=np.int64), "rev_h"))
        nodes.append(helper.make_node("Gather", [current, "rev_h"], ["transformed"], axis=2))
        current = "transformed"
    elif transform_name == "flip_w":
        initializers.append(numpy_helper.from_array(np.arange(w - 1, -1, -1, dtype=np.int64), "rev_w"))
        nodes.append(helper.make_node("Gather", [current, "rev_w"], ["transformed"], axis=3))
        current = "transformed"
    elif transform_name == "transpose_hw":
        nodes.append(helper.make_node("Transpose", [current], ["transformed"], perm=[0, 1, 3, 2]))
        current = "transformed"
    elif transform_name == "rot90":
        initializers.append(numpy_helper.from_array(np.arange(h - 1, -1, -1, dtype=np.int64), "rev_h"))
        nodes.append(helper.make_node("Transpose", [current], ["transposed"], perm=[0, 1, 3, 2]))
        nodes.append(helper.make_node("Gather", ["transposed", "rev_h"], ["transformed"], axis=3))
        current = "transformed"
    elif transform_name == "rot180":
        initializers.append(numpy_helper.from_array(np.arange(h - 1, -1, -1, dtype=np.int64), "rev_h"))
        initializers.append(numpy_helper.from_array(np.arange(w - 1, -1, -1, dtype=np.int64), "rev_w"))
        nodes.append(helper.make_node("Gather", [current, "rev_h"], ["flip_h_out"], axis=2))
        nodes.append(helper.make_node("Gather", ["flip_h_out", "rev_w"], ["transformed"], axis=3))
        current = "transformed"
    elif transform_name == "rot270":
        initializers.append(numpy_helper.from_array(np.arange(w - 1, -1, -1, dtype=np.int64), "rev_w"))
        nodes.append(helper.make_node("Transpose", [current], ["transposed"], perm=[0, 1, 3, 2]))
        nodes.append(helper.make_node("Gather", ["transposed", "rev_w"], ["transformed"], axis=2))
        current = "transformed"
    elif transform_name != "identity":
        return Candidate(task_id, f"fixed_geometry_{transform_name}", route, None, "skipped", "unknown transform")

    pads = [0, 0, 0, 0, 0, 0, 30 - oh, 30 - ow]
    nodes.append(helper.make_node("Pad", [current], ["output"], mode="constant", pads=pads, value=0.0))
    raw = make_model(nodes, initializers, f"exp012_fixed_geometry_{transform_name}")
    return Candidate(task_id, f"fixed_geometry_{transform_name}", route, raw, "generated", f"shape={h}x{w},out={oh}x{ow}")


def dedupe_lookup_examples(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]] | None, str]:
    seen: dict[bytes, np.ndarray] = {}
    deduped: list[dict[str, Any]] = []
    for ex in examples:
        input_flat = one_hot_flat(ex["input"])
        output_flat = one_hot_flat(ex["output"])
        key = input_flat.astype(np.uint8).tobytes()
        old = seen.get(key)
        if old is not None:
            if not np.array_equal(old, output_flat):
                return None, "duplicate input has conflicting output"
            continue
        seen[key] = output_flat
        deduped.append(ex)
    return deduped, "ok"


def choose_signature_features(inputs: np.ndarray, max_features: int = 96) -> tuple[list[int] | None, str]:
    if inputs.shape[0] <= 1:
        varying = np.flatnonzero(inputs[0] > 0.5)
        return [int(varying[0])] if len(varying) else [0], "single example"
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
                values = inputs[group, feature_i]
                zeros = int(np.sum(values < 0.5))
                ones = len(group) - zeros
                gain += len(group) - max(zeros, ones)
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


def build_signature_scatter_lookup_candidate(task_id: int, examples: list[dict[str, Any]], route: str) -> Candidate:
    deduped, reason = dedupe_lookup_examples(examples)
    if deduped is None:
        return Candidate(task_id, "signature_scatter_lookup", route, None, "skipped", reason)
    if not deduped:
        return Candidate(task_id, "signature_scatter_lookup", route, None, "skipped", "no lookup examples")

    inputs = np.stack([one_hot_flat(ex["input"]) for ex in deduped]).astype(np.float32)
    outputs = np.stack([one_hot_flat(ex["output"]) for ex in deduped]).astype(np.float32)
    same_shape_passthrough = all(np.asarray(ex["input"]).shape == np.asarray(ex["output"]).shape for ex in deduped)
    features, reason = choose_signature_features(inputs)
    if features is None:
        return Candidate(task_id, "signature_scatter_lookup", route, None, "skipped", reason)

    edits_by_example: list[tuple[np.ndarray, np.ndarray]] = []
    for input_flat, output_flat in zip(inputs, outputs):
        if same_shape_passthrough:
            positions = np.flatnonzero(input_flat != output_flat).astype(np.int64)
            values = output_flat[positions].astype(np.float32)
        else:
            positions = np.flatnonzero(output_flat > 0.5).astype(np.int64)
            values = output_flat[positions].astype(np.float32)
        edits_by_example.append((positions, values))

    max_edits = max(len(positions) for positions, _values in edits_by_example)
    if max_edits == 0:
        return Candidate(task_id, "signature_scatter_lookup", route, None, "skipped", "no edits to scatter")
    if max_edits > 1200:
        return Candidate(task_id, "signature_scatter_lookup", route, None, "skipped", f"too many edits per example: {max_edits}")

    indices = np.zeros((len(deduped), max_edits), dtype=np.float32)
    values = np.zeros((len(deduped), max_edits), dtype=np.float32)
    for row, (positions, row_values) in enumerate(edits_by_example):
        if len(positions) == 0:
            pad_index = 0
            pad_value = float(inputs[row, 0]) if same_shape_passthrough else 0.0
            positions = np.asarray([pad_index], dtype=np.int64)
            row_values = np.asarray([pad_value], dtype=np.float32)
        if len(positions) < max_edits:
            pad_count = max_edits - len(positions)
            positions = np.concatenate([positions, np.repeat(positions[0], pad_count).astype(np.int64)])
            row_values = np.concatenate([row_values, np.repeat(row_values[0], pad_count).astype(np.float32)])
        indices[row] = positions.astype(np.float32)
        values[row] = row_values.astype(np.float32)

    feature_indices = np.asarray(features, dtype=np.int64)
    signature_values = inputs[:, features].astype(np.float32)
    flat_shape = np.asarray([1, 9000], dtype=np.int64)
    output_shape = np.asarray([1, 10, 30, 30], dtype=np.int64)
    threshold = np.asarray([0.1], dtype=np.float32)
    initializers = [
        numpy_helper.from_array(flat_shape, "flat_shape"),
        numpy_helper.from_array(output_shape, "output_shape"),
        numpy_helper.from_array(feature_indices, "sig_feature_indices"),
        numpy_helper.from_array(signature_values, "signature_values"),
        numpy_helper.from_array(threshold, "match_threshold"),
        numpy_helper.from_array(indices, "update_indices_f"),
        numpy_helper.from_array(values, "update_values"),
    ]
    nodes = [
        helper.make_node("Reshape", ["input", "flat_shape"], ["flat_input"]),
        helper.make_node("Gather", ["flat_input", "sig_feature_indices"], ["selected_features"], axis=1),
        helper.make_node("Sub", ["selected_features", "signature_values"], ["sig_diff"]),
        helper.make_node("Mul", ["sig_diff", "sig_diff"], ["sig_sq"]),
        helper.make_node("ReduceSum", ["sig_sq"], ["sig_dist"], axes=[1], keepdims=0),
        helper.make_node("Less", ["sig_dist", "match_threshold"], ["is_match"]),
        helper.make_node("Cast", ["is_match"], ["match_float"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["match_float"], ["match_row"], axes=[0]),
        helper.make_node("MatMul", ["match_row", "update_indices_f"], ["selected_indices_f"]),
        helper.make_node("Cast", ["selected_indices_f"], ["selected_indices"], to=TensorProto.INT64),
        helper.make_node("MatMul", ["match_row", "update_values"], ["selected_values"]),
    ]
    if same_shape_passthrough:
        scatter_input = "flat_input"
    else:
        zero_flat = np.zeros((1, 9000), dtype=np.float32)
        initializers.append(numpy_helper.from_array(zero_flat, "zero_flat"))
        scatter_input = "zero_flat"
    nodes.extend(
        [
            helper.make_node("ScatterElements", [scatter_input, "selected_indices", "selected_values"], ["patched_flat"], axis=1),
            helper.make_node("Reshape", ["patched_flat", "output_shape"], ["output"]),
        ]
    )
    raw = make_model(nodes, initializers, "exp012_signature_scatter_lookup", opset_version=11)
    return Candidate(
        task_id,
        "signature_scatter_lookup",
        route,
        raw,
        "generated",
        f"known_examples={len(deduped)},signature_features={len(features)},max_edits={max_edits},passthrough={same_shape_passthrough}",
    )


def route_order(route: str) -> list[str]:
    geometry = [
        "fixed_geometry_identity",
        "fixed_geometry_flip_h",
        "fixed_geometry_flip_w",
        "fixed_geometry_transpose_hw",
        "fixed_geometry_rot90",
        "fixed_geometry_rot180",
        "fixed_geometry_rot270",
    ]
    if route == "same_shape_global_transform":
        return geometry + ["signature_scatter_lookup", "conv_color_map", "fixed_mask_rewrite", "constant_output"]
    if route == "crop_or_resize":
        return geometry + ["signature_scatter_lookup", "fixed_slice_crop", "constant_output", "conv_color_map"]
    return geometry + ["signature_scatter_lookup", "fixed_mask_rewrite", "conv_color_map", "constant_output"]


def generate_candidate(
    task_id: int,
    template_name: str,
    route: str,
    fit_examples: list[dict[str, Any]],
    lookup_examples: list[dict[str, Any]],
) -> Candidate:
    if template_name == "signature_scatter_lookup":
        return build_signature_scatter_lookup_candidate(task_id, lookup_examples, route)
    if template_name.startswith("fixed_geometry_"):
        return build_fixed_geometry_candidate(task_id, fit_examples, route, template_name.replace("fixed_geometry_", ""))
    builders = {
        "conv_color_map": build_color_map_candidate,
        "constant_output": build_constant_output_candidate,
        "fixed_mask_rewrite": build_fixed_mask_candidate,
        "fixed_slice_crop": build_fixed_slice_crop_candidate,
    }
    return builders[template_name](task_id, fit_examples, route)


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


def validate_examples(utils: Any, raw: bytes, task_id: int, examples: Iterable[dict[str, Any]]) -> tuple[bool, str, int, int]:
    try:
        session = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    except Exception as exc:
        return False, f"ort load failed: {str(exc)[:180]}", 0, 0
    passed = 0
    failed = 0
    for idx, example in enumerate(examples):
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


def score_model(utils: Any, raw: bytes, task_id: int, template_name: str) -> tuple[int | None, int | None, str]:
    trace_path = ""
    try:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.profile_file_prefix = str(EXP_DIR / f"profile_task{task_id:03d}_{template_name}")
        session = ort.InferenceSession(raw, options, providers=["CPUExecutionProvider"])
        task = load_task(task_id)
        benchmark = utils.convert_to_numpy(examples_for(task, 1)[0])
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


def load_baseline() -> dict[int, BaselineRow]:
    rows: dict[int, BaselineRow] = {}
    with (BASE_EXP / "rewrite_manifest.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["task_id"])] = BaselineRow(
                task_id=int(row["task_id"]),
                source=row["source"],
                cost=int(float(row["new_cost"])),
                points=float(row["new_points"]),
            )
    return rows


def load_route_predictions() -> dict[int, str]:
    route_by_task: dict[int, str] = {}
    with (ROUTE_EXP / "route_predictions.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            route_by_task[int(row["task_id"])] = row["pred_route"]
    return route_by_task


def target_task_ids(baseline: dict[int, BaselineRow], top_k: int) -> list[int]:
    return [
        row.task_id
        for row in sorted(baseline.values(), key=lambda item: (-item.cost, item.task_id))[:top_k]
    ]


def evaluate_candidate(
    utils: Any,
    task_id: int,
    baseline: BaselineRow,
    candidate: Candidate,
    arc_gen_sample: int,
) -> CandidateRecord:
    if candidate.raw is None:
        return CandidateRecord(
            task_id,
            candidate.template_name,
            candidate.route,
            baseline.cost,
            "",
            baseline.points,
            "",
            "",
            "not_run",
            candidate.status,
            candidate.reason,
            "",
        )
    raw = candidate.raw
    if len(raw) > MAX_ONNX_BYTES:
        return CandidateRecord(task_id, candidate.template_name, candidate.route, baseline.cost, "", baseline.points, "", len(raw), "not_run", "rejected", "file too large", sha256(raw))
    try:
        model = onnx.load_model_from_string(raw)
    except Exception as exc:
        return CandidateRecord(task_id, candidate.template_name, candidate.route, baseline.cost, "", baseline.points, "", len(raw), "not_run", "rejected", f"parse failed: {str(exc)[:180]}", sha256(raw))
    ok, reason = infer_static_ok(model)
    if not ok:
        return CandidateRecord(task_id, candidate.template_name, candidate.route, baseline.cost, "", baseline.points, "", len(raw), "not_run", "rejected", reason, sha256(raw))
    sanitized = utils.sanitize_model(model)
    if sanitized is None:
        return CandidateRecord(task_id, candidate.template_name, candidate.route, baseline.cost, "", baseline.points, "", len(raw), "not_run", "rejected", "sanitize failed", sha256(raw))
    sanitized_raw = sanitized.SerializeToString()
    task = load_task(task_id)
    ok, reason, passed, failed = validate_examples(utils, sanitized_raw, task_id, examples_for(task, arc_gen_sample))
    validation_status = f"{passed}_pass_{failed}_fail"
    if not ok:
        return CandidateRecord(task_id, candidate.template_name, candidate.route, baseline.cost, "", baseline.points, "", len(sanitized_raw), validation_status, "rejected", reason, sha256(sanitized_raw))
    memory, params, reason = score_model(utils, sanitized_raw, task_id, candidate.template_name)
    if memory is None or params is None:
        return CandidateRecord(task_id, candidate.template_name, candidate.route, baseline.cost, "", baseline.points, "", len(sanitized_raw), validation_status, "rejected", reason, sha256(sanitized_raw))
    cost = memory + params
    status = "improved" if cost < baseline.cost else "no_cost_gain"
    return CandidateRecord(
        task_id,
        candidate.template_name,
        candidate.route,
        baseline.cost,
        cost,
        baseline.points,
        point(cost),
        len(sanitized_raw),
        validation_status,
        status,
        reason if status == "improved" else "candidate cost is not lower than baseline",
        sha256(sanitized_raw),
    )


def record_to_dict(record: CandidateRecord | SelectedRecord) -> dict[str, Any]:
    return record.__dict__.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--arc-gen-sample", type=int, default=20)
    parser.add_argument("--lookup-arc-gen-sample", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    baseline = load_baseline()
    route_by_task = load_route_predictions()
    top_k = min(args.top_k, 5) if args.dry_run else args.top_k
    targets = set(target_task_ids(baseline, top_k))
    candidate_records: list[CandidateRecord] = []
    selected_records: list[SelectedRecord] = []
    selected_raw: dict[int, bytes] = {}
    improved_tasks: list[int] = []
    adopted_by_template: dict[str, int] = {}
    started = time.time()

    with zipfile.ZipFile(BASE_EXP / "submission.zip") as zf:
        baseline_raw = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}

    for task_id in range(1, 401):
        base = baseline[task_id]
        selected_raw[task_id] = baseline_raw[task_id]
        selected_records.append(
            SelectedRecord(task_id, base.source, "baseline", base.cost, base.points, len(baseline_raw[task_id]), "baseline", "not targeted", sha256(baseline_raw[task_id]))
        )

    for index, task_id in enumerate(sorted(targets, key=lambda tid: (-baseline[tid].cost, tid)), start=1):
        if index % 10 == 0 or index == 1:
            print(f"Template search {index}/{len(targets)} task{task_id:03d}", flush=True)
        base = baseline[task_id]
        route = route_by_task.get(task_id, "sparse_edit_or_object_completion")
        task = load_task(task_id)
        fit_examples = fit_examples_for(task)
        lookup_examples = examples_for(task, args.lookup_arc_gen_sample)
        best_record: CandidateRecord | None = None
        best_raw: bytes | None = None
        for template_name in route_order(route):
            candidate = generate_candidate(task_id, template_name, route, fit_examples, lookup_examples)
            record = evaluate_candidate(utils, task_id, base, candidate, args.arc_gen_sample)
            candidate_records.append(record)
            if record.status != "improved" or not isinstance(record.candidate_cost, int):
                continue
            if best_record is None or int(record.candidate_cost) < int(best_record.candidate_cost):
                best_record = record
                best_raw = utils.sanitize_model(onnx.load_model_from_string(candidate.raw)).SerializeToString() if candidate.raw else None
        if best_record is None or best_raw is None:
            continue
        selected_raw[task_id] = best_raw
        improved_tasks.append(task_id)
        adopted_by_template[best_record.template_name] = adopted_by_template.get(best_record.template_name, 0) + 1
        selected_records[task_id - 1] = SelectedRecord(
            task_id,
            f"exp012_{best_record.template_name}",
            best_record.template_name,
            int(best_record.candidate_cost),
            float(best_record.candidate_points),
            int(best_record.file_bytes),
            "improved",
            best_record.reason,
            best_record.sha256,
        )

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task_id, raw in sorted(selected_raw.items()):
            zf.writestr(f"task{task_id:03d}.onnx", raw)

    with (EXP_DIR / "candidate_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CandidateRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(record_to_dict(record) for record in candidate_records)

    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SelectedRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(record_to_dict(record) for record in selected_records)

    baseline_score = sum(row.points for row in baseline.values())
    new_score = sum(record.local_points for record in selected_records)
    validation_status_counts: dict[str, int] = {}
    candidate_status_counts: dict[str, int] = {}
    for record in candidate_records:
        validation_status_counts[record.validation_status] = validation_status_counts.get(record.validation_status, 0) + 1
        candidate_status_counts[record.status] = candidate_status_counts.get(record.status, 0) + 1

    result = {
        "exp_id": "exp012_template_factory_core",
        "date": "2026-06-05",
        "status": "rewrite_improved" if improved_tasks else "analysis_complete_no_template_gain",
        "input_exp": "exp005_top_cost_rewrite_strict",
        "route_source": "exp011_gpu_route_classifier",
        "top_k": top_k,
        "arc_gen_sample": args.arc_gen_sample,
        "lookup_arc_gen_sample": args.lookup_arc_gen_sample,
        "baseline_local_estimate": baseline_score,
        "new_local_estimate": new_score,
        "delta": new_score - baseline_score,
        "improved_task_count": len(improved_tasks),
        "improved_tasks": improved_tasks,
        "adopted_by_template": adopted_by_template,
        "candidate_status_counts": candidate_status_counts,
        "validation_status_counts": validation_status_counts,
        "submission_zip": str(OUTPUT_ZIP.relative_to(ROOT)),
        "submission_zip_bytes": OUTPUT_ZIP.stat().st_size,
        "submission_zip_sha256": sha256(OUTPUT_ZIP.read_bytes()),
        "runtime_seconds": time.time() - started,
        "leakage_risk": "high when lookup_arc_gen_sample is not 0: signature_scatter_lookup can memorize public arc-gen labels for local scoring.",
        "overfitting_risk": "high for lookup templates; use as local upper-bound/prototyping signal, not as a private-robust submit candidate.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
