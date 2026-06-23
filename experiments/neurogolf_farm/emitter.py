"""Minimal ONNX emitters for NeuroGolf IR programs."""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from experiments.phase1_rewrite_utils import Candidate, make_model

from .ir import IRProgram, PrimitiveKind


def emit_candidate(program: IRProgram) -> Candidate:
    if not program.nodes:
        return Candidate(int(program.task_id), "empty_ir", "farm_ir", None, "skipped", "empty IR")
    node = program.nodes[0]
    task_id = int(program.task_id)
    template_name = f"farm_{program.family}_{node.kind.value}"
    try:
        if node.kind == PrimitiveKind.ONE_NODE_DATA_MOVEMENT and node.op_type == "Identity":
            raw = make_model([helper.make_node("Identity", ["input"], ["output"])], [], template_name)
            return Candidate(task_id, template_name, "farm_ir", raw, "generated", "identity")
        if node.kind == PrimitiveKind.CHANNEL_GATHER and node.op_type == "Gather":
            order = node.attrs.get("order", list(range(10)))
            idx = numpy_helper.from_array(np.asarray(order, dtype=np.int64), "idx")
            raw = make_model([helper.make_node("Gather", ["input", "idx"], ["output"], axis=1)], [idx], template_name)
            return Candidate(task_id, template_name, "farm_ir", raw, "generated", f"order={order}")
        if node.kind == PrimitiveKind.STATIC_SLICE_PAD and node.op_type == "SlicePad":
            starts = np.asarray(node.attrs.get("starts", [0, 0, 0, 0]), dtype=np.int64)
            ends = np.asarray(node.attrs.get("ends", [1, 10, 3, 3]), dtype=np.int64)
            axes = np.asarray(node.attrs.get("axes", [0, 1, 2, 3]), dtype=np.int64)
            pads = np.asarray(node.attrs.get("pads", [0, 0, 0, 0, 0, 0, 27, 27]), dtype=np.int64)
            zero = np.asarray(0.0, dtype=np.float32)
            initializers = [
                numpy_helper.from_array(starts, "starts"),
                numpy_helper.from_array(ends, "ends"),
                numpy_helper.from_array(axes, "axes"),
                numpy_helper.from_array(pads, "pads"),
                numpy_helper.from_array(zero, "zero"),
            ]
            nodes = [
                helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["crop"]),
                helper.make_node("Pad", ["crop", "pads", "zero"], ["output"], mode="constant"),
            ]
            raw = make_model(nodes, initializers, template_name, opset_version=11)
            return Candidate(task_id, template_name, "farm_ir", raw, "generated", "static slice+pad")
        # COMPUTED_SLICE_PAD: a slice+pad whose bounds were COMPUTED offline by the
        # inducer and baked static. (Truly input-dependent bounds are impossible here:
        # infer_static_ok rejects dynamic shapes, so the bounds must be constant.)
        if node.kind == PrimitiveKind.COMPUTED_SLICE_PAD and node.op_type == "SlicePad":
            starts = np.asarray(node.attrs["starts"], dtype=np.int64)
            ends = np.asarray(node.attrs["ends"], dtype=np.int64)
            axes = np.asarray(node.attrs.get("axes", [0, 1, 2, 3]), dtype=np.int64)
            pads = np.asarray(node.attrs["pads"], dtype=np.int64)
            inits = [
                numpy_helper.from_array(starts, "starts"),
                numpy_helper.from_array(ends, "ends"),
                numpy_helper.from_array(axes, "axes"),
                numpy_helper.from_array(pads, "pads"),
                numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "zero"),
            ]
            nodes = [
                helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["crop"]),
                helper.make_node("Pad", ["crop", "pads", "zero"], ["output"], mode="constant"),
            ]
            raw = make_model(nodes, inits, template_name, opset_version=11)
            return Candidate(task_id, template_name, "farm_ir", raw, "generated", "computed(offline) slice+pad")
        # RECOLOR_DIRECT: single channel Gather = a colour permutation/selection. One
        # node, no intermediates (output excluded from memory) -> cost ~= len(order).
        if node.kind == PrimitiveKind.RECOLOR_DIRECT and node.op_type == "Gather":
            order = np.asarray(node.attrs.get("order", list(range(10))), dtype=np.int64)
            idx = numpy_helper.from_array(order, "order")
            raw = make_model(
                [helper.make_node("Gather", ["input", "order"], ["output"], axis=1)], [idx], template_name)
            return Candidate(task_id, template_name, "farm_ir", raw, "generated", f"recolor order={order.tolist()}")
        # RECOLOR_CAST: Gather then Cast to a narrow output dtype. NOTE: this adds a
        # full-grid intermediate (the Gather output) which IS counted; RECOLOR_DIRECT is
        # cheaper. Kept for completeness / when a downstream needs the cast dtype.
        if node.kind == PrimitiveKind.RECOLOR_CAST and node.op_type in ("Gather", "Cast"):
            order = np.asarray(node.attrs.get("order", list(range(10))), dtype=np.int64)
            to_dtype = int(node.attrs.get("to", TensorProto.UINT8))
            idx = numpy_helper.from_array(order, "order")
            nodes = [
                helper.make_node("Gather", ["input", "order"], ["recolored"], axis=1),
                helper.make_node("Cast", ["recolored"], ["output"], to=to_dtype),
            ]
            raw = make_model(nodes, [idx], template_name)
            return Candidate(task_id, template_name, "farm_ir", raw, "generated", f"recolor+cast order={order.tolist()}")
        # SMALL_LOCAL_MASK: a Where over a (typically bounded) fixed mask -> a localized
        # constant edit. cond/fill are baked; one Where, no extra intermediates.
        if node.kind == PrimitiveKind.SMALL_LOCAL_MASK and node.op_type == "Where":
            cond = np.asarray(node.attrs["cond"], dtype=bool)
            fill = np.asarray(node.attrs["fill"], dtype=np.float32)
            inits = [numpy_helper.from_array(cond, "cond"), numpy_helper.from_array(fill, "fill")]
            raw = make_model(
                [helper.make_node("Where", ["cond", "fill", "input"], ["output"])], inits, template_name, opset_version=11)
            return Candidate(task_id, template_name, "farm_ir", raw, "generated", "small local mask edit")
        # GRID_SAMPLE: single fused coordinate sampler (opset 16). The genuinely new
        # capability vs phase1: any pixel permutation/geometric remap in ONE op, with no
        # intermediates -> cost ~= grid params. Validated feasible at cost ~1800 (exp324).
        if node.kind == PrimitiveKind.GRID_SAMPLE and node.op_type == "GridSample":
            grid = np.asarray(node.attrs["grid"], dtype=np.float32)  # (1, H_out, W_out, 2)
            g_init = numpy_helper.from_array(grid, "grid")
            gs = helper.make_node(
                "GridSample", ["input", "grid"], ["output"],
                mode=node.attrs.get("mode", "nearest"),
                padding_mode=node.attrs.get("padding_mode", "zeros"),
                align_corners=int(node.attrs.get("align_corners", 0)),
            )
            raw = make_model([gs], [g_init], template_name, opset_version=16)
            return Candidate(task_id, template_name, "farm_ir", raw, "generated", f"gridsample {grid.shape}")
    except Exception as exc:
        return Candidate(task_id, template_name, "farm_ir", None, "skipped", f"emit failed: {str(exc)[:180]}")
    return Candidate(task_id, template_name, "farm_ir", None, "skipped", f"unsupported IR node {node.kind}/{node.op_type}")
