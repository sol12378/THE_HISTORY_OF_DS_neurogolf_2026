"""Minimal ONNX emitters for NeuroGolf IR programs."""

from __future__ import annotations

import numpy as np
from onnx import helper, numpy_helper

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
    except Exception as exc:
        return Candidate(task_id, template_name, "farm_ir", None, "skipped", f"emit failed: {str(exc)[:180]}")
    return Candidate(task_id, template_name, "farm_ir", None, "skipped", f"unsupported IR node {node.kind}/{node.op_type}")
