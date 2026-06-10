"""NeuroGolf-specific intermediate representation.

The IR is intentionally small. Its job is not to replace ONNX; it is a
cost-aware checkpoint before emitting ONNX candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PrimitiveKind(str, Enum):
    ONE_NODE_DATA_MOVEMENT = "one_node_data_movement"
    CHANNEL_GATHER = "channel_gather"
    GRID_SAMPLE = "grid_sample"
    RECOLOR_DIRECT = "recolor_direct"
    RECOLOR_CAST = "recolor_cast"
    STATIC_SLICE_PAD = "static_slice_pad"
    COMPUTED_SLICE_PAD = "computed_slice_pad"
    SMALL_LOCAL_MASK = "small_local_mask"
    GRAPH_SURGERY = "graph_surgery"
    FULL_GRID_COMPOSITION = "full_grid_composition"
    CONNECTIVITY_UNROLL = "connectivity_unroll"
    SPARSE_WRITEBACK = "sparse_writeback"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int | str | None, ...]
    dtype: str = "float32"

    @property
    def known_elements(self) -> int | None:
        product = 1
        for dim in self.shape:
            if not isinstance(dim, int):
                return None
            product *= dim
        return product


@dataclass(frozen=True)
class IRNode:
    name: str
    kind: PrimitiveKind
    op_type: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    output: TensorSpec | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    risk_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class IRProgram:
    task_id: str
    family: str
    intent: str
    nodes: tuple[IRNode, ...]
    source: str = "manual"
    claimed_local_delta: float | None = None
    claimed_public_lb: float | None = None
    notes: str = ""

    @property
    def primitive_kinds(self) -> tuple[str, ...]:
        return tuple(node.kind.value for node in self.nodes)

    @property
    def op_types(self) -> tuple[str, ...]:
        return tuple(node.op_type for node in self.nodes)


def recolor_direct_program(
    task_id: str,
    *,
    output_shape: tuple[int | str | None, ...],
    output_dtype: str = "uint8",
    param_count: int = 44,
    source: str = "recolor_supplier",
) -> IRProgram:
    return IRProgram(
        task_id=task_id,
        family="recolor",
        intent="direct channel gather recolor without dtype cast",
        source=source,
        nodes=(
            IRNode(
                name="recolor_direct",
                kind=PrimitiveKind.RECOLOR_DIRECT,
                op_type="Gather",
                output=TensorSpec("output", output_shape, output_dtype),
                attrs={"param_count": param_count},
            ),
        ),
    )


def recolor_cast_program(
    task_id: str,
    *,
    output_shape: tuple[int | str | None, ...],
    output_dtype: str = "uint8",
    direct_param_count: int = 44,
    cast_param_count: int = 96,
    source: str = "recolor_supplier",
) -> IRProgram:
    return IRProgram(
        task_id=task_id,
        family="recolor",
        intent="channel gather recolor with explicit cast fallback",
        source=source,
        nodes=(
            IRNode(
                name="recolor_gather",
                kind=PrimitiveKind.RECOLOR_CAST,
                op_type="Gather",
                output=TensorSpec("recolor_tmp", output_shape, output_dtype),
                attrs={"param_count": direct_param_count},
            ),
            IRNode(
                name="recolor_cast",
                kind=PrimitiveKind.RECOLOR_CAST,
                op_type="Cast",
                output=TensorSpec("output", output_shape, output_dtype),
                attrs={"param_count": cast_param_count},
            ),
        ),
    )


def gridsample_program(
    task_id: str,
    *,
    output_shape: tuple[int | str | None, ...],
    output_dtype: str = "uint8",
    param_count: int = 8,
    source: str = "gridsample_supplier",
) -> IRProgram:
    return IRProgram(
        task_id=task_id,
        family="one_node_gridsample",
        intent="fused coordinate sampling with uint8 output",
        source=source,
        nodes=(
            IRNode(
                name="grid_sample",
                kind=PrimitiveKind.GRID_SAMPLE,
                op_type="GridSample",
                output=TensorSpec("output", output_shape, output_dtype),
                attrs={"param_count": param_count},
            ),
        ),
    )
