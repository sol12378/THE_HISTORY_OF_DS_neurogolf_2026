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
