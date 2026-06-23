"""Pre-emission cost guardrails for NeuroGolf ONNX candidates.

The current competition objective is parameter count plus output tensor bytes.
The extractor therefore reports numeric proxy fields in addition to the coarse
accept/reject band. The proxy is not a replacement for the official
score_network path; it is a cheap pre-emission screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Iterable

from .ir import IRProgram, PrimitiveKind


FULL_GRID_ELEMENTS = 1 * 10 * 30 * 30
HARD_BAD_OPS = {"Loop", "Scan", "NonZero", "Unique", "Compress"}
HIGH_RISK_OPS = {"MatMul", "Conv", "Tile", "Resize"}
LOW_COST_OPS = {"Identity", "Transpose", "Slice", "Gather", "GridSample", "Pad", "Cast", "Reshape", "Squeeze", "Unsqueeze"}
CONDITIONAL_HIGH_RISK_OPS = {"ScatterND", "GatherND", "ScatterElements"}
DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "bool": 1,
}
ONNX_DTYPE_BYTES = {
    1: 4,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 4,
    7: 8,
    9: 1,
    10: 2,
    11: 8,
    12: 4,
    13: 8,
    16: 2,
}


@dataclass
class CostSignal:
    subject: str
    predicted_cost_band: str
    hard_reject: bool
    reasons: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    primitive_kinds: tuple[str, ...] = ()
    node_count: int | None = None
    full_grid_intermediate_count: int | None = None
    param_count: int | None = None
    param_bytes: int | None = None
    memory_bytes_proxy: int | None = None
    cost_proxy: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "predicted_cost_band": self.predicted_cost_band,
            "hard_reject": self.hard_reject,
            "reasons": self.reasons,
            "risk_tags": self.risk_tags,
            "primitive_kinds": self.primitive_kinds,
            "node_count": self.node_count,
            "full_grid_intermediate_count": self.full_grid_intermediate_count,
            "param_count": self.param_count,
            "param_bytes": self.param_bytes,
            "memory_bytes_proxy": self.memory_bytes_proxy,
            "cost_proxy": self.cost_proxy,
        }


class CostExtractor:
    """Rejects candidate designs that historically explode NeuroGolf cost."""

    def assess_ir(self, program: IRProgram) -> CostSignal:
        reasons: list[str] = []
        risk_tags: list[str] = []
        hard_reject = False
        full_grid_intermediates = 0
        memory_bytes_proxy = 0
        param_count_proxy = 0

        for node in program.nodes:
            param_count_proxy += int(node.attrs.get("param_count", 0) or 0)
            if node.output and node.output.known_elements is not None:
                if node.output.name == "output":
                    memory_bytes_proxy += node.output.known_elements * DTYPE_BYTES.get(node.output.dtype, 4)
                elif node.output.known_elements >= FULL_GRID_ELEMENTS:
                    full_grid_intermediates += 1
            for tag in node.risk_tags:
                if tag not in risk_tags:
                    risk_tags.append(tag)
            if node.kind in {
                PrimitiveKind.FULL_GRID_COMPOSITION,
                PrimitiveKind.CONNECTIVITY_UNROLL,
                PrimitiveKind.SPARSE_WRITEBACK,
            }:
                # P2-1: size-conditional. These lanes only explode cost when they
                # materialize a full-grid output/intermediate. A small/bounded-region
                # version (e.g. sparse write into a small patch) is allowed; the
                # official score_network + full-arc gate still has the final say.
                out_elems = node.output.known_elements if node.output else None
                if out_elems is not None and out_elems >= FULL_GRID_ELEMENTS:
                    hard_reject = True
                    reasons.append(f"{node.name}: {node.kind.value} over full-grid output is pre-rejected")
                else:
                    risk_tags.append(f"conditional_kind:{node.kind.value}")
            if node.op_type in HARD_BAD_OPS:
                hard_reject = True
                reasons.append(f"{node.name}: {node.op_type} is pre-rejected")
            if node.op_type in CONDITIONAL_HIGH_RISK_OPS:
                if node.output and (node.output.known_elements or 0) >= FULL_GRID_ELEMENTS:
                    hard_reject = True
                    reasons.append(f"{node.name}: {node.op_type} over full-grid output is pre-rejected")
                else:
                    risk_tags.append(f"conditional_op:{node.op_type}")
            if node.op_type in HIGH_RISK_OPS:
                risk_tags.append(f"op:{node.op_type}")

        if full_grid_intermediates >= 2:
            hard_reject = True
            reasons.append(f"{full_grid_intermediates} full-grid intermediates before final output")

        cost_proxy = param_count_proxy + memory_bytes_proxy
        if hard_reject:
            band = "reject"
        elif cost_proxy <= 600 and all(node.op_type in LOW_COST_OPS for node in program.nodes):
            band = "250-600_plausible"
        elif cost_proxy <= 2000:
            band = "600-2000_probe"
        else:
            band = "high_cost_probe_only"

        return CostSignal(
            subject=program.task_id,
            predicted_cost_band=band,
            hard_reject=hard_reject,
            reasons=reasons,
            risk_tags=sorted(set(risk_tags)),
            primitive_kinds=program.primitive_kinds,
            node_count=len(program.nodes),
            full_grid_intermediate_count=full_grid_intermediates,
            param_count=param_count_proxy,
            param_bytes=param_count_proxy,
            memory_bytes_proxy=memory_bytes_proxy,
            cost_proxy=cost_proxy,
        )

    def assess_onnx_path(self, model_path: str | Path) -> CostSignal:
        path = Path(model_path)
        try:
            import onnx  # type: ignore
        except Exception as exc:
            return CostSignal(
                subject=str(path),
                predicted_cost_band="unknown_no_onnx",
                hard_reject=False,
                reasons=[f"onnx import unavailable: {exc}"],
            )

        model = onnx.load(str(path))
        node_types = [node.op_type for node in model.graph.node]
        hard_reject = False
        reasons: list[str] = []
        risk_tags: list[str] = []

        for op_type in sorted(set(node_types)):
            if op_type in HARD_BAD_OPS:
                hard_reject = True
                reasons.append(f"{op_type} is pre-rejected")
            if op_type in CONDITIONAL_HIGH_RISK_OPS:
                risk_tags.append(f"conditional_op:{op_type}")
            if op_type in HIGH_RISK_OPS:
                risk_tags.append(f"op:{op_type}")

        param_count, param_bytes = self._initializer_stats(model.graph.initializer)
        full_grid_intermediates, _ = self._value_info_stats(model.graph.value_info)
        _, memory_bytes_proxy = self._value_info_stats(model.graph.output)
        if full_grid_intermediates >= 2:
            hard_reject = True
            reasons.append(f"{full_grid_intermediates} full-grid intermediates before final output")
        if any(op in CONDITIONAL_HIGH_RISK_OPS for op in node_types) and full_grid_intermediates > 0:
            risk_tags.append("conditional_full_grid_data_movement")

        cost_proxy = param_count + memory_bytes_proxy
        if hard_reject:
            band = "reject"
        elif cost_proxy <= 600 and len(node_types) <= 6:
            band = "250-600_plausible"
        elif cost_proxy <= 2000:
            band = "600-2000_probe"
        else:
            band = "high_cost_probe_only"

        return CostSignal(
            subject=str(path),
            predicted_cost_band=band,
            hard_reject=hard_reject,
            reasons=reasons,
            risk_tags=sorted(set(risk_tags)),
            node_count=len(node_types),
            full_grid_intermediate_count=full_grid_intermediates,
            param_count=param_count,
            param_bytes=param_bytes,
            memory_bytes_proxy=memory_bytes_proxy,
            cost_proxy=cost_proxy,
        )

    @staticmethod
    def _value_info_stats(value_infos: Iterable[object]) -> tuple[int, int]:
        full_grid_count = 0
        memory_bytes = 0
        for value_info in value_infos:
            tensor_type = getattr(getattr(value_info, "type", None), "tensor_type", None)
            elem_type = getattr(tensor_type, "elem_type", 1)
            shape = getattr(tensor_type, "shape", None)
            dims = getattr(shape, "dim", [])
            product = 1
            known = True
            for dim in dims:
                dim_value = getattr(dim, "dim_value", 0)
                if not dim_value:
                    known = False
                    break
                product *= int(dim_value)
            if known and product >= FULL_GRID_ELEMENTS:
                full_grid_count += 1
            if known:
                memory_bytes += product * ONNX_DTYPE_BYTES.get(int(elem_type), 4)
        return full_grid_count, memory_bytes

    @staticmethod
    def _initializer_stats(initializers: Iterable[object]) -> tuple[int, int]:
        param_count = 0
        param_bytes = 0
        for tensor in initializers:
            dims = list(getattr(tensor, "dims", []))
            n = math.prod(dims) if dims else 1
            param_count += int(n)
            param_bytes += int(n) * ONNX_DTYPE_BYTES.get(int(getattr(tensor, "data_type", 1)), 4)
        return param_count, param_bytes
