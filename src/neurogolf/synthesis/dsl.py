from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np


Grid = np.ndarray
PrimitiveFn = Callable[[Grid, Mapping[str, object]], Grid]


@dataclass(frozen=True)
class PrimitiveSpec:
    """A small, auditable primitive that can later be lowered to static ONNX."""

    name: str
    family: str
    description: str
    onnx_lowering: str
    leakage_risk: str = "low"
    private_risk: str = "unknown"


@dataclass(frozen=True)
class ProgramStep:
    primitive: str
    params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Program:
    steps: tuple[ProgramStep, ...]

    def run(self, grid: Grid, registry: Mapping[str, PrimitiveFn]) -> Grid:
        out = np.asarray(grid, dtype=np.int64)
        for step in self.steps:
            if step.primitive not in registry:
                raise KeyError(f"unknown primitive: {step.primitive}")
            out = registry[step.primitive](out, step.params)
        return out


def identity(grid: Grid, params: Mapping[str, object]) -> Grid:
    return np.asarray(grid, dtype=np.int64).copy()


def apply_color_map(grid: Grid, params: Mapping[str, object]) -> Grid:
    mapping = {int(k): int(v) for k, v in dict(params.get("mapping", {})).items()}
    out = np.asarray(grid, dtype=np.int64).copy()
    for src, dst in mapping.items():
        out[grid == src] = dst
    return out


def fixed_crop(grid: Grid, params: Mapping[str, object]) -> Grid:
    r0 = int(params["r0"])
    c0 = int(params["c0"])
    height = int(params["height"])
    width = int(params["width"])
    return np.asarray(grid, dtype=np.int64)[r0 : r0 + height, c0 : c0 + width].copy()


def replace_background(grid: Grid, params: Mapping[str, object]) -> Grid:
    background = int(params.get("background", 0))
    color = int(params["color"])
    out = np.asarray(grid, dtype=np.int64).copy()
    out[out == background] = color
    return out


def infer_color_map(pairs: list[tuple[Grid, Grid]]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for x, y in pairs:
        if x.shape != y.shape:
            return None
        for src, dst in zip(x.reshape(-1), y.reshape(-1)):
            src_i = int(src)
            dst_i = int(dst)
            if src_i in mapping and mapping[src_i] != dst_i:
                return None
            mapping[src_i] = dst_i
    return mapping


def default_registry() -> dict[str, PrimitiveFn]:
    return {
        "identity": identity,
        "color_map": apply_color_map,
        "fixed_crop": fixed_crop,
        "replace_background": replace_background,
    }


PRIMITIVE_CATALOG: tuple[PrimitiveSpec, ...] = (
    PrimitiveSpec(
        name="identity",
        family="same_shape_global_transform",
        description="Return the input unchanged; useful as a baseline and pipeline sentinel.",
        onnx_lowering="Identity",
        private_risk="low",
    ),
    PrimitiveSpec(
        name="color_map",
        family="same_shape_global_transform",
        description="Apply a constant per-color remapping learned from training pairs.",
        onnx_lowering="Equal + Where per mapped color",
        private_risk="medium",
    ),
    PrimitiveSpec(
        name="fixed_crop",
        family="crop_resize",
        description="Extract a fixed rectangle shared by all training examples.",
        onnx_lowering="Slice with constant starts/ends",
        private_risk="medium",
    ),
    PrimitiveSpec(
        name="signature_lookup",
        family="lookup",
        description="Memorize known input signatures and scatter a fixed output.",
        onnx_lowering="MatMul/Gather/ScatterND style lookup",
        leakage_risk="high",
        private_risk="high",
    ),
    PrimitiveSpec(
        name="boundary_flood_fill",
        family="region_partition_fill",
        description="Fill background components by boundary reachability using unrolled dilation.",
        onnx_lowering="Conv + Where unrolled for a fixed grid radius",
        private_risk="medium",
    ),
    PrimitiveSpec(
        name="rectangular_room_fill",
        family="region_partition_fill",
        description="Detect line-grid rooms and recolor enclosed or exterior cells.",
        onnx_lowering="Equal + Reduce/Conv + Where with static iterations",
        private_risk="medium",
    ),
    PrimitiveSpec(
        name="point_to_line_pattern",
        family="sparse_edit_or_object_completion",
        description="Expand sparse seed points into lines, rays, boxes, or repeated motifs.",
        onnx_lowering="Constant kernels + Conv/Gather/Where",
        private_risk="medium",
    ),
)
