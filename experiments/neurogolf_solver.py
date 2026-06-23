"""Layer-A general solver (P3-1): a small param-driven DSL that is (a) numpy-evaluable
for deterministic verification on the FULL arc-gen set, and (b) lowerable to compact
static ONNX for the fixed-input-shape case. Programs come from an enumerative search
AND/OR an LLM proposer (neurogolf_llm_proposer). Every lowered program is adopted only
through phase1.evaluate_candidate (static gate + full-arc + cost < union).

DSL op = (name, params). A Program = tuple[op, ...] applied left-to-right.
Atoms: geometric(kind) / recolor(map) / crop(top,left,h,w) / tile(ry,rx).
"""
from __future__ import annotations

import itertools
from typing import Any, Callable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import phase1_rewrite_utils as P

GEO_KINDS = ("identity", "flip_h", "flip_v", "rot90", "rot180", "rot270", "transpose", "anti_transpose")


# ------------------------- numpy DSL (verification) -------------------------
def _geo(grid: np.ndarray, kind: str) -> np.ndarray:
    if kind == "identity":
        return grid
    if kind == "flip_h":
        return grid[:, ::-1]
    if kind == "flip_v":
        return grid[::-1, :]
    if kind == "rot90":
        return np.rot90(grid, 1)
    if kind == "rot180":
        return np.rot90(grid, 2)
    if kind == "rot270":
        return np.rot90(grid, 3)
    if kind == "transpose":
        return grid.T
    if kind == "anti_transpose":
        return grid[::-1, ::-1].T
    raise ValueError(kind)


def apply_op(grid: np.ndarray, op: tuple[str, dict]) -> np.ndarray:
    name, params = op
    if name == "geometric":
        return np.ascontiguousarray(_geo(grid, params["kind"]))
    if name == "recolor":
        mapping = {int(k): int(v) for k, v in params["map"].items()}
        out = grid.copy()
        for src, dst in mapping.items():
            out[grid == src] = dst
        return out
    if name == "crop":
        t, l, h, w = params["top"], params["left"], params["h"], params["w"]
        return np.ascontiguousarray(grid[t:t + h, l:l + w])
    if name == "tile":
        return np.tile(grid, (params["ry"], params["rx"]))
    raise ValueError(name)


def apply_program(grid: np.ndarray, program: tuple) -> np.ndarray | None:
    g = grid
    for op in program:
        try:
            g = apply_op(g, op)
        except Exception:
            return None
        if g.ndim != 2 or g.size == 0 or max(g.shape) > 30:
            return None
    return g


def verify_program(examples: list[dict[str, Any]], program: tuple) -> bool:
    for ex in examples:
        x = P.grid_to_array(ex["input"])
        y = P.grid_to_array(ex["output"])
        pred = apply_program(x, program)
        if pred is None or pred.shape != y.shape or not np.array_equal(pred, y):
            return False
    return True


# ------------------------- ONNX lowering (fixed input shape) -------------------------
def _gather(axis: int, idx: list[int], in_name: str, out_name: str, prefix: str):
    init = numpy_helper.from_array(np.asarray(idx, dtype=np.int64), f"{prefix}_idx")
    node = helper.make_node("Gather", [in_name, f"{prefix}_idx"], [out_name], axis=axis)
    return [node], [init]


def lower_program(program: tuple, ih: int, iw: int) -> bytes | None:
    """Compose the program into a static ONNX over (1,10,30,30). Requires a single
    input shape (ih,iw) (variable-shape tasks are not statically lowerable). Works on
    the actual (1,10,h,w) tensor, tracking shape, padding to 30x30 at the end. Returns
    None if any op is not statically lowerable for this shape."""
    nodes: list = []
    inits: list = []
    # crop the padded input down to its active (ih,iw)
    inits += [
        numpy_helper.from_array(np.asarray([0, 0, 0, 0], dtype=np.int64), "in_starts"),
        numpy_helper.from_array(np.asarray([1, 10, ih, iw], dtype=np.int64), "in_ends"),
        numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "in_axes"),
    ]
    nodes.append(helper.make_node("Slice", ["input", "in_starts", "in_ends", "in_axes"], ["t0"]))
    cur, h, w, k = "t0", ih, iw, 0
    for op in program:
        name, params = op
        nxt = f"t{k + 1}"
        if name == "geometric":
            kind = params["kind"]
            if kind == "identity":
                nodes.append(helper.make_node("Identity", [cur], [nxt]));
            elif kind == "flip_h":
                n, i = _gather(3, list(range(w - 1, -1, -1)), cur, nxt, f"p{k}"); nodes += n; inits += i
            elif kind == "flip_v":
                n, i = _gather(2, list(range(h - 1, -1, -1)), cur, nxt, f"p{k}"); nodes += n; inits += i
            elif kind == "rot180":
                n, i = _gather(3, list(range(w - 1, -1, -1)), cur, f"{nxt}_a", f"p{k}h"); nodes += n; inits += i
                n, i = _gather(2, list(range(h - 1, -1, -1)), f"{nxt}_a", nxt, f"p{k}v"); nodes += n; inits += i
            elif kind == "transpose":
                nodes.append(helper.make_node("Transpose", [cur], [nxt], perm=[0, 1, 3, 2])); h, w = w, h
            elif kind == "rot90":  # ccw = transpose(flip_h)
                n, i = _gather(3, list(range(w - 1, -1, -1)), cur, f"{nxt}_a", f"p{k}h"); nodes += n; inits += i
                nodes.append(helper.make_node("Transpose", [f"{nxt}_a", ], [nxt], perm=[0, 1, 3, 2])); h, w = w, h
            elif kind == "rot270":  # cw = transpose(flip_v)
                n, i = _gather(2, list(range(h - 1, -1, -1)), cur, f"{nxt}_a", f"p{k}v"); nodes += n; inits += i
                nodes.append(helper.make_node("Transpose", [f"{nxt}_a"], [nxt], perm=[0, 1, 3, 2])); h, w = w, h
            elif kind == "anti_transpose":
                n, i = _gather(3, list(range(w - 1, -1, -1)), cur, f"{nxt}_a", f"p{k}h"); nodes += n; inits += i
                n, i = _gather(2, list(range(h - 1, -1, -1)), f"{nxt}_a", f"{nxt}_b", f"p{k}v"); nodes += n; inits += i
                nodes.append(helper.make_node("Transpose", [f"{nxt}_b"], [nxt], perm=[0, 1, 3, 2])); h, w = w, h
            else:
                return None
        elif name == "recolor":
            mapping = {int(a): int(b) for a, b in params["map"].items()}
            weight = np.zeros((10, 10, 1, 1), dtype=np.float32)
            for src in range(10):
                weight[mapping.get(src, src), src, 0, 0] = 1.0
            inits.append(numpy_helper.from_array(weight, f"p{k}_w"))
            nodes.append(helper.make_node("Conv", [cur, f"p{k}_w"], [nxt]))
        elif name == "crop":
            t, l, ch, cw = params["top"], params["left"], params["h"], params["w"]
            if t + ch > h or l + cw > w:
                return None
            inits += [
                numpy_helper.from_array(np.asarray([0, 0, t, l], dtype=np.int64), f"p{k}_s"),
                numpy_helper.from_array(np.asarray([1, 10, t + ch, l + cw], dtype=np.int64), f"p{k}_e"),
                numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), f"p{k}_ax"),
            ]
            nodes.append(helper.make_node("Slice", [cur, f"p{k}_s", f"p{k}_e", f"p{k}_ax"], [nxt]))
            h, w = ch, cw
        elif name == "tile":
            ry, rx = params["ry"], params["rx"]
            if h * ry > 30 or w * rx > 30:
                return None
            inits.append(numpy_helper.from_array(np.asarray([1, 1, ry, rx], dtype=np.int64), f"p{k}_rep"))
            nodes.append(helper.make_node("Tile", [cur, f"p{k}_rep"], [nxt]))
            h, w = h * ry, w * rx
        else:
            return None
        cur, k = nxt, k + 1
    # pad active (h,w) back to (30,30)
    inits.append(numpy_helper.from_array(np.asarray([0, 0, 0, 0, 0, 0, 30 - h, 30 - w], dtype=np.int64), "fin_pads"))
    inits.append(numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "fin_zero"))
    nodes.append(helper.make_node("Pad", [cur, "fin_pads", "fin_zero"], ["output"], mode="constant"))
    try:
        return P.make_model(nodes, inits, "solver_program", opset_version=11)
    except Exception:
        return None


# ------------------------- enumerative search -------------------------
def _candidate_recolor_maps(examples: list[dict[str, Any]], pre: tuple) -> list[dict]:
    """Infer a single consistent color map AFTER applying program-prefix `pre`."""
    mapping: dict[int, int] = {}
    for ex in examples:
        x = apply_program(P.grid_to_array(ex["input"]), pre)
        y = P.grid_to_array(ex["output"])
        if x is None or x.shape != y.shape:
            return []
        for s, d in zip(x.ravel(), y.ravel()):
            s, d = int(s), int(d)
            if s in mapping and mapping[s] != d:
                return []
            mapping[s] = d
    return [mapping] if any(s != d for s, d in mapping.items()) else []


def enumerate_programs(examples: list[dict[str, Any]], max_depth: int = 2) -> list[tuple]:
    """Geometric (x recolor) compositions up to small depth, each numpy-verified on all
    examples. Crop is inferred as a common fixed crop when output<input."""
    found: list[tuple] = []
    geos = [("geometric", {"kind": k}) for k in GEO_KINDS]
    # depth-1 geometric, and geometric+recolor
    bases: list[tuple] = [()]
    for g in geos:
        bases.append((g,))
    if max_depth >= 2:
        for g1, g2 in itertools.product(geos, geos):
            bases.append((g1, g2))
    seen: set[str] = set()
    for pre in bases:
        if verify_program(examples, pre):
            if repr(pre) not in seen:
                found.append(pre); seen.add(repr(pre))
            continue
        for m in _candidate_recolor_maps(examples, pre):
            prog = pre + (("recolor", {"map": {str(k): v for k, v in m.items()}}),)
            if verify_program(examples, prog) and repr(prog) not in seen:
                found.append(prog); seen.add(repr(prog))
    return found


def program_to_candidate(task_id: int, program: tuple, examples: list[dict[str, Any]]) -> "P.Candidate | None":
    shapes = {P.grid_to_array(ex["input"]).shape for ex in examples}
    if len(shapes) != 1:
        return None  # variable input shape -> not statically lowerable
    ih, iw = next(iter(shapes))
    raw = lower_program(program, ih, iw)
    if raw is None:
        return None
    name = "solver_" + "_".join(op[0][:4] + (op[1].get("kind", "")[:4] if op[0] == "geometric" else "") for op in program) or "solver_id"
    return P.Candidate(task_id, name[:60], "solver", raw, "generated", f"program={program}")
