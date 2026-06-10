from __future__ import annotations

import csv
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import infer_static_ok, load_base_tasks, load_neurogolf_utils, score_model, sha256  # noqa: E402


EXP_ID = "exp202_task185_selected_lattice_extract_cost_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185
CURRENT_PUBLIC_BEST_LB = 6005.93


@dataclass(frozen=True)
class Row:
    variant: str
    status: str
    node_count: int
    memory: int | str
    params: int | str
    cost: int | str
    file_bytes: int
    sha256: str
    static_reason: str
    score_reason: str
    runtime_reason: str
    notes: str


def init_i(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.int64), name)


def init_f(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.float32), name)


def make_model(nodes: list[Any], inits: list[Any], name: str, output_shape: list[int] | None = None) -> bytes:
    if output_shape is None:
        output_shape = [1, 10, 30, 30]
    graph = helper.make_graph(
        nodes,
        f"{name}_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        inits,
    )
    model = helper.make_model(graph, producer_name=name, ir_version=10, opset_imports=[helper.make_opsetid("", 11)])
    return model.SerializeToString()


def selected_indices(start: int = 4, spacing: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray([start + spacing * k for k in range(4)], dtype=np.int64)
    cols = rows.copy()
    row_idx = np.zeros((1, 10, 4, 30), dtype=np.int64)
    for i, r in enumerate(rows):
        row_idx[:, :, i, :] = r
    col_idx = np.zeros((1, 10, 4, 4), dtype=np.int64)
    for j, c in enumerate(cols):
        col_idx[:, :, :, j] = c
    return row_idx, col_idx


def model_gatherelements_lattice_core_pad() -> bytes:
    # Cost proxy for a selected 4x4 lattice extraction. Static indices stand in
    # for dynamic indices emitted by the axis selector. The important question is
    # whether extraction+core has a cost budget under task185 baseline.
    row_idx, col_idx = selected_indices()
    weight = np.zeros((10, 1, 2, 2), dtype=np.float32)
    weight[:, 0, :, :] = 1.0
    nodes = [
        helper.make_node("GatherElements", ["input", "row_idx"], ["selected_rows"], axis=2),
        helper.make_node("GatherElements", ["selected_rows", "col_idx"], ["lattice_4x4"], axis=3),
        helper.make_node("Conv", ["lattice_4x4", "w"], ["sum2"], group=10),
        helper.make_node("Equal", ["sum2", "four"], ["eq4"]),
        helper.make_node("Cast", ["eq4"], ["small"], to=TensorProto.FLOAT),
        helper.make_node("Pad", ["small", "pads", "zero"], ["output"], mode="constant"),
    ]
    inits = [
        init_i("row_idx", row_idx),
        init_i("col_idx", col_idx),
        init_f("w", weight),
        init_f("four", np.asarray([4.0])),
        init_f("zero", np.asarray([0.0])),
        init_i("pads", np.asarray([0, 0, 0, 0, 0, 0, 27, 27])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_gatherelements_lattice_core_pad")


def model_gathernd_16_cells_core_pad() -> bytes:
    # Alternative small sparse extraction: gather 16 spatial cells after NHWC
    # transpose, then reshape to NCHW 4x4 and run the same 2x2 homogeneous core.
    rows = [4 + 5 * k for k in range(4)]
    cols = rows.copy()
    coords = np.asarray([[0, r, c] for r in rows for c in cols], dtype=np.int64)
    weight = np.zeros((10, 1, 2, 2), dtype=np.float32)
    weight[:, 0, :, :] = 1.0
    nodes = [
        helper.make_node("Transpose", ["input"], ["nhwc"], perm=[0, 2, 3, 1]),
        helper.make_node("GatherND", ["nhwc", "coords"], ["cells_16x10"]),
        helper.make_node("Transpose", ["cells_16x10"], ["cells_10x16"], perm=[1, 0]),
        helper.make_node("Reshape", ["cells_10x16", "shape_4x4"], ["lattice_4x4"]),
        helper.make_node("Conv", ["lattice_4x4", "w"], ["sum2"], group=10),
        helper.make_node("Equal", ["sum2", "four"], ["eq4"]),
        helper.make_node("Cast", ["eq4"], ["small"], to=TensorProto.FLOAT),
        helper.make_node("Pad", ["small", "pads", "zero"], ["output"], mode="constant"),
    ]
    inits = [
        init_i("coords", coords),
        init_i("shape_4x4", np.asarray([1, 10, 4, 4])),
        init_f("w", weight),
        init_f("four", np.asarray([4.0])),
        init_f("zero", np.asarray([0.0])),
        init_i("pads", np.asarray([0, 0, 0, 0, 0, 0, 27, 27])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_gathernd_16_cells_core_pad")


def smoke(raw: bytes) -> str:
    try:
        sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
        y = sess.run(None, {"input": np.zeros((1, 10, 30, 30), dtype=np.float32)})[0]
        return "ok" if tuple(y.shape) == (1, 10, 30, 30) else f"bad shape {tuple(y.shape)}"
    except Exception as exc:
        return f"runtime failed: {str(exc)[:180]}"


def score_raw(utils: Any, variant: str, raw: bytes, notes: str) -> Row:
    runtime_reason = smoke(raw)
    model = onnx.load_model_from_string(raw)
    ok, static_reason = infer_static_ok(model)
    if runtime_reason != "ok" or not ok:
        return Row(variant, "rejected", len(model.graph.node), "", "", "", len(raw), sha256(raw), static_reason, "", runtime_reason, notes)
    memory, params, score_reason = score_model(utils, raw, TASK_ID, variant, EXP_DIR)
    if memory is None or params is None:
        return Row(variant, "score_failed", len(model.graph.node), "", "", "", len(raw), sha256(raw), static_reason, score_reason, runtime_reason, notes)
    return Row(variant, "scored", len(model.graph.node), int(memory), int(params), int(memory) + int(params), len(raw), sha256(raw), static_reason, score_reason, runtime_reason, notes)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()[TASK_ID]
    builders = [
        (
            "gatherelements_lattice_core_pad",
            model_gatherelements_lattice_core_pad,
            "Selected 4x4 lattice extraction via two GatherElements ops, then homogeneous 2x2 core.",
        ),
        (
            "gathernd_16_cells_core_pad",
            model_gathernd_16_cells_core_pad,
            "Selected 16-cell sparse GatherND extraction, then homogeneous 2x2 core.",
        ),
    ]
    rows: list[Row] = []
    for variant, build, notes in builders:
        raw = build()
        (EXP_DIR / f"{variant}.onnx").write_bytes(raw)
        rows.append(score_raw(utils, variant, raw, notes))

    with (EXP_DIR / "cost_probe.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])

    scored = [row for row in rows if row.status == "scored"]
    best = None if not scored else min(scored, key=lambda row: int(row.cost))
    axis_selector_proxy_cost = 4156
    projected_best_with_axis_selector = None if best is None else int(best.cost) + axis_selector_proxy_cost
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "cost_probe_ready",
        "task_id": TASK_ID,
        "hypothesis": "If selected row/col indices are available, task185 lattice extraction + homogeneous core can fit under the baseline cost budget.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "current_public_best_lb": CURRENT_PUBLIC_BEST_LB,
        "rows": [asdict(row) for row in rows],
        "best_scored": None if best is None else asdict(best),
        "axis_selector_proxy_cost_from_exp201": axis_selector_proxy_cost,
        "projected_best_with_axis_selector": projected_best_with_axis_selector,
        "projected_under_baseline": None if projected_best_with_axis_selector is None else projected_best_with_axis_selector < base.cost,
        "decision": (
            "If projected_under_baseline is true, proceed to a correctness-first dynamic-index task185 lowering. "
            "Prefer the lower extraction primitive and avoid enumerating all window pairs."
        ),
        "submission_decision": "no_submit: cost probe only",
        "leakage_risk": "low: cost proxy only, no output lookup.",
        "overfitting_risk": "medium-low: static indices are a proxy for dynamic selected indices, not a candidate.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task185でselected row/col indexが得られた後、4x4 lattice extraction + homogeneous 2x2 coreがbaseline cost内に収まるか測る。

## 結果

- best_scored: `{None if best is None else best.variant}`
- best_cost: `{None if best is None else best.cost}`
- axis_selector_proxy_cost_from_exp201: `{axis_selector_proxy_cost}`
- projected_best_with_axis_selector: `{projected_best_with_axis_selector}`
- baseline_cost: `{base.cost}`

## 判断

projected costがbaseline未満なら、次はdynamic selected indexを実際に作るcorrectness-first loweringへ進む。window pair全列挙は避ける。

## リスク

- leakage risk: low。cost proxyのみ。
- overfitting risk: medium-low。static indexはdynamic selectorの代用で、提出candidateではない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
