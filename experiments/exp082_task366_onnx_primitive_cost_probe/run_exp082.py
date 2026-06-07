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

from experiments.phase1_rewrite_utils import (  # noqa: E402
    infer_static_ok,
    load_neurogolf_utils,
    load_task,
    make_model,
    score_model,
    sha256,
)


EXP_ID = "exp082_task366_onnx_primitive_cost_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 366


@dataclass(frozen=True)
class PrimitiveRow:
    primitive: str
    status: str
    memory: int | str
    params: int | str
    cost: int | str
    file_bytes: int
    sha256: str
    static_reason: str
    score_reason: str
    notes: str


def init_i(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.int64), name)


def init_f(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.float32), name)


def model_identity() -> bytes:
    return make_model([helper.make_node("Identity", ["input"], ["output"])], [], EXP_ID + "_identity")


def model_static_slice_pad(axis: str) -> bytes:
    # Copy a fixed half-panel into the top-left of a 30x30 output. This is not a
    # correct task366 solution; it measures the cost of Slice+Pad panel movement.
    if axis == "vertical_bottom":
        starts = np.asarray([0, 0, 15, 0])
        ends = np.asarray([1, 10, 30, 30])
        pads = np.asarray([0, 0, 0, 0, 0, 0, 15, 0])
    elif axis == "horizontal_right":
        starts = np.asarray([0, 0, 0, 15])
        ends = np.asarray([1, 10, 30, 30])
        pads = np.asarray([0, 0, 0, 0, 0, 0, 0, 15])
    else:
        raise ValueError(axis)
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["panel"]),
        helper.make_node("Pad", ["panel", "pads", "zero"], ["output"], mode="constant"),
    ]
    inits = [
        init_i("starts", starts),
        init_i("ends", ends),
        init_i("axes", np.asarray([0, 1, 2, 3])),
        init_i("pads", pads),
        init_f("zero", np.asarray([0.0])),
    ]
    return make_model(nodes, inits, EXP_ID + "_" + axis, opset_version=11)


def model_marker_mask_conv() -> bytes:
    # A tiny local detector proxy: sum non-background channels with 1x1 Conv.
    weight = np.zeros((1, 10, 1, 1), dtype=np.float32)
    weight[0, 1:, 0, 0] = 1.0
    nodes = [
        helper.make_node("Conv", ["input", "w"], ["nz_sum"]),
        helper.make_node("Greater", ["nz_sum", "zero"], ["mask_b"]),
        helper.make_node("Cast", ["mask_b"], ["mask"], to=TensorProto.FLOAT),
        helper.make_node("Tile", ["mask", "repeats"], ["output"]),
    ]
    inits = [
        init_f("w", weight),
        init_f("zero", np.asarray([0.0])),
        init_i("repeats", np.asarray([1, 10, 1, 1])),
    ]
    return make_model(nodes, inits, EXP_ID + "_marker_mask", opset_version=11)


def model_where_overlay_proxy() -> bytes:
    # Measures cost of one full-grid Where overlay. This is a known risk pattern but
    # gives an upper-bound primitive for patch paste.
    nodes = [
        helper.make_node("Greater", ["input", "zero"], ["mask"]),
        helper.make_node("Where", ["mask", "input", "zeros"], ["output"]),
    ]
    inits = [
        init_f("zero", np.asarray([0.0])),
        init_f("zeros", np.zeros((1, 10, 30, 30), dtype=np.float32)),
    ]
    return make_model(nodes, inits, EXP_ID + "_where_overlay", opset_version=11)


def score_raw(utils: Any, name: str, raw: bytes) -> PrimitiveRow:
    model = onnx.load_model_from_string(raw)
    ok, static_reason = infer_static_ok(model)
    if not ok:
        return PrimitiveRow(name, "rejected_static", "", "", "", len(raw), sha256(raw), static_reason, "", "static check failed")
    memory, params, score_reason = score_model(utils, raw, TASK_ID, name, EXP_DIR)
    if memory is None or params is None:
        return PrimitiveRow(name, "score_failed", "", "", "", len(raw), sha256(raw), static_reason, score_reason, "score failed")
    return PrimitiveRow(name, "scored", memory, params, memory + params, len(raw), sha256(raw), static_reason, score_reason, "primitive cost only; not a valid task solution")


def smoke_runtime(raw: bytes) -> None:
    # Catch shape/runtime mistakes before score_model. Input content is irrelevant.
    sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    x = np.zeros((1, 10, 30, 30), dtype=np.float32)
    y = sess.run(None, {"input": x})[0]
    if y.shape != (1, 10, 30, 30):
        raise RuntimeError(f"bad output shape {y.shape}")


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    builders = {
        "identity": model_identity,
        "static_vertical_bottom_slice_pad": lambda: model_static_slice_pad("vertical_bottom"),
        "static_horizontal_right_slice_pad": lambda: model_static_slice_pad("horizontal_right"),
        "marker_mask_conv_tile": model_marker_mask_conv,
        "where_overlay_proxy": model_where_overlay_proxy,
    }
    rows: list[PrimitiveRow] = []
    for name, build in builders.items():
        raw = build()
        (EXP_DIR / f"{name}.onnx").write_bytes(raw)
        try:
            smoke_runtime(raw)
            rows.append(score_raw(utils, name, raw))
        except Exception as exc:
            rows.append(PrimitiveRow(name, "runtime_failed", "", "", "", len(raw), sha256(raw), "", str(exc)[:180], "runtime smoke failed"))

    with (EXP_DIR / "primitive_costs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PrimitiveRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])

    scored = [r for r in rows if r.status == "scored"]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "cost_probe_ready",
        "hypothesis": "task366 structural lowering can stay small only if Slice/Pad/Where/Conv primitives are cheap enough when composed a few times.",
        "primitive_count": len(rows),
        "scored_count": len(scored),
        "rows": [asdict(r) for r in rows],
        "decision": "Use measured primitive costs to set a pre-emission budget for task366; avoid full-grid Tile/Where repetition if it pushes cost above 600.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: primitive cost probe only; no valid task solution",
        "leakage_risk": "low: no task labels encoded.",
        "overfitting_risk": "low: cost-only primitive probe.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`task366` object-marker copy loweringに使うONNX primitiveの公式costを測る。

## 結果

- primitive count: `{len(rows)}`
- scored count: `{len(scored)}`

## 判断

これは提出候補ではない。`Slice` / `Pad` / `Conv` / `Where` の実costを見て、次のcorrectness-first compilerのpre-emission budgetを決める。

## Risk

- leakage risk: low。task labelを埋め込んでいない。
- overfitting risk: low。cost-only診断。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
