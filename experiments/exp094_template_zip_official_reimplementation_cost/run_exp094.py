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

from experiments.phase1_rewrite_utils import infer_static_ok, load_neurogolf_utils, make_model, score_model, sha256  # noqa: E402


EXP_ID = "exp094_template_zip_official_reimplementation_cost"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID_FOR_SCORE = 48


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


def model_identity() -> bytes:
    return make_model([helper.make_node("Identity", ["input"], ["output"])], [], f"{EXP_ID}_identity", opset_version=11)


def model_slice_flip_lr() -> bytes:
    inits = [
        init_i("starts", np.asarray([-1])),
        init_i("ends", np.asarray([-(1 << 31)])),
        init_i("axes", np.asarray([3])),
        init_i("steps", np.asarray([-1])),
    ]
    nodes = [helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["output"])]
    return make_model(nodes, inits, f"{EXP_ID}_flip_lr", opset_version=11)


def model_slice_rot180() -> bytes:
    inits = [
        init_i("starts", np.asarray([-1, -1])),
        init_i("ends", np.asarray([-(1 << 31), -(1 << 31)])),
        init_i("axes", np.asarray([2, 3])),
        init_i("steps", np.asarray([-1, -1])),
    ]
    nodes = [helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["output"])]
    return make_model(nodes, inits, f"{EXP_ID}_rot180", opset_version=11)


def model_transpose_hw() -> bytes:
    nodes = [helper.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])]
    return make_model(nodes, [], f"{EXP_ID}_transpose_hw", opset_version=11)


def model_rot90_ccw() -> bytes:
    inits = [
        init_i("starts", np.asarray([-1])),
        init_i("ends", np.asarray([-(1 << 31)])),
        init_i("axes", np.asarray([2])),
        init_i("steps", np.asarray([-1])),
    ]
    nodes = [
        helper.make_node("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2]),
        helper.make_node("Slice", ["t", "starts", "ends", "axes", "steps"], ["output"]),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_rot90_ccw", opset_version=11)


def model_color_reorder_gather() -> bytes:
    # Official one-hot recolor can be channel Gather, not integer-grid LUT.
    # Example mapping swaps colors 2 and 3 by reordering output channels.
    idx = np.arange(10)
    idx[2], idx[3] = 3, 2
    nodes = [helper.make_node("Gather", ["input", "idx"], ["output"], axis=1)]
    return make_model(nodes, [init_i("idx", idx)], f"{EXP_ID}_channel_gather_recolor", opset_version=11)


def smoke(raw: bytes) -> str:
    try:
        sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
        y = sess.run(None, {"input": np.zeros((1, 10, 30, 30), dtype=np.float32)})[0]
        return "ok" if tuple(y.shape) == (1, 10, 30, 30) else f"bad shape {tuple(y.shape)}"
    except Exception as exc:
        return f"failed: {str(exc)[:220]}"


def score_raw(utils: Any, variant: str, raw: bytes, notes: str) -> Row:
    runtime_reason = smoke(raw)
    model = onnx.load_model_from_string(raw)
    ok, static_reason = infer_static_ok(model)
    if runtime_reason != "ok" or not ok:
        return Row(variant, "rejected", len(model.graph.node), "", "", "", len(raw), sha256(raw), static_reason, "", runtime_reason, notes)
    memory, params, reason = score_model(utils, raw, TASK_ID_FOR_SCORE, variant, EXP_DIR)
    if memory is None or params is None:
        return Row(variant, "score_failed", len(model.graph.node), "", "", "", len(raw), sha256(raw), static_reason, reason, runtime_reason, notes)
    return Row(variant, "scored", len(model.graph.node), int(memory), int(params), int(memory) + int(params), len(raw), sha256(raw), static_reason, reason, runtime_reason, notes)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    builders = [
        ("official_identity", model_identity, "Official one-hot Identity equivalent."),
        ("official_flip_lr", model_slice_flip_lr, "Official one-hot Slice flip over width axis."),
        ("official_rot180", model_slice_rot180, "Official one-hot Slice flip over H/W axes."),
        ("official_transpose_hw", model_transpose_hw, "Official one-hot H/W Transpose."),
        ("official_rot90_ccw", model_rot90_ccw, "Official one-hot Transpose + Slice."),
        ("official_channel_gather_recolor", model_color_reorder_gather, "Official one-hot channel Gather recolor."),
    ]
    rows = []
    for variant, build, notes in builders:
        raw = build()
        (EXP_DIR / f"{variant}.onnx").write_bytes(raw)
        rows.append(score_raw(utils, variant, raw, notes))
    with (EXP_DIR / "official_reimplementation_costs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "cost_probe_ready",
        "hypothesis": "Even though the zip templates are not directly official-compatible, their simple geometric ideas may be cheap when reimplemented directly in official one-hot form.",
        "rows": [asdict(r) for r in rows],
        "decision": (
            "Use the zip as a template taxonomy, not as drop-in code. "
            "Direct official one-hot reimplementations of pure data-movement transforms can be low-cost, "
            "but uint8/int-grid cost claims do not transfer to tasks requiring one-hot<->integer conversion."
        ),
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: primitive cost probe only",
        "leakage_risk": "low.",
        "overfitting_risk": "low.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

zipテンプレートは直接公式互換ではないが、同じ幾何変換を公式one-hot形式で再実装した場合に250〜600級へ入るかを測る。

## 結果

`result.json` / `official_reimplementation_costs.csv` を参照。

## Decision

zipはdrop-in codeではなく、template taxonomyとして使う。純粋な `Slice` / `Transpose` / channel `Gather` 変換は公式one-hotでも安くなり得るが、uint8整数grid前提のcost値はそのまま移植できない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
