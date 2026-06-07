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


EXP_ID = "exp090_task185_nonconv_2x2_cost_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185


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


def make_model(nodes: list[Any], inits: list[Any], name: str) -> bytes:
    graph = helper.make_graph(
        nodes,
        f"{name}_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 3, 3])],
        inits,
    )
    model = helper.make_model(
        graph,
        producer_name=name,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 11)],
    )
    return model.SerializeToString()


def model_mul4_static_lattice() -> bytes:
    # Static proxy: extract a 4x4 lattice matrix, then multiply the four shifted
    # 3x3 views. With one-hot inputs, the product is 1 only for homogeneous
    # nonzero 2x2 blocks of the same color channel.
    nodes = [
        helper.make_node("Slice", ["input", "m4_starts", "m4_ends", "axes", "m4_steps"], ["m4"]),
        helper.make_node("Slice", ["m4", "s00", "e00", "axes"], ["a"]),
        helper.make_node("Slice", ["m4", "s01", "e01", "axes"], ["b"]),
        helper.make_node("Slice", ["m4", "s10", "e10", "axes"], ["c"]),
        helper.make_node("Slice", ["m4", "s11", "e11", "axes"], ["d"]),
        helper.make_node("Mul", ["a", "b"], ["ab"]),
        helper.make_node("Mul", ["c", "d"], ["cd"]),
        helper.make_node("Mul", ["ab", "cd"], ["output"]),
    ]
    inits = [
        init_i("m4_starts", np.asarray([0, 0, 5, 5])),
        init_i("m4_ends", np.asarray([1, 10, 17, 17])),
        init_i("axes", np.asarray([0, 1, 2, 3])),
        init_i("m4_steps", np.asarray([1, 1, 3, 3])),
        init_i("s00", np.asarray([0, 0, 0, 0])),
        init_i("e00", np.asarray([1, 10, 3, 3])),
        init_i("s01", np.asarray([0, 0, 0, 1])),
        init_i("e01", np.asarray([1, 10, 3, 4])),
        init_i("s10", np.asarray([0, 0, 1, 0])),
        init_i("e10", np.asarray([1, 10, 4, 3])),
        init_i("s11", np.asarray([0, 0, 1, 1])),
        init_i("e11", np.asarray([1, 10, 4, 4])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_mul4_static_lattice")


def model_mul4_no_m4_output() -> bytes:
    # Same logic but Slice each shifted view directly from input. This avoids
    # materializing m4, but repeats static Slice initializers.
    nodes = [
        helper.make_node("Slice", ["input", "s00", "e00", "axes", "steps"], ["a"]),
        helper.make_node("Slice", ["input", "s01", "e01", "axes", "steps"], ["b"]),
        helper.make_node("Slice", ["input", "s10", "e10", "axes", "steps"], ["c"]),
        helper.make_node("Slice", ["input", "s11", "e11", "axes", "steps"], ["d"]),
        helper.make_node("Mul", ["a", "b"], ["ab"]),
        helper.make_node("Mul", ["c", "d"], ["cd"]),
        helper.make_node("Mul", ["ab", "cd"], ["output"]),
    ]
    inits = [
        init_i("axes", np.asarray([0, 1, 2, 3])),
        init_i("steps", np.asarray([1, 1, 3, 3])),
        init_i("s00", np.asarray([0, 0, 5, 5])),
        init_i("e00", np.asarray([1, 10, 14, 14])),
        init_i("s01", np.asarray([0, 0, 5, 8])),
        init_i("e01", np.asarray([1, 10, 14, 17])),
        init_i("s10", np.asarray([0, 0, 8, 5])),
        init_i("e10", np.asarray([1, 10, 17, 14])),
        init_i("s11", np.asarray([0, 0, 8, 8])),
        init_i("e11", np.asarray([1, 10, 17, 17])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_mul4_direct")


def smoke(raw: bytes) -> str:
    try:
        sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
        y = sess.run(None, {"input": np.zeros((1, 10, 30, 30), dtype=np.float32)})[0]
        return "ok" if tuple(y.shape) == (1, 10, 3, 3) else f"bad shape {tuple(y.shape)}"
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
        ("mul4_static_lattice", model_mul4_static_lattice, "Slice 4x4 lattice, four shifted Slice views, three Mul nodes."),
        ("mul4_direct_shifted_slices", model_mul4_no_m4_output, "Four direct shifted strided Slice views, three Mul nodes; avoids m4 intermediate."),
    ]
    rows = []
    for variant, build, notes in builders:
        raw = build()
        (EXP_DIR / f"{variant}.onnx").write_bytes(raw)
        rows.append(score_raw(utils, variant, raw, notes))
    with (EXP_DIR / "cost_probe.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "cost_probe_ready",
        "hypothesis": "Non-Conv Mul-of-four one-hot equality may reduce task185 core 2x2 compression below exp089 cost 1147.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "rows": [asdict(row) for row in rows],
        "decision": "If non-Conv proxy is still >600, task185 needs artifact surgery or a more fused Slice/Mul representation before full dynamic lowering.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: cost proxy only",
        "leakage_risk": "low: no labels/output lookup.",
        "overfitting_risk": "medium: static lattice coordinate proxy only.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`task185` のhomogeneous 2x2 block判定をConvなしで表し、`exp089` のcore proxy cost `1147` を下げられるか測る。

## 結果

`result.json` と `cost_probe.csv` を参照。

## Decision

600を超える場合は、標準ONNXでの新規loweringより既存artifact surgeryや、よりfusedな表現を探す。

## Risk

- leakage risk: low。
- overfitting risk: medium。static lattice位置のproxyで、提出候補ではない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
