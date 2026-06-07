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


EXP_ID = "exp089_task185_homogeneous2x2_onnx_cost_probe"
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


def init_f(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.float32), name)


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


def model_static_slice_only() -> bytes:
    nodes = [helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["output"])]
    inits = [
        init_i("starts", np.asarray([0, 0, 5, 5])),
        init_i("ends", np.asarray([1, 10, 17, 17])),
        init_i("axes", np.asarray([0, 1, 2, 3])),
        init_i("steps", np.asarray([1, 1, 3, 3])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_static_slice_only")


def model_homogeneous2x2_conv() -> bytes:
    # Proxy after a static strided Slice extracts the 4x4 lattice. A grouped 2x2
    # Conv sums each color channel over each 2x2 block, then Equal(4) keeps only
    # homogeneous nonzero blocks. This measures the core compression cost.
    weight = np.zeros((10, 1, 2, 2), dtype=np.float32)
    weight[:, 0, :, :] = 1.0
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["m4"]),
        helper.make_node("Conv", ["m4", "w"], ["sum2"], group=10),
        helper.make_node("Equal", ["sum2", "four"], ["eq4"]),
        helper.make_node("Cast", ["eq4"], ["output"], to=TensorProto.FLOAT),
    ]
    inits = [
        init_i("starts", np.asarray([0, 0, 5, 5])),
        init_i("ends", np.asarray([1, 10, 17, 17])),
        init_i("axes", np.asarray([0, 1, 2, 3])),
        init_i("steps", np.asarray([1, 1, 3, 3])),
        init_f("w", weight),
        init_f("four", np.asarray([4.0])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_homogeneous2x2_conv")


def smoke(raw: bytes) -> str:
    try:
        sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        y = sess.run(None, {"input": x})[0]
        if tuple(y.shape) != (1, 10, 3, 3):
            return f"bad shape {tuple(y.shape)}"
        return "ok"
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
        ("static_slice_4x4_to_3x3_shape_probe", model_static_slice_only, "static strided Slice cost floor; not correct shape semantics beyond output rank."),
        ("homogeneous2x2_conv_proxy", model_homogeneous2x2_conv, "static strided Slice + grouped 2x2 Conv + Equal + Cast core rule proxy."),
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
        "hypothesis": "task185 2x2 homogeneous block compression can be <=600 if 4x4 lattice extraction is cheap.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "rows": [asdict(row) for row in rows],
        "decision": "If homogeneous2x2 proxy is <=600, focus on dynamic 4x4 lattice extraction; otherwise seek graph surgery or alternative non-Conv equality encoding.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: cost proxy only",
        "leakage_risk": "low: no output lookup.",
        "overfitting_risk": "medium: static slice start/step is proxy-only; full lowering must infer lattice location.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`task185` のcore ruleである4x4 lattice matrix -> 3x3 homogeneous 2x2 block判定が、ONNX cost `<=600` に入るかを測る。

## 結果

`result.json` / `cost_probe.csv` を参照。

## 判断

core compressionが軽ければ、残る課題はdynamic lattice extraction。重ければ、Conv以外の同色判定または既存artifact surgeryへ切り替える。

## Risk

- leakage risk: low。
- overfitting risk: medium。static Sliceはproxyであり、提出候補ではない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
