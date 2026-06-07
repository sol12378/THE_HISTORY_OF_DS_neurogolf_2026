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


EXP_ID = "exp092_task048_small_connectivity_cost_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 48


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


def make_model(nodes: list[Any], inits: list[Any], out_shape: list[int], name: str) -> bytes:
    graph = helper.make_graph(
        nodes,
        f"{name}_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shape)],
        inits,
    )
    model = helper.make_model(graph, producer_name=name, ir_version=10, opset_imports=[helper.make_opsetid("", 11)])
    return model.SerializeToString()


def model_reachability_core(steps: int) -> bytes:
    # Cost proxy only: slice color-8 mask to 8x8 and run 4-neighbor dilation.
    # A full solution must seed from one color-2 component and test contact with the other.
    kernel = np.asarray([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=np.float32)
    nodes: list[Any] = [
        helper.make_node("Slice", ["input", "ch8_starts", "ch8_ends", "axes4"], ["ch8"]),
        helper.make_node("Slice", ["ch8", "hw_starts", "hw_ends", "axes4"], ["passable"]),
    ]
    current = "passable"
    for i in range(steps):
        nodes.extend(
            [
                helper.make_node("Conv", [current, "kernel"], [f"nbr_{i}"], pads=[1, 1, 1, 1]),
                helper.make_node("Greater", [f"nbr_{i}", "zero"], [f"reach_b_{i}"]),
                helper.make_node("Cast", [f"reach_b_{i}"], [f"reach_f_{i}"], to=TensorProto.FLOAT),
                helper.make_node("Mul", [f"reach_f_{i}", "passable"], [f"reach_{i}"]),
            ]
        )
        current = f"reach_{i}"
    nodes.append(helper.make_node("Identity", [current], ["output"]))
    inits = [
        init_i("ch8_starts", np.asarray([0, 8, 0, 0])),
        init_i("ch8_ends", np.asarray([1, 9, 30, 30])),
        init_i("hw_starts", np.asarray([0, 0, 0, 0])),
        init_i("hw_ends", np.asarray([1, 1, 8, 8])),
        init_i("axes4", np.asarray([0, 1, 2, 3])),
        init_f("kernel", kernel),
        init_f("zero", np.asarray([0.0])),
    ]
    return make_model(nodes, inits, [1, 1, 8, 8], f"{EXP_ID}_reach_{steps}")


def model_onecell_output_floor() -> bytes:
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes4"], ["cell"]),
        helper.make_node("Identity", ["cell"], ["output"]),
    ]
    inits = [
        init_i("starts", np.asarray([0, 0, 0, 0])),
        init_i("ends", np.asarray([1, 10, 1, 1])),
        init_i("axes4", np.asarray([0, 1, 2, 3])),
    ]
    return make_model(nodes, inits, [1, 10, 1, 1], f"{EXP_ID}_onecell_floor")


def smoke(raw: bytes, out_shape: tuple[int, ...]) -> str:
    try:
        sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
        y = sess.run(None, {"input": np.zeros((1, 10, 30, 30), dtype=np.float32)})[0]
        return "ok" if tuple(y.shape) == out_shape else f"bad shape {tuple(y.shape)}"
    except Exception as exc:
        return f"runtime failed: {str(exc)[:180]}"


def score_raw(utils: Any, variant: str, raw: bytes, out_shape: tuple[int, ...], notes: str) -> Row:
    runtime_reason = smoke(raw, out_shape)
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
        ("onecell_output_floor", model_onecell_output_floor, (1, 10, 1, 1), "1x1 output one-hot slice floor."),
        ("reachability_4step_8x8", lambda: model_reachability_core(4), (1, 1, 8, 8), "4-step 8x8 Conv dilation proxy."),
        ("reachability_8step_8x8", lambda: model_reachability_core(8), (1, 1, 8, 8), "8-step 8x8 Conv dilation proxy."),
    ]
    rows = []
    for variant, build, shape, notes in builders:
        raw = build()
        (EXP_DIR / f"{variant}.onnx").write_bytes(raw)
        rows.append(score_raw(utils, variant, raw, shape, notes))
    with (EXP_DIR / "cost_probe.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "cost_probe_ready",
        "hypothesis": "Because task048 input is at most 8x8 and output is 1x1, bounded connectivity may be low enough for cost<=600.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "rows": [asdict(r) for r in rows],
        "decision": "If 8-step proxy is far above 600, avoid naive connectivity unroll and look for graph surgery or closed-form path features.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: cost proxy only",
        "leakage_risk": "low: no labels encoded.",
        "overfitting_risk": "medium: proxy only; not a full correctness model.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`task048` の8x8小領域connectivity ruleをONNXへ落とす余地があるか、bounded dilation proxyでcostを測る。

## 結果

`result.json` / `cost_probe.csv` を参照。

## Decision

8-step proxyが600を大きく超える場合、naive connectivity unrollは避け、既存artifact surgeryまたは閉形式のpath特徴を探す。

## Risk

- leakage risk: low。
- overfitting risk: medium。proxyであり提出候補ではない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
