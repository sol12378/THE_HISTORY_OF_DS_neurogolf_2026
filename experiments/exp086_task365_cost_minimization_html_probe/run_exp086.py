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
    load_base_tasks,
    load_neurogolf_utils,
    score_model,
    sha256,
)


EXP_ID = "exp086_task365_cost_minimization_html_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 365


@dataclass(frozen=True)
class ProbeRow:
    variant: str
    status: str
    output_shape: str
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


def make_probe_model(nodes: list[Any], inits: list[Any], output_shape: list[int], name: str, opset: int = 11) -> bytes:
    graph = helper.make_graph(
        nodes,
        f"{name}_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        inits,
    )
    model = helper.make_model(
        graph,
        producer_name=name,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    return model.SerializeToString()


def model_slice_small_output(h: int, w: int) -> bytes:
    nodes = [helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["output"])]
    inits = [
        init_i("starts", np.asarray([0, 0, 0, 0])),
        init_i("ends", np.asarray([1, 10, h, w])),
        init_i("axes", np.asarray([0, 1, 2, 3])),
    ]
    return make_probe_model(nodes, inits, [1, 10, h, w], f"{EXP_ID}_slice_{h}x{w}")


def model_slice_pad(h: int, w: int) -> bytes:
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["crop"]),
        helper.make_node("Pad", ["crop", "pads", "zero"], ["output"], mode="constant"),
    ]
    inits = [
        init_i("starts", np.asarray([0, 0, 0, 0])),
        init_i("ends", np.asarray([1, 10, h, w])),
        init_i("axes", np.asarray([0, 1, 2, 3])),
        init_i("pads", np.asarray([0, 0, 0, 0, 0, 0, 30 - h, 30 - w])),
        init_f("zero", np.asarray([0.0])),
    ]
    return make_probe_model(nodes, inits, [1, 10, 30, 30], f"{EXP_ID}_slice_pad_{h}x{w}")


def model_gather_window_pad(h: int, w: int) -> bytes:
    # Dynamic-looking window copy using Gather row/col index tensors, then Pad.
    # This approximates the memory floor of a rectangle selector after the chosen
    # top-left and shape are known.
    nodes = [
        helper.make_node("Gather", ["input", "rows"], ["row_g"], axis=2),
        helper.make_node("Gather", ["row_g", "cols"], ["crop"], axis=3),
        helper.make_node("Pad", ["crop", "pads", "zero"], ["output"], mode="constant"),
    ]
    inits = [
        init_i("rows", np.arange(h)),
        init_i("cols", np.arange(w)),
        init_i("pads", np.asarray([0, 0, 0, 0, 0, 0, 30 - h, 30 - w])),
        init_f("zero", np.asarray([0.0])),
    ]
    return make_probe_model(nodes, inits, [1, 10, 30, 30], f"{EXP_ID}_gather_pad_{h}x{w}")


def model_small_equal_pad(h: int, w: int) -> bytes:
    # Test the HTML advice "apply CHEAP ops after Slice"; Equal is done on a
    # small crop, but the final output still needs the official 30x30 one-hot frame.
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["crop"]),
        helper.make_node("Equal", ["crop", "one"], ["eq"]),
        helper.make_node("Cast", ["eq"], ["eq_f"], to=TensorProto.FLOAT),
        helper.make_node("Pad", ["eq_f", "pads", "zero"], ["output"], mode="constant"),
    ]
    inits = [
        init_i("starts", np.asarray([0, 0, 0, 0])),
        init_i("ends", np.asarray([1, 10, h, w])),
        init_i("axes", np.asarray([0, 1, 2, 3])),
        init_f("one", np.asarray([1.0])),
        init_i("pads", np.asarray([0, 0, 0, 0, 0, 0, 30 - h, 30 - w])),
        init_f("zero", np.asarray([0.0])),
    ]
    return make_probe_model(nodes, inits, [1, 10, 30, 30], f"{EXP_ID}_eq_pad_{h}x{w}")


def model_minimal_identity() -> bytes:
    return make_probe_model([helper.make_node("Identity", ["input"], ["output"])], [], [1, 10, 30, 30], f"{EXP_ID}_identity")


def smoke(raw: bytes, expected_shape: tuple[int, ...]) -> str:
    try:
        sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        y = sess.run(None, {"input": x})[0]
        if tuple(y.shape) != expected_shape:
            return f"bad shape {tuple(y.shape)} expected {expected_shape}"
        return "ok"
    except Exception as exc:
        return f"runtime failed: {str(exc)[:180]}"


def score_raw(utils: Any, variant: str, raw: bytes, expected_shape: tuple[int, ...], notes: str) -> ProbeRow:
    out_shape = "x".join(str(x) for x in expected_shape)
    runtime_reason = smoke(raw, expected_shape)
    model = onnx.load_model_from_string(raw)
    ok, static_reason = infer_static_ok(model)
    if runtime_reason != "ok" or not ok:
        return ProbeRow(
            variant,
            "rejected",
            out_shape,
            len(model.graph.node),
            "",
            "",
            "",
            len(raw),
            sha256(raw),
            static_reason,
            "",
            runtime_reason,
            notes,
        )
    memory, params, score_reason = score_model(utils, raw, TASK_ID, variant, EXP_DIR)
    if memory is None or params is None:
        return ProbeRow(
            variant,
            "score_failed",
            out_shape,
            len(model.graph.node),
            "",
            "",
            "",
            len(raw),
            sha256(raw),
            static_reason,
            score_reason,
            runtime_reason,
            notes,
        )
    return ProbeRow(
        variant,
        "scored",
        out_shape,
        len(model.graph.node),
        int(memory),
        int(params),
        int(memory) + int(params),
        len(raw),
        sha256(raw),
        static_reason,
        score_reason,
        runtime_reason,
        notes,
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()[TASK_ID]
    builders: list[tuple[str, Any, tuple[int, ...], str]] = [
        ("identity_30x30_floor", model_minimal_identity, (1, 10, 30, 30), "公式30x30出力のゼロノード級memory floor。"),
        ("slice_3x3_small_output", lambda: model_slice_small_output(3, 3), (1, 10, 3, 3), "Padなし小出力が採点可能かを見る。"),
        ("slice_6x6_small_output", lambda: model_slice_small_output(6, 6), (1, 10, 6, 6), "task365最大shape相当のPadなし小出力。"),
        ("slice_3x3_pad30", lambda: model_slice_pad(3, 3), (1, 10, 30, 30), "FREE Slice後に公式30x30へPadする最小proxy。"),
        ("slice_6x6_pad30", lambda: model_slice_pad(6, 6), (1, 10, 30, 30), "task365最大shape cropをPadで戻すproxy。"),
        ("gather_6x6_pad30", lambda: model_gather_window_pad(6, 6), (1, 10, 30, 30), "選択済みrow/colをGatherしてPadするproxy。"),
        ("small_equal_6x6_pad30", lambda: model_small_equal_pad(6, 6), (1, 10, 30, 30), "小領域CHEAP op + Pad のproxy。"),
    ]
    rows: list[ProbeRow] = []
    for variant, build, shape, notes in builders:
        raw = build()
        (EXP_DIR / f"{variant}.onnx").write_bytes(raw)
        rows.append(score_raw(utils, variant, raw, shape, notes))

    with (EXP_DIR / "cost_probe.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ProbeRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])

    scored = [row for row in rows if row.status == "scored"]
    min_scored_cost = min([int(row.cost) for row in scored], default=None)
    pad_costs = {row.variant: row.cost for row in scored if "pad30" in row.variant or "identity" in row.variant}
    decision = (
        "score_network alone accepts small static outputs at cost 12, but official validation compares padded one-hot tensors directly. "
        "For a full task365 solution the output must behave as [1,10,30,30]; the minimal 6x6 Slice+Pad proxy already costs 1461 before object selection. "
        "Therefore HTML-style FREE-op minimization is useful for large reductions, but task365 is unlikely to reach cost<=600 unless the rule can avoid the 6x6 crop memory floor or split into a smaller shape-specific artifact, which a single task model cannot generally do."
    )
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "cost_probe_ready",
        "hypothesis": "HTMLのFREE-op小領域化技法によりtask365 dense-rectangle cropが250〜600級へ近づく可能性がある。",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "rows": [asdict(row) for row in rows],
        "min_scored_cost": min_scored_cost,
        "pad_or_identity_costs": pad_costs,
        "decision": decision,
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: primitive/proxy cost probe only",
        "leakage_risk": "low: task labels/output patterns are not encoded; cost-only probe.",
        "overfitting_risk": "low: no candidate solution emitted.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

ユーザー提供HTMLのONNX cost最小化技法が、`task365` dense rectangle cropを `cost<=250〜600` 級へ落とす現実的余地を持つか、公式 `score_network` で小さなproxyを実測する。

## 仮説

`Slice/Gather/Pad` だけで小cropを扱えれば、`task365` の正答rule (`max_count2` object crop) は既存artifact cost `{base.cost}` から大きく削れる。ただし公式出力が `[1,10,30,30]` 固定の場合、Pad後のmemory floorだけで600を超える可能性がある。

## 結果

- scored variants: `{len(scored)}/{len(rows)}`
- min scored cost: `{min_scored_cost}`
- baseline cost: `{base.cost}`

## 判断

{decision}

## 追加解釈

`score_network` は小さいgraph outputも計測できるが、公式 `verify_subset` は `convert_to_numpy(example)["output"]` と `run_network` 出力を `np.array_equal` で直接比較する。したがって、Padなし `1x10x3x3` / `1x10x6x6` はcost診断としては有効でも、`task365` の提出候補にはならない。

`task365` は selected shape に `(6,6)` が1例あり、hidden側でも最大shapeが出る可能性を考えると、少なくとも `6x6` 相当のcrop領域をpadded one-hot上に表現する必要がある。`Slice+Pad` proxyの `1461` が、selectorなしの下限に近い。

## Risk

- leakage risk: low。cost-only probeで、output lookupはない。
- overfitting risk: low。提出候補ではない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
