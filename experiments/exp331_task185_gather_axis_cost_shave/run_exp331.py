from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    evaluate_candidate,
    infer_static_ok,
    load_base_tasks,
    load_neurogolf_utils,
    point,
    score_model,
    sha256,
)

EXP_ID = "exp331_task185_gather_axis_cost_shave"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185
CURRENT_PUBLIC_BEST_LB = 6008.96

EXP330_PATH = ROOT / "experiments" / "exp330_task185_compact_dynamic_full_candidate" / "run_exp330.py"
spec = importlib.util.spec_from_file_location("exp330_runner", EXP330_PATH)
exp330 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["exp330_runner"] = exp330
spec.loader.exec_module(exp330)


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
    validation_status: str
    candidate_status: str
    candidate_reason: str
    local_delta_if_any: float | str
    notes: str


def compact_gather_axis_nodes() -> list[Any]:
    return [
        helper.make_node("Gather", ["start_values", "best_row"], ["row_start"]),
        helper.make_node("Gather", ["spacing_values", "best_row"], ["row_spacing"]),
        helper.make_node("Gather", ["start_values", "best_col"], ["col_start"]),
        helper.make_node("Gather", ["spacing_values", "best_col"], ["col_spacing"]),
        helper.make_node("Mul", ["row_spacing", "steps4"], ["row_offsets"]),
        helper.make_node("Add", ["row_start", "row_offsets"], ["row_vec_raw"]),
        helper.make_node("Mul", ["col_spacing", "steps4"], ["col_offsets"]),
        helper.make_node("Add", ["col_start", "col_offsets"], ["col_vec_raw"]),
        helper.make_node("Reshape", ["row_vec_raw", "vec4_shape"], ["row_vec"]),
        helper.make_node("Reshape", ["col_vec_raw", "vec4_shape"], ["col_vec"]),
    ]


def build_candidate() -> bytes:
    nodes = exp330.common_selector_nodes()
    nodes.extend(compact_gather_axis_nodes())
    nodes.extend(
        [
            helper.make_node("Gather", ["score_input", "row_vec"], ["selected_rows"], axis=2),
            helper.make_node("Gather", ["selected_rows", "col_vec"], ["lattice_4x4"], axis=3),
            helper.make_node("Conv", ["lattice_4x4", "core_w"], ["sum2"], group=10),
            helper.make_node("Equal", ["sum2", "four"], ["eq4"]),
            helper.make_node("Cast", ["eq4"], ["small_nonzero"], to=TensorProto.FLOAT),
            helper.make_node("Slice", ["small_nonzero", "c1_start", "c1_end", "axis1"], ["small_tail"]),
            helper.make_node("ReduceMax", ["small_tail"], ["any_tail"], axes=[1], keepdims=1),
            helper.make_node("Sub", ["one_f", "any_tail"], ["bg_small"]),
            helper.make_node("Concat", ["bg_small", "small_tail"], ["small_out"], axis=1),
            helper.make_node("Pad", ["small_out", "pads", "zero"], ["output"], mode="constant"),
        ]
    )
    specs = exp330.window_specs()
    nonzero_mask = np.ones((1, 10, 1, 1), dtype=np.float32)
    nonzero_mask[:, 0, :, :] = 0.0
    core_w = np.ones((10, 1, 2, 2), dtype=np.float32)
    inits = [
        exp330.init_i("start1", np.asarray([1])),
        exp330.init_i("end10", np.asarray([10])),
        exp330.init_i("axis1", np.asarray([1])),
        exp330.init_i("one_i64", np.asarray([1])),
        exp330.init_i("depth", np.asarray(10)),
        exp330.init_f("onehot_values", np.asarray([0.0, 1.0])),
        exp330.init_i("bg_shape", np.asarray([1, 10, 1, 1])),
        exp330.init_f("one_f", np.asarray([1.0])),
        exp330.init_f("nonzero_channel_mask", nonzero_mask),
        exp330.init_f("row_line_w", np.ones((1, 1, 1, 30), dtype=np.float32)),
        exp330.init_f("col_line_w", np.ones((1, 1, 30, 1), dtype=np.float32)),
        exp330.init_f("row_win_w", np.ones((1, 1, 4, 1), dtype=np.float32)),
        exp330.init_f("col_win_w", np.ones((1, 1, 1, 4), dtype=np.float32)),
        exp330.init_i("start_values", np.asarray([s for s, _ in specs])),
        exp330.init_i("spacing_values", np.asarray([sp for _, sp in specs])),
        exp330.init_i("steps4", np.asarray([0, 1, 2, 3])),
        exp330.init_i("vec4_shape", np.asarray([4])),
        exp330.init_f("core_w", core_w),
        exp330.init_f("four", np.asarray([4.0])),
        exp330.init_i("c1_start", np.asarray([1])),
        exp330.init_i("c1_end", np.asarray([10])),
        exp330.init_f("zero", np.asarray([0.0])),
        exp330.init_i("pads", np.asarray([0, 0, 0, 0, 0, 0, 27, 27])),
    ]
    return exp330.make_model(nodes, inits, f"{EXP_ID}_gather_axis_float")


def smoke(raw: bytes) -> str:
    try:
        sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
        y = sess.run(None, {"input": np.zeros((1, 10, 30, 30), dtype=np.float32)})[0]
        return "ok" if tuple(y.shape) == (1, 10, 30, 30) else f"bad shape {tuple(y.shape)}"
    except Exception as exc:
        return f"runtime failed: {str(exc)[:180]}"


def eval_raw(utils: Any, base: BaseTask, variant: str, raw: bytes) -> Row:
    runtime_reason = smoke(raw)
    model = onnx.load_model_from_string(raw)
    ok, static_reason = infer_static_ok(model)
    memory: int | str = ""
    params: int | str = ""
    cost: int | str = ""
    score_reason = ""
    validation_status = "not_run"
    candidate_status = "rejected"
    candidate_reason = runtime_reason if runtime_reason != "ok" else static_reason
    local_delta: float | str = ""
    if runtime_reason == "ok" and ok:
        memory_raw, params_raw, score_reason = score_model(utils, raw, TASK_ID, variant, EXP_DIR)
        if memory_raw is not None and params_raw is not None:
            memory = int(memory_raw)
            params = int(params_raw)
            cost = memory + params
            local_delta = point(cost) - base.points
        ev, _ = evaluate_candidate(
            utils,
            Candidate(TASK_ID, variant, base.route, raw, "generated", "task185 Gather axis lattice extraction"),
            base,
            -1,
            EXP_DIR,
        )
        validation_status = ev.validation_status
        candidate_status = ev.status
        candidate_reason = ev.reason
    return Row(
        variant=variant,
        status="scored" if cost != "" else "rejected",
        node_count=len(model.graph.node),
        memory=memory,
        params=params,
        cost=cost,
        file_bytes=len(raw),
        sha256=sha256(raw),
        static_reason=static_reason,
        score_reason=score_reason,
        runtime_reason=runtime_reason,
        validation_status=validation_status,
        candidate_status=candidate_status,
        candidate_reason=candidate_reason,
        local_delta_if_any=local_delta,
        notes="replace exp330 Tile+GatherElements index tensors with Gather(axis=2/3)",
    )


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()[TASK_ID]
    variants = [("gather_axis_float", build_candidate())]
    rows: list[Row] = []
    for variant, raw in variants:
        (EXP_DIR / f"{variant}.onnx").write_bytes(raw)
        rows.append(eval_raw(utils, base, variant, raw))
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])
    improved_rows = [row for row in rows if row.candidate_status == "improved"]
    best_row = min(rows, key=lambda r: float("inf") if r.cost == "" else int(r.cost))
    best_improved = None if not improved_rows else min(improved_rows, key=lambda r: int(r.cost))
    expected_delta = 0.0 if best_improved is None else float(best_improved.local_delta_if_any)
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-11",
        "status": "candidate_probe_complete",
        "task_id": TASK_ID,
        "hypothesis": "Replacing exp330 Tile+GatherElements dynamic index tensors with ONNX Gather(axis=2/3) will shave enough memory to make task185 cheaper than the exp297 baseline.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "current_public_best_lb": CURRENT_PUBLIC_BEST_LB,
        "rows": [asdict(row) for row in rows],
        "best_row": asdict(best_row),
        "accepted_count": len(improved_rows),
        "best_improved": None if best_improved is None else asdict(best_improved),
        "expected_local_delta": expected_delta,
        "expected_public_lb_if_calibrated": CURRENT_PUBLIC_BEST_LB + expected_delta,
        "decision": (
            "Submit only if bundled delta exceeds the public display threshold."
            if best_improved is not None
            else "No submit. This was the last narrow task185 shave; pivot to task251/task037 compact lowering."
        ),
        "submission_decision": "bundle_later" if best_improved is not None and expected_delta < 0.05 else ("submit_after_review" if best_improved is not None else "no_submit"),
        "leakage_risk": "low: input-only detector/selector and deterministic lattice extraction; no output lookup or public-score-driven constants.",
        "overfitting_risk": "medium-low: task-specific geometry, but all examples must pass full-arc gate and no public repair source is used.",
        "runtime_seconds": time.time() - started,
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp330 の task185 full candidate は `267_pass_0_fail` だが baseline より cost が `+200` 高かった。原因候補である `Tile + GatherElements` の大きな index tensor を、ONNX `Gather(axis=2/3)` に置き換えて cost shave する。

## 仮説

`row_vec` / `col_vec` を rank-1 dynamic index として使えば、`row_idx [1,10,4,30]` と `col_idx [1,10,4,4]` を materialize せずに同じ 4x4 lattice を抽出できる。

## 結果

- baseline_cost: `{base.cost}`
- rows: `{ {r.variant: {'validation': r.validation_status, 'status': r.candidate_status, 'cost': r.cost, 'delta': r.local_delta_if_any} for r in rows} }`
- best_row: `{best_row.variant}`
- best_cost: `{best_row.cost}`
- decision: `{result['decision']}`
- submission_decision: `{result['submission_decision']}`

## 解釈

`Gather(axis=2/3)` が full-arc valid かつ cost 改善なら、task185 は bundle 候補になる。改善しない場合、この task185 レーンは「correctness-complete だが cost が届かない」として一旦停止し、task251/task037 の compact lowering に pivot する。

## Risk

- leakage risk: {result['leakage_risk']}
- overfitting risk: {result['overfitting_risk']}
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
