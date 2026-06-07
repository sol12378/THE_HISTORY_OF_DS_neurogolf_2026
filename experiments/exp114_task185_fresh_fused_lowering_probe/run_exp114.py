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
    BaseTask,
    Candidate,
    CandidateEval,
    evaluate_candidate,
    infer_static_ok,
    load_base_tasks,
    load_neurogolf_utils,
    score_model,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp114_task185_fresh_fused_lowering_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185
CURRENT_LOCAL_ESTIMATE = 6282.93615685804
CAMPAIGN_INDEX = 16
CURRENT_BUNDLE = ROOT / "experiments" / "exp111_focused_surgery_second_pass" / "submission.zip"
OUTPUT_ZIP = EXP_DIR / "submission.zip"


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
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(
        graph,
        producer_name=name,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 11)],
    )
    return model.SerializeToString()


def model_slice_conv_pad() -> bytes:
    weight = np.zeros((10, 1, 2, 2), dtype=np.float32)
    weight[:, 0, :, :] = 1.0
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["m4"]),
        helper.make_node("Conv", ["m4", "w"], ["sum2"], group=10),
        helper.make_node("Equal", ["sum2", "four"], ["eq4"]),
        helper.make_node("Cast", ["eq4"], ["small"], to=TensorProto.FLOAT),
        helper.make_node("Pad", ["small", "pads", "zero"], ["output"], mode="constant"),
    ]
    inits = [
        init_i("starts", np.asarray([0, 0, 5, 5])),
        init_i("ends", np.asarray([1, 10, 17, 17])),
        init_i("axes", np.asarray([0, 1, 2, 3])),
        init_i("steps", np.asarray([1, 1, 3, 3])),
        init_f("w", weight),
        init_f("four", np.asarray([4.0])),
        init_i("pads", np.asarray([0, 0, 0, 0, 0, 0, 27, 27])),
        init_f("zero", np.asarray([0.0])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_slice_conv_pad")


def model_direct_four_slice_mul_pad() -> bytes:
    nodes = [
        helper.make_node("Slice", ["input", "s00", "e00", "axes", "steps"], ["a"]),
        helper.make_node("Slice", ["input", "s01", "e01", "axes", "steps"], ["b"]),
        helper.make_node("Slice", ["input", "s10", "e10", "axes", "steps"], ["c"]),
        helper.make_node("Slice", ["input", "s11", "e11", "axes", "steps"], ["d"]),
        helper.make_node("Mul", ["a", "b"], ["ab"]),
        helper.make_node("Mul", ["c", "d"], ["cd"]),
        helper.make_node("Mul", ["ab", "cd"], ["small"]),
        helper.make_node("Pad", ["small", "pads", "zero"], ["output"], mode="constant"),
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
        init_i("pads", np.asarray([0, 0, 0, 0, 0, 0, 27, 27])),
        init_f("zero", np.asarray([0.0])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_direct_four_slice_mul_pad")


def model_gathernd_lattice_conv_pad() -> bytes:
    # Gather the 16 lattice cells as NHWC color vectors, transpose to NCHW 4x4,
    # then reuse the grouped 2x2 Conv core. This replaces strided Slice with a
    # sparse coordinate extraction to test whether memory beats params.
    coords = np.asarray([[0, r, c] for r in [5, 8, 11, 14] for c in [5, 8, 11, 14]], dtype=np.int64)
    weight = np.zeros((10, 1, 2, 2), dtype=np.float32)
    weight[:, 0, :, :] = 1.0
    nodes = [
        helper.make_node("Transpose", ["input"], ["nhwc"], perm=[0, 2, 3, 1]),
        helper.make_node("GatherND", ["nhwc", "coords"], ["flat16"]),
        helper.make_node("Reshape", ["flat16", "shape16"], ["m4_nhwc"]),
        helper.make_node("Transpose", ["m4_nhwc"], ["m4"], perm=[0, 3, 1, 2]),
        helper.make_node("Conv", ["m4", "w"], ["sum2"], group=10),
        helper.make_node("Equal", ["sum2", "four"], ["eq4"]),
        helper.make_node("Cast", ["eq4"], ["small"], to=TensorProto.FLOAT),
        helper.make_node("Pad", ["small", "pads", "zero"], ["output"], mode="constant"),
    ]
    inits = [
        init_i("coords", coords),
        init_i("shape16", np.asarray([1, 4, 4, 10])),
        init_f("w", weight),
        init_f("four", np.asarray([4.0])),
        init_i("pads", np.asarray([0, 0, 0, 0, 0, 0, 27, 27])),
        init_f("zero", np.asarray([0.0])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_gathernd_lattice_conv_pad")


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


def load_current_tasks() -> dict[int, BaseTask]:
    base_tasks = load_base_tasks()
    import zipfile

    with zipfile.ZipFile(CURRENT_BUNDLE) as zf:
        raws = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}
    return {
        task_id: BaseTask(
            task_id=task_id,
            cost=task.cost,
            points=task.points,
            source=task.source,
            template_name=task.template_name,
            route=task.route,
            raw=raws[task_id],
        )
        for task_id, task in base_tasks.items()
    }


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    current_tasks = load_current_tasks()
    base = current_tasks[TASK_ID]
    builders = [
        ("slice_conv_pad_final", model_slice_conv_pad, "Static lattice Slice + grouped Conv core + Pad to final official output shape."),
        ("direct_four_slice_mul_pad_final", model_direct_four_slice_mul_pad, "Four direct strided Slice views + Mul equality + Pad; avoids 4x4 m4 materialization."),
        ("gathernd_lattice_conv_pad_final", model_gathernd_lattice_conv_pad, "Sparse GatherND lattice extraction + grouped Conv + Pad; tests params-vs-memory tradeoff."),
    ]
    rows = []
    raw_by_variant: dict[str, bytes] = {}
    for variant, build, notes in builders:
        raw = build()
        raw_by_variant[variant] = raw
        (EXP_DIR / f"{variant}.onnx").write_bytes(raw)
        rows.append(score_raw(utils, variant, raw, notes))

    evals: list[CandidateEval] = []
    accepted: dict[int, bytes] = {}
    best_eval: CandidateEval | None = None
    for row in rows:
        if row.status != "scored":
            continue
        candidate = Candidate(
            TASK_ID,
            str(row.variant),
            base.route,
            raw_by_variant[str(row.variant)],
            "generated",
            str(row.notes),
        )
        ev, raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
        evals.append(ev)
        if raw is not None and ev.status == "improved" and (best_eval is None or int(ev.candidate_cost) < int(best_eval.candidate_cost)):
            best_eval = ev
            accepted[TASK_ID] = raw

    final_raw = {task_id: task.raw for task_id, task in current_tasks.items()}
    final_raw.update(accepted)
    write_zip(OUTPUT_ZIP, final_raw)

    with (EXP_DIR / "cost_probe.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CandidateEval.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(ev) for ev in evals])

    status_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}
    for ev in evals:
        status_counts[ev.status] = status_counts.get(ev.status, 0) + 1
        validation_counts[ev.validation_status] = validation_counts.get(ev.validation_status, 0) + 1

    scored = [row for row in rows if row.status == "scored"]
    best = min(scored, key=lambda row: int(row.cost)) if scored else None
    delta = 0.0 if best_eval is None else float(best_eval.candidate_points) - float(best_eval.baseline_points)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "cost_probe_ready",
        "campaign_index": CAMPAIGN_INDEX,
        "task_id": TASK_ID,
        "hypothesis": "Task185 fresh fused lowering may beat prior core proxies when final output shape and extraction strategy are measured together.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "rows": [asdict(row) for row in rows],
        "best_variant": None if best is None else asdict(best),
        "evaluated_candidate_count": len(evals),
        "candidate_status_counts": status_counts,
        "validation_counts": validation_counts,
        "accepted_tasks": sorted(accepted),
        "accepted_count": len(accepted),
        "best_eval": asdict(best_eval) if best_eval is not None else None,
        "base_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "local_estimate_delta": delta,
        "new_local_estimate": CURRENT_LOCAL_ESTIMATE + delta,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "decision": "Use this cost floor to choose the next real correctness candidate. If all variants remain >600, task185 cannot be solved by ordinary Slice/GatherND plus equality; seek a more fused artifact-specific representation.",
        "submission_decision": "submit_candidate_after_review" if accepted else "no_submit",
        "leakage_risk": "low: no label/output lookup.",
        "overfitting_risk": "medium: static lattice position proxy; correctness against all examples is not claimed.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## Hypothesis

task185 fresh fused loweringで、最終official output shapeまで含めたcost floorを測る。`Slice` extraction, direct shifted `Slice+Mul`, sparse `GatherND` extractionを比較する。

## Result

- rows: see `result.json` / `cost_probe.csv`
- best variant: `{None if best is None else best.variant}`
- best cost: `{None if best is None else best.cost}`
- validation counts: `{validation_counts}`
- accepted tasks: `{sorted(accepted)}`
- local delta: `{delta:.6f}`

## Interpretation

これはcost probeであり提出候補ではない。600級に届くvariantがあればfull correctness loweringへ進む。届かない場合は、標準ONNX primitiveの組み合わせではtask185の250〜600化が難しく、よりfusedな表現または別taskへ優先度を移す。

## Risk

- leakage risk: low。
- overfitting risk: medium。static lattice位置proxyで、全例correctnessは主張しない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
