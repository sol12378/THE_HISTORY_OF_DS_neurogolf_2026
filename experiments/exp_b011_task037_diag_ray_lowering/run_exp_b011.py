from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
from onnx import helper, numpy_helper


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    evaluate_candidate,
    examples_for,
    load_neurogolf_utils,
    load_task,
    load_routes,
    make_model,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp_b011_task037_diag_ray_lowering"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp005_top_cost_rewrite_strict"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
STRICT_SEED_SCORE = 6282.230228
TASK_ID = 37


@dataclass(frozen=True)
class LoweringAttempt:
    task_id: int
    template_name: str
    baseline_cost: int
    candidate_cost: int | str
    baseline_points: float
    candidate_points: float | str
    validation_status: str
    status: str
    reason: str
    file_bytes: int | str
    sha256: str


def task_shapes(task_id: int) -> dict[str, Any]:
    task = load_task(task_id)
    inputs = Counter((len(ex["input"]), len(ex["input"][0])) for ex in examples_for(task, -1))
    outputs = Counter((len(ex["output"]), len(ex["output"][0])) for ex in examples_for(task, -1))
    return {"input_shapes": {str(k): v for k, v in inputs.items()}, "output_shapes": {str(k): v for k, v in outputs.items()}}


def load_strict_seed_tasks() -> dict[int, BaseTask]:
    routes = load_routes()
    rows: dict[int, dict[str, str]] = {}
    with (BASE_EXP / "rewrite_manifest.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["task_id"])] = row
    with zipfile.ZipFile(BASE_EXP / "submission.zip") as zf:
        raws = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}
    out: dict[int, BaseTask] = {}
    for task_id, row in rows.items():
        out[task_id] = BaseTask(
            task_id=task_id,
            cost=int(float(row["new_cost"])),
            points=float(row["new_points"]),
            source=row["source"],
            template_name="strict_seed",
            route=routes.get(task_id, ""),
            raw=raws[task_id],
        )
    return out


def diag_weight(name: str, direction: str, kernel: int) -> Any:
    center = kernel // 2
    weight = np.zeros((10, 1, kernel, kernel), dtype=np.float32)
    for color in range(1, 10):
        for d in range(1, center + 1):
            if direction == "nw":
                rr, cc = center - d, center - d
            elif direction == "se":
                rr, cc = center + d, center + d
            elif direction == "ne":
                rr, cc = center - d, center + d
            elif direction == "sw":
                rr, cc = center + d, center - d
            else:
                raise ValueError(direction)
            weight[color, 0, rr, cc] = 1.0
    return numpy_helper.from_array(weight, name)


def build_diag_ray_candidate(task_id: int, base: BaseTask, kernel: int) -> Candidate:
    if kernel % 2 != 1:
        return Candidate(task_id, f"task037_diag_ray_k{kernel}", base.route, None, "skipped", "kernel must be odd")
    pad = kernel // 2
    initializers = [
        diag_weight("w_nw", "nw", kernel),
        diag_weight("w_se", "se", kernel),
        diag_weight("w_ne", "ne", kernel),
        diag_weight("w_sw", "sw", kernel),
        numpy_helper.from_array(np.asarray([0, 0, 0, 0], dtype=np.int64), "starts0"),
        numpy_helper.from_array(np.asarray([1, 1, 30, 30], dtype=np.int64), "ends0"),
        numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "axes4"),
        numpy_helper.from_array(np.asarray([0, 1, 0, 0], dtype=np.int64), "starts_color"),
        numpy_helper.from_array(np.asarray([1, 10, 30, 30], dtype=np.int64), "ends_color"),
        numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), "one"),
        numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero"),
        numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), "half"),
        numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "axes"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axis_color"),
        numpy_helper.from_array(np.asarray([0, 1, 0, 0], dtype=np.int64), "fill_starts"),
        numpy_helper.from_array(np.asarray([1, 10, 30, 30], dtype=np.int64), "fill_ends"),
    ]
    conv_attrs = {"group": 10, "pads": [pad, pad, pad, pad]}
    nodes = [
        helper.make_node("Conv", ["input", "w_nw"], ["nw"], **conv_attrs),
        helper.make_node("Conv", ["input", "w_se"], ["se"], **conv_attrs),
        helper.make_node("Conv", ["input", "w_ne"], ["ne"], **conv_attrs),
        helper.make_node("Conv", ["input", "w_sw"], ["sw"], **conv_attrs),
        helper.make_node("Greater", ["nw", "zero"], ["nw_b"]),
        helper.make_node("Greater", ["se", "zero"], ["se_b"]),
        helper.make_node("Greater", ["ne", "zero"], ["ne_b"]),
        helper.make_node("Greater", ["sw", "zero"], ["sw_b"]),
        helper.make_node("And", ["nw_b", "se_b"], ["diag_a"]),
        helper.make_node("And", ["ne_b", "sw_b"], ["diag_b"]),
        helper.make_node("Or", ["diag_a", "diag_b"], ["diag_any"]),
        helper.make_node("Slice", ["input", "starts0", "ends0", "axes4"], ["background"]),
        helper.make_node("Greater", ["background", "half"], ["background_b"]),
        helper.make_node("Cast", ["background_b"], ["background_f"], to=1),
        helper.make_node("Cast", ["diag_any"], ["diag_f0"], to=1),
        helper.make_node("Mul", ["diag_f0", "background_f"], ["fill_all"]),
        helper.make_node("Slice", ["fill_all", "fill_starts", "fill_ends", "axes4"], ["fill_colors"]),
        helper.make_node("ReduceMax", ["fill_colors"], ["any_fill"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["one", "any_fill"], ["keep_bg"]),
        helper.make_node("Mul", ["background", "keep_bg"], ["out0"]),
        helper.make_node("Slice", ["input", "starts_color", "ends_color", "axes4"], ["input_colors"]),
        helper.make_node("Add", ["input_colors", "fill_colors"], ["out_colors_pre"]),
        helper.make_node("Clip", ["out_colors_pre"], ["out_colors"], min=0.0, max=1.0),
        helper.make_node("Concat", ["out0", "out_colors"], ["output"], axis=1),
    ]
    raw = make_model(nodes, initializers, f"{EXP_ID}_k{kernel}")
    return Candidate(task_id, f"task037_diag_ray_k{kernel}", base.route, raw, "generated", "diag opposite same-color ray fill")


def selected_row(task_id: int, base: BaseTask, eval_row: LoweringAttempt | None, raw: bytes) -> dict[str, Any]:
    if eval_row is None:
        return {
            "task_id": task_id,
            "source": base.source,
            "template_name": base.template_name,
            "route": base.route,
            "cost": base.cost,
            "local_points": base.points,
            "file_bytes": len(base.raw),
            "status": "baseline",
            "reason": "no accepted task037 diag ray lowering",
            "sha256": sha256(base.raw),
        }
    return {
        "task_id": task_id,
        "source": EXP_ID,
        "template_name": eval_row.template_name,
        "route": base.route,
        "cost": eval_row.candidate_cost,
        "local_points": eval_row.candidate_points,
        "file_bytes": len(raw),
        "status": "improved",
        "reason": eval_row.reason,
        "sha256": sha256(raw),
    }


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_strict_seed_tasks()
    base = base_tasks[TASK_ID]
    final_raw = {task_id: task.raw for task_id, task in base_tasks.items()}
    attempts: list[LoweringAttempt] = []
    accepted: tuple[LoweringAttempt, bytes] | None = None

    for kernel in [19, 21, 23, 25, 27, 29]:
        cand = build_diag_ray_candidate(TASK_ID, base, kernel)
        eval_row, raw = evaluate_candidate(utils, cand, base, -1, EXP_DIR)
        attempt = LoweringAttempt(
            task_id=eval_row.task_id,
            template_name=eval_row.template_name,
            baseline_cost=eval_row.baseline_cost,
            candidate_cost=eval_row.candidate_cost,
            baseline_points=eval_row.baseline_points,
            candidate_points=eval_row.candidate_points,
            validation_status=eval_row.validation_status,
            status=eval_row.status,
            reason=eval_row.reason,
            file_bytes=eval_row.file_bytes,
            sha256=eval_row.sha256,
        )
        attempts.append(attempt)
        if raw is not None and attempt.status == "improved":
            if accepted is None or int(attempt.candidate_cost) < int(accepted[0].candidate_cost):
                accepted = (attempt, raw)

    if accepted is not None:
        final_raw[TASK_ID] = accepted[1]
    write_zip(OUTPUT_ZIP, final_raw)

    selected = [
        selected_row(task_id, task, accepted[0] if task_id == TASK_ID and accepted is not None else None, final_raw[task_id])
        for task_id, task in sorted(base_tasks.items())
    ]
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(LoweringAttempt.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in attempts])
    fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    delta = 0.0
    if accepted is not None:
        delta = float(accepted[0].candidate_points) - base.points
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_submit_candidate" if accepted is not None else "no_gain",
        "task_id": TASK_ID,
        "rule_source_exp": "exp_b010_component_object_role_sparse_fill_miner",
        "rule": "opposite_ray_diag_same_color",
        "shape_profile": task_shapes(TASK_ID),
        "baseline_exp": str(BASE_EXP.relative_to(ROOT)),
        "baseline_strict_seed_score": STRICT_SEED_SCORE,
        "baseline_task_cost": base.cost,
        "baseline_task_points": base.points,
        "accepted": asdict(accepted[0]) if accepted is not None else None,
        "new_task_cost": int(accepted[0].candidate_cost) if accepted is not None else base.cost,
        "new_task_points": float(accepted[0].candidate_points) if accepted is not None else base.points,
        "local_estimate_delta": delta,
        "new_strict_seed_local_estimate": STRICT_SEED_SCORE + delta,
        "candidate_status_counts": dict(Counter(row.status for row in attempts)),
        "attempts": [asdict(row) for row in attempts],
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_recommended_for_single_task_delta_calibration" if accepted is not None else "no_submit: no cost gain",
        "leakage_risk": "low: explicit diagonal same-color ray rule, no signature lookup or arc-gen labels in synthesis.",
        "overfitting_risk": "medium: rule passes all local arc-gen; single-task Kaggle delta is needed to calibrate hidden behavior.",
        "decision": "submit single-task delta if accepted; otherwise replace Conv visibility with cheaper diagonal reduction lowering.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp_b010でfull passしたtask037 `opposite_ray_diag_same_color` ruleを実ONNXへloweringし、strict seedより低costになるかを確認する。

## 結果

- baseline task cost: {base.cost}
- baseline task points: {base.points:.6f}
- accepted: {asdict(accepted[0]) if accepted is not None else None}
- local estimate delta: {delta:.6f}
- new strict seed local estimate: {STRICT_SEED_SCORE + delta:.6f}

## 解釈

対角方向の同色端点に挟まれたbackgroundを埋めるruleはlocal all arc-genで正しい。Conv可視性loweringがcost gainを出せば、単一task delta submissionでlocal/LB対応を見る。

## Risk

- leakage risk: low。説明可能ruleで、signature lookupではない。
- overfitting risk: medium。arc-gen full pass済みでもhidden生成差をKaggle deltaで確認する。

## Decision

acceptedがあればsingle-task delta calibrationとして提出候補。なければ、Convではなくdiagonal-specific small reductionへloweringを縮小する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
