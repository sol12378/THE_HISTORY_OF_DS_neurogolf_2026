from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import date
from typing import Any

import numpy as np
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (
    BaseTask,
    Candidate,
    evaluate_candidate,
    load_base_tasks,
    load_neurogolf_utils,
    make_model,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp044_task031_dynamic_bbox_lowering"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
OVERLAY_EXPS = [
    ROOT / "experiments" / "exp040_deeper_fullarc_safe_bypass",
    ROOT / "experiments" / "exp041_task145_deeper_mul_chain",
]
OUTPUT_ZIP = EXP_DIR / "submission.zip"
TARGET_TASK = 31
AUTHORITATIVE_BASE_SCORE = 6480.302477938763


@dataclass
class EvalRow:
    task_id: int
    variant: str
    baseline_cost: int
    candidate_cost: int | str
    baseline_points: float
    candidate_points: float | str
    file_bytes: int | str
    validation_status: str
    status: str
    reason: str
    sha256: str


def load_current_best_tasks() -> dict[int, BaseTask]:
    base = load_base_tasks(BASE_EXP)
    for overlay_exp in OVERLAY_EXPS:
        if not (overlay_exp / "selected_manifest.csv").exists():
            continue
        with zipfile.ZipFile(overlay_exp / "submission.zip") as zf:
            overlay_raws = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}
        with (overlay_exp / "selected_manifest.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "improved":
                    continue
                task_id = int(row["task_id"])
                if "candidate_cost" in row:
                    cost = int(float(row["candidate_cost"]))
                    points = float(row["candidate_points"])
                else:
                    cost = int(float(row["cost"]))
                    points = float(row["local_points"])
                base[task_id] = replace(
                    base[task_id],
                    cost=cost,
                    points=points,
                    source=f"{EXP_ID}_base_overlay_{overlay_exp.name}",
                    template_name=row["template_name"],
                    raw=overlay_raws[task_id],
                )
    return base


def c_int(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.int64), name)


def build_dynamic_bbox_candidate(task_id: int, route: str) -> Candidate:
    nodes: list[Any] = []
    inits: list[Any] = []

    # Object mask: any non-background color in channels 1..9.
    inits.extend(
        [
            c_int("ch_starts", np.asarray([0, 1, 0, 0])),
            c_int("ch_ends", np.asarray([1, 10, 30, 30])),
            c_int("axes4", np.asarray([0, 1, 2, 3])),
        ]
    )
    nodes.append(helper.make_node("Slice", ["input", "ch_starts", "ch_ends", "axes4"], ["fg_ch"]))
    nodes.append(helper.make_node("ReduceSum", ["fg_ch"], ["fg_sum"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Greater", ["fg_sum", "zero_f"], ["fg_mask"]))
    nodes.append(helper.make_node("Cast", ["fg_mask"], ["fg_float"], to=TensorProto.FLOAT))

    # row_has/col_has are [1,1,30,1] and [1,1,1,30].
    nodes.append(helper.make_node("ReduceMax", ["fg_float"], ["row_has_4d"], axes=[3], keepdims=1))
    nodes.append(helper.make_node("ReduceMax", ["fg_float"], ["col_has_4d"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("Squeeze", ["row_has_4d"], ["row_has"], axes=[0, 1, 3]))
    nodes.append(helper.make_node("Squeeze", ["col_has_4d"], ["col_has"], axes=[0, 1, 2]))
    nodes.append(helper.make_node("Gather", ["row_has", "rev30"], ["row_rev"], axis=0))
    nodes.append(helper.make_node("Gather", ["col_has", "rev30"], ["col_rev"], axis=0))
    nodes.append(helper.make_node("ArgMax", ["row_has"], ["r0"], axis=0, keepdims=1))
    nodes.append(helper.make_node("ArgMax", ["col_has"], ["c0"], axis=0, keepdims=1))
    nodes.append(helper.make_node("ArgMax", ["row_rev"], ["r_last_rev"], axis=0, keepdims=1))
    nodes.append(helper.make_node("ArgMax", ["col_rev"], ["c_last_rev"], axis=0, keepdims=1))
    nodes.append(helper.make_node("Sub", ["thirty", "r_last_rev"], ["r1"]))
    nodes.append(helper.make_node("Sub", ["thirty", "c_last_rev"], ["c1"]))
    nodes.append(helper.make_node("Sub", ["r1", "r0"], ["height"]))
    nodes.append(helper.make_node("Sub", ["c1", "c0"], ["width"]))

    # Build dynamic source indices for every output cell and gather all channels.
    nodes.append(helper.make_node("Add", ["out_r", "r0"], ["src_r_raw"]))
    nodes.append(helper.make_node("Add", ["out_c", "c0"], ["src_c_raw"]))
    nodes.append(helper.make_node("Less", ["out_r", "height"], ["valid_r"]))
    nodes.append(helper.make_node("Less", ["out_c", "width"], ["valid_c"]))
    nodes.append(helper.make_node("And", ["valid_r", "valid_c"], ["valid_11"]))
    nodes.append(helper.make_node("Add", ["src_r_raw", "zero_ch"], ["src_r_full"]))
    nodes.append(helper.make_node("Add", ["src_c_raw", "zero_ch"], ["src_c_full"]))
    nodes.append(helper.make_node("And", ["valid_11", "valid_ch"], ["valid_full"]))
    nodes.append(helper.make_node("Where", ["valid_full", "src_r_full", "zero_i_full"], ["src_r_safe"]))
    nodes.append(helper.make_node("Where", ["valid_full", "src_c_full", "zero_i_full"], ["src_c_safe"]))
    for name in ["batch_idx", "channel_idx", "src_r_safe", "src_c_safe"]:
        nodes.append(helper.make_node("Unsqueeze", [name], [f"{name}_u"], axes=[4]))
    nodes.append(helper.make_node("Concat", ["batch_idx_u", "channel_idx_u", "src_r_safe_u", "src_c_safe_u"], ["gather_idx"], axis=4))
    nodes.append(helper.make_node("GatherND", ["input", "gather_idx"], ["gathered"]))
    nodes.append(helper.make_node("Cast", ["valid_full"], ["valid_float"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", ["gathered", "valid_float"], ["output"]))

    batch_idx = np.zeros((1, 10, 30, 30), dtype=np.int64)
    channel_idx = np.broadcast_to(np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1), (1, 10, 30, 30)).copy()
    out_r = np.broadcast_to(np.arange(30, dtype=np.int64).reshape(1, 1, 30, 1), (1, 1, 30, 30)).copy()
    out_c = np.broadcast_to(np.arange(30, dtype=np.int64).reshape(1, 1, 1, 30), (1, 1, 30, 30)).copy()
    zero_ch = np.zeros((1, 10, 30, 30), dtype=np.int64)
    valid_ch = np.ones((1, 10, 30, 30), dtype=bool)
    inits.extend(
        [
            numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero_f"),
            c_int("rev30", np.arange(29, -1, -1)),
            c_int("thirty", np.asarray([30])),
            c_int("out_r", out_r),
            c_int("out_c", out_c),
            c_int("zero_ch", zero_ch),
            c_int("zero_i_full", zero_ch),
            numpy_helper.from_array(valid_ch, "valid_ch"),
            c_int("batch_idx", batch_idx),
            c_int("channel_idx", channel_idx),
        ]
    )

    raw = make_model(nodes, inits, EXP_ID, opset_version=11)
    return Candidate(task_id, "dynamic_bbox_nonzero_gathernd", route, raw, "generated", "ArgMax bbox + GatherND shift")


def selected_row(task_id: int, base: BaseTask, improved: EvalRow | None, raw: bytes) -> dict[str, Any]:
    if improved is None:
        return {
            "task_id": task_id,
            "source": base.source,
            "template_name": base.template_name,
            "route": base.route,
            "cost": base.cost,
            "local_points": base.points,
            "file_bytes": len(base.raw),
            "status": "baseline",
            "reason": "no dynamic bbox gain",
            "sha256": sha256(base.raw),
        }
    return {
        "task_id": task_id,
        "source": f"{EXP_ID}_{improved.variant}",
        "template_name": improved.variant,
        "route": base.route,
        "cost": improved.candidate_cost,
        "local_points": improved.candidate_points,
        "file_bytes": len(raw),
        "status": "improved",
        "reason": improved.reason,
        "sha256": sha256(raw),
    }


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_current_best_tasks()
    base = base_tasks[TARGET_TASK]
    candidate = build_dynamic_bbox_candidate(TARGET_TASK, base.route)
    eval_row, raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
    row = EvalRow(
        task_id=eval_row.task_id,
        variant=eval_row.template_name,
        baseline_cost=eval_row.baseline_cost,
        candidate_cost=eval_row.candidate_cost,
        baseline_points=eval_row.baseline_points,
        candidate_points=eval_row.candidate_points,
        file_bytes=eval_row.file_bytes,
        validation_status=eval_row.validation_status,
        status=eval_row.status,
        reason=eval_row.reason,
        sha256=eval_row.sha256,
    )
    improved = row if raw is not None and row.status == "improved" else None
    final_raw = {task_id: task.raw for task_id, task in base_tasks.items()}
    if improved is not None and raw is not None:
        final_raw[TARGET_TASK] = raw
    write_zip(OUTPUT_ZIP, final_raw)

    selected_rows = [selected_row(task_id, base_tasks[task_id], improved if task_id == TARGET_TASK else None, final_raw[task_id]) for task_id in sorted(base_tasks)]
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(EvalRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerow(asdict(row))
    selected_fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected_fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    accepted_delta = 0.0
    if improved is not None:
        accepted_delta = float(improved.candidate_points) - improved.baseline_points
    baseline_score = AUTHORITATIVE_BASE_SCORE
    new_score = baseline_score + accepted_delta
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_below_target" if improved else "no_gain",
        "target_task": TARGET_TASK,
        "baseline_local_estimate": baseline_score,
        "new_local_estimate": new_score,
        "delta": new_score - baseline_score,
        "gap_to_6500": 6500.0 - new_score,
        "gap_to_7000": 7000.0 - new_score,
        "gap_to_7400": 7400.0 - new_score,
        "gap_to_7600": 7600.0 - new_score,
        "gap_to_7700": 7700.0 - new_score,
        "candidate": asdict(row),
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "accounting_note": "Total score uses exp041 authoritative best plus accepted deltas; selected_manifest is a convenience bundle manifest.",
        "leakage_risk": "low: explicit bbox rule, no signature lookup.",
        "overfitting_risk": "low-to-medium: task-specific but validated on all arc-gen.",
        "decision": "adopt if improved; if no gain, dynamic GatherND bbox is too memory-heavy and needs a smaller object compiler.",
    }
    with (EXP_DIR / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    notes = f"""# {EXP_ID}

## 仮説

task031の `bbox_nonzero` はexp042最大hitであり、動的bboxを `ArgMax + GatherND` で閉形式loweringすれば、既存artifact cost 18616を下回れる可能性がある。

## 実験

- 対象: task031
- rule: nonzero bboxを左上へcrop
- lowering: `ReduceSum/ReduceMax/ArgMax` でbboxを検出し、`GatherND` で全channelを左上へshift、bbox外をmaskする
- validation: train + test + all arc-gen

## 結果

- status: {row.status}
- baseline cost: {row.baseline_cost}
- candidate cost: {row.candidate_cost}
- baseline local estimate: {baseline_score:.6f}
- new local estimate: {new_score:.6f}
- delta: {new_score - baseline_score:.6f}
- reason: {row.reason}

## 解釈

この実験はobject/bbox compilerの最初の実cost検証である。改善しない場合、`GatherND` 用の巨大indexテンソルがcostを支配している可能性が高く、bbox専用のより小さいloweringが必要。

## リスク

- leakage risk: 低。signature lookupなし。
- overfitting risk: 低〜中。all arc-gen通過時のみ採用。

## 次

改善しなければ、`GatherND` full-grid方式をguardrailへ入れ、row/colごとのSlice候補選択や既存artifact surgeryとのhybridへ移る。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
