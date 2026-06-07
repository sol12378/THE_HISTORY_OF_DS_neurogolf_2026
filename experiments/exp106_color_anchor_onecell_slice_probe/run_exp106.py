from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import date
from typing import Any

import numpy as np
from onnx import helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    CandidateEval,
    evaluate_candidate,
    examples_for,
    grid_to_array,
    load_neurogolf_utils,
    load_task,
    make_model,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp106_color_anchor_onecell_slice_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
CURRENT_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"
SMALL_OUTPUT_CSV = ROOT / "experiments" / "exp087_small_output_crop_candidate_scan" / "small_output_candidates.csv"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
CURRENT_LOCAL_ESTIMATE = 6282.812218


def init_i(name: str, values: list[int] | np.ndarray) -> Any:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


def load_current_tasks() -> dict[int, BaseTask]:
    rows: dict[int, dict[str, str]] = {}
    with (CURRENT_EXP / "selected_manifest.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["task_id"])] = row
    with zipfile.ZipFile(CURRENT_EXP / "submission.zip") as zf:
        raws = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}
    tasks: dict[int, BaseTask] = {}
    for task_id, row in rows.items():
        tasks[task_id] = BaseTask(
            task_id=task_id,
            cost=int(float(row["cost"])),
            points=float(row["local_points"]),
            source=row["source"],
            template_name=row["template_name"],
            route=row.get("route", ""),
            raw=raws[task_id],
        )
    return tasks


def load_onecell_targets() -> list[int]:
    targets: list[int] = []
    with SMALL_OUTPUT_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["output_shape_hist"].strip() != "1x1:" + row["example_count"]:
                continue
            if float(row["gain_to_600"]) <= 0:
                continue
            targets.append(int(row["task_id"]))
    return targets


def color_bbox_min(x: np.ndarray, color: int) -> tuple[int, int] | None:
    rr, cc = np.where(x == color)
    if len(rr) == 0:
        return None
    return int(rr.min()), int(cc.min())


def fit_color_anchor_offset(examples: list[dict[str, Any]], max_delta: int = 5) -> tuple[int, int, int, str] | None:
    for color in range(1, 10):
        for dr in range(-max_delta, max_delta + 1):
            for dc in range(-max_delta, max_delta + 1):
                ok = True
                for ex in examples:
                    x = grid_to_array(ex["input"])
                    y = grid_to_array(ex["output"])
                    if y.shape != (1, 1):
                        ok = False
                        break
                    anchor = color_bbox_min(x, color)
                    if anchor is None:
                        ok = False
                        break
                    rr, cc = anchor[0] + dr, anchor[1] + dc
                    if rr < 0 or cc < 0 or rr >= x.shape[0] or cc >= x.shape[1]:
                        ok = False
                        break
                    if int(x[rr, cc]) != int(y[0, 0]):
                        ok = False
                        break
                if ok:
                    return color, dr, dc, f"color={color};bbox_min_offset=({dr},{dc})"
    return None


def build_color_anchor_candidate(task_id: int, route: str, color: int, dr: int, dc: int) -> Candidate:
    nodes = [
        helper.make_node("Slice", ["input", "anchor_starts", "anchor_ends", "axes"], ["anchor_mask"]),
        helper.make_node("ReduceSum", ["anchor_mask"], ["row_sum"], axes=[3], keepdims=0),
        helper.make_node("ArgMax", ["row_sum"], ["row_arg"], axis=2, keepdims=0),
        helper.make_node("ReduceSum", ["anchor_mask"], ["col_sum"], axes=[2], keepdims=0),
        helper.make_node("ArgMax", ["col_sum"], ["col_arg"], axis=2, keepdims=0),
        helper.make_node("Reshape", ["row_arg", "shape1"], ["row0"]),
        helper.make_node("Reshape", ["col_arg", "shape1"], ["col0"]),
        helper.make_node("Add", ["row0", "dr"], ["row_start"]),
        helper.make_node("Add", ["col0", "dc"], ["col_start"]),
        helper.make_node("Add", ["row_start", "one"], ["row_end"]),
        helper.make_node("Add", ["col_start", "one"], ["col_end"]),
        helper.make_node("Concat", ["zero", "zero", "row_start", "col_start"], ["crop_starts"], axis=0),
        helper.make_node("Concat", ["one", "ten", "row_end", "col_end"], ["crop_ends"], axis=0),
        helper.make_node("Slice", ["input", "crop_starts", "crop_ends", "axes"], ["cell"]),
        helper.make_node(
            "Pad",
            ["cell"],
            ["output"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 29, 29],
            value=0.0,
        ),
    ]
    initializers = [
        init_i("anchor_starts", [0, color, 0, 0]),
        init_i("anchor_ends", [1, color + 1, 30, 30]),
        init_i("axes", [0, 1, 2, 3]),
        init_i("shape1", [1]),
        init_i("zero", [0]),
        init_i("one", [1]),
        init_i("ten", [10]),
        init_i("dr", [dr]),
        init_i("dc", [dc]),
    ]
    raw = make_model(nodes, initializers, f"{EXP_ID}_task{task_id:03d}_color_anchor", opset_version=10)
    return Candidate(task_id, "color_bbox_min_onecell_slice_pad", route, raw, "generated", f"color={color};offset=({dr},{dc})")


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_current_tasks()
    targets = load_onecell_targets()
    fit_rows: list[dict[str, Any]] = []
    evals: list[CandidateEval] = []
    accepted: dict[int, bytes] = {}
    best_eval_by_task: dict[int, CandidateEval] = {}

    for task_id in targets:
        base = base_tasks[task_id]
        examples = examples_for(load_task(task_id), -1)
        fit = fit_color_anchor_offset(examples)
        if fit is None:
            fit_rows.append({"task_id": task_id, "fit": False, "reason": "no color bbox-min onecell offset"})
            continue
        color, dr, dc, reason = fit
        fit_rows.append({"task_id": task_id, "fit": True, "reason": reason})
        candidate = build_color_anchor_candidate(task_id, base.route, color, dr, dc)
        ev, raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
        evals.append(ev)
        if raw is None or ev.status != "improved":
            continue
        current = best_eval_by_task.get(task_id)
        if current is None or int(ev.candidate_cost) < int(current.candidate_cost):
            accepted[task_id] = raw
            best_eval_by_task[task_id] = ev

    final_raw = {task_id: task.raw for task_id, task in base_tasks.items()}
    final_raw.update(accepted)
    write_zip(OUTPUT_ZIP, final_raw)

    with (EXP_DIR / "fit_scan.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "fit", "reason"])
        writer.writeheader()
        writer.writerows(fit_rows)

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(CandidateEval.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([asdict(ev) for ev in evals])

    accepted_rows = [asdict(best_eval_by_task[t]) for t in sorted(best_eval_by_task)]
    delta = sum(float(row["candidate_points"]) - float(row["baseline_points"]) for row in accepted_rows)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_bundle_candidate" if accepted else "no_gain",
        "campaign_index": 8,
        "base_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "local_estimate_delta": delta,
        "new_local_estimate": CURRENT_LOCAL_ESTIMATE + delta,
        "target_count": len(targets),
        "fit_count": sum(1 for row in fit_rows if row["fit"]),
        "generated_candidate_count": len(evals),
        "accepted_tasks": sorted(accepted),
        "accepted_count": len(accepted),
        "accepted": accepted_rows,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_candidate_after_review" if accepted else "no_submit",
        "decision": "Color-specific anchor is a tighter tiny_dynamic_shape_index probe than all-nonzero bbox-min.",
        "leakage_risk": "medium: color and offset are fit from all examples but the rule is small and input-derived.",
        "overfitting_risk": "medium: accepted candidates still require visual rule inspection before submit.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## Hypothesis

exp105 showed that all-nonzero bbox-min is too coarse for 1x1 P0 outputs. A color-specific bbox-min anchor may isolate the relevant marker/object and allow a small `Slice(channel c) -> ArgMax -> Slice(one cell) -> Pad` lowering.

## Result

- targets: `{len(targets)}`
- color-anchor fits: `{sum(1 for row in fit_rows if row["fit"])}`
- generated candidates: `{len(evals)}`
- accepted tasks: `{sorted(accepted)}`
- local delta: `{delta:.6f}`
- new local estimate: `{CURRENT_LOCAL_ESTIMATE + delta:.6f}`

## Interpretation

This probes a concrete `tiny_dynamic_shape_index` variant. If fit remains 0, 1x1 P0 outputs are not simple marker-neighbor extraction and should move to rule-specific diagnostics.

## Risk

- leakage risk: medium。
- overfitting risk: medium。色+offsetの小ruleでも、視覚的説明がない場合はsubmitしない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
