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


EXP_ID = "exp104_p0_fixed_3x3_static_index_probe"
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


def load_targets(limit: int = 80) -> list[int]:
    targets: list[int] = []
    with SMALL_OUTPUT_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["max_output_area"]) > 9:
                continue
            if int(row["grid_size_changed_examples"]) <= 0:
                continue
            if float(row["gain_to_600"]) <= 0:
                continue
            targets.append(int(row["task_id"]))
            if len(targets) >= limit:
                break
    return targets


def fit_static_position_map(examples: list[dict[str, Any]]) -> tuple[list[int], tuple[int, int], str] | None:
    first_y = grid_to_array(examples[0]["output"])
    oh, ow = first_y.shape
    if oh * ow > 9:
        return None
    maps: list[int] = []
    for out_r in range(oh):
        for out_c in range(ow):
            possible: set[tuple[int, int]] | None = None
            for ex in examples:
                x = grid_to_array(ex["input"])
                y = grid_to_array(ex["output"])
                if y.shape != (oh, ow):
                    return None
                target = int(y[out_r, out_c])
                local = set(zip(*np.where(x == target)))
                possible = local if possible is None else possible.intersection(local)
                if not possible:
                    return None
            coord = sorted(possible)[0]
            maps.append(coord[0] * 30 + coord[1])
    return maps, (oh, ow), f"shape={oh}x{ow};indices={maps}"


def build_static_index_candidate(task_id: int, route: str, indices: list[int], shape: tuple[int, int], reason: str) -> Candidate:
    oh, ow = shape
    nodes = [
        helper.make_node("Reshape", ["input", "flat_shape"], ["flat"]),
        helper.make_node("Gather", ["flat", "idx"], ["picked"], axis=2),
        helper.make_node("Reshape", ["picked", "small_shape"], ["small"]),
        helper.make_node(
            "Pad",
            ["small"],
            ["output"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 30 - oh, 30 - ow],
            value=0.0,
        ),
    ]
    initializers = [
        init_i("flat_shape", [1, 10, 900]),
        init_i("idx", indices),
        init_i("small_shape", [1, 10, oh, ow]),
    ]
    raw = make_model(nodes, initializers, f"{EXP_ID}_task{task_id:03d}_static_index", opset_version=10)
    return Candidate(task_id, "static_position_gather_pad", route, raw, "generated", reason)


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_current_tasks()
    targets = load_targets()
    evals: list[CandidateEval] = []
    fit_rows: list[dict[str, Any]] = []
    accepted: dict[int, bytes] = {}
    best_eval_by_task: dict[int, CandidateEval] = {}

    for task_id in targets:
        base = base_tasks[task_id]
        examples = examples_for(load_task(task_id), -1)
        fit = fit_static_position_map(examples)
        if fit is None:
            fit_rows.append({"task_id": task_id, "fit": False, "reason": "no fixed per-cell source coordinate"})
            continue
        indices, shape, reason = fit
        fit_rows.append({"task_id": task_id, "fit": True, "reason": reason})
        candidate = build_static_index_candidate(task_id, base.route, indices, shape, reason)
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

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(CandidateEval.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([asdict(ev) for ev in evals])

    with (EXP_DIR / "fit_scan.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "fit", "reason"])
        writer.writeheader()
        writer.writerows(fit_rows)

    accepted_rows = [asdict(best_eval_by_task[t]) for t in sorted(best_eval_by_task)]
    delta = sum(float(row["candidate_points"]) - float(row["baseline_points"]) for row in accepted_rows)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_bundle_candidate" if accepted else "no_gain",
        "campaign_index": 6,
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
        "decision": "Static index map is a real compiler emission lane, but only useful when fixed per-cell coordinates both fit and beat the current artifact.",
        "leakage_risk": "medium: coordinates are inferred from all available examples; no raw output lookup is used, but task-specific constants can overfit.",
        "overfitting_risk": "medium-to-high for accepted candidates unless the index map is explainable by a visible geometric rule.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## Hypothesis

P0 small-output tasks may hide a very cheap `one_node_gather_index_map` / `computed_slice_pad` style lowering: if every output cell is copied from a fixed input coordinate across all examples, `Reshape -> Gather(axis=2) -> Reshape -> Pad` can replace expensive crop/shape artifacts.

## Result

- targets: `{len(targets)}`
- fixed-position fits: `{sum(1 for row in fit_rows if row["fit"])}`
- generated candidates: `{len(evals)}`
- accepted tasks: `{sorted(accepted)}`
- local delta: `{delta:.6f}`
- new local estimate: `{CURRENT_LOCAL_ESTIMATE + delta:.6f}`

## Interpretation

This is an actual candidate-emission experiment, not a catalog. If accepted is empty, fixed absolute index maps are too brittle or already beaten by current artifacts; the next compiler lane should make the index dynamic from shape/object anchors rather than hard-code coordinates.

## Risk

- leakage risk: medium。
- overfitting risk: medium-to-high。acceptedが出ても、座標が視覚的ruleで説明できるか確認してからsubmit判断する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
