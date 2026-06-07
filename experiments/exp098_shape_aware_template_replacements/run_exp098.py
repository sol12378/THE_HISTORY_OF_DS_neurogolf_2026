from __future__ import annotations

import csv
import json
import pathlib
import sys
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
    point,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp098_shape_aware_template_replacements"
EXP_DIR = ROOT / "experiments" / EXP_ID
CURRENT_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
CURRENT_LOCAL_ESTIMATE = 6282.812218

TARGETS = [
    (150, "flip_lr"),
    (155, "flip_ud"),
    (87, "rot180"),
    (140, "rot180"),
    (380, "rot90_ccw"),
]


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


def transformed_shape(shape: tuple[int, int], variant: str) -> tuple[int, int]:
    h, w = shape
    if variant in {"flip_lr", "flip_ud", "rot180"}:
        return h, w
    if variant == "rot90_ccw":
        return w, h
    raise ValueError(variant)


def transform_grid(arr: np.ndarray, variant: str) -> np.ndarray:
    if variant == "flip_lr":
        return np.flip(arr, axis=1).copy()
    if variant == "flip_ud":
        return np.flip(arr, axis=0).copy()
    if variant == "rot180":
        return np.flip(np.flip(arr, axis=0), axis=1).copy()
    if variant == "rot90_ccw":
        return np.rot90(arr, k=1).copy()
    raise ValueError(variant)


def shape_inventory(task_id: int, variant: str) -> dict[str, Any]:
    task = load_task(task_id)
    examples = examples_for(task, -1)
    input_shapes = [grid_to_array(ex["input"]).shape for ex in examples]
    output_shapes = [grid_to_array(ex["output"]).shape for ex in examples]
    exact = 0
    for ex in examples:
        pred = transform_grid(grid_to_array(ex["input"]), variant)
        if np.array_equal(pred, grid_to_array(ex["output"])):
            exact += 1
    return {
        "task_id": task_id,
        "variant": variant,
        "n_examples": len(examples),
        "input_shapes": sorted({f"{h}x{w}" for h, w in input_shapes}),
        "output_shapes": sorted({f"{h}x{w}" for h, w in output_shapes}),
        "fixed_input_shape": len(set(input_shapes)) == 1,
        "fixed_output_shape": len(set(output_shapes)) == 1,
        "python_exact": exact,
    }


def build_shape_aware_model(task_id: int, variant: str) -> Candidate:
    inv = shape_inventory(task_id, variant)
    if inv["python_exact"] != inv["n_examples"]:
        return Candidate(task_id, variant, "", None, "skipped", "python transform no longer exact")
    if not inv["fixed_input_shape"] or not inv["fixed_output_shape"]:
        return Candidate(task_id, variant, "", None, "skipped", f"variable shapes: in={inv['input_shapes']} out={inv['output_shapes']}")

    ih, iw = map(int, inv["input_shapes"][0].split("x"))
    oh, ow = map(int, inv["output_shapes"][0].split("x"))
    if transformed_shape((ih, iw), variant) != (oh, ow):
        return Candidate(task_id, variant, "", None, "skipped", f"shape transform mismatch: {(ih, iw)} -> {(oh, ow)}")

    nodes = [helper.make_node("Slice", ["input", "crop_starts", "crop_ends", "crop_axes"], ["crop"])]
    initializers = [
        init_i("crop_starts", [0, 0, 0, 0]),
        init_i("crop_ends", [1, 10, ih, iw]),
        init_i("crop_axes", [0, 1, 2, 3]),
    ]
    current = "crop"

    if variant == "flip_lr":
        initializers.extend([init_i("lr_starts", [-1]), init_i("lr_ends", [-(1 << 31)]), init_i("lr_axes", [3]), init_i("neg_steps", [-1])])
        nodes.append(helper.make_node("Slice", [current, "lr_starts", "lr_ends", "lr_axes", "neg_steps"], ["transformed"]))
        current = "transformed"
    elif variant == "flip_ud":
        initializers.extend([init_i("ud_starts", [-1]), init_i("ud_ends", [-(1 << 31)]), init_i("ud_axes", [2]), init_i("neg_steps", [-1])])
        nodes.append(helper.make_node("Slice", [current, "ud_starts", "ud_ends", "ud_axes", "neg_steps"], ["transformed"]))
        current = "transformed"
    elif variant == "rot180":
        initializers.extend([init_i("r_starts", [-1, -1]), init_i("r_ends", [-(1 << 31), -(1 << 31)]), init_i("r_axes", [2, 3]), init_i("r_steps", [-1, -1])])
        nodes.append(helper.make_node("Slice", [current, "r_starts", "r_ends", "r_axes", "r_steps"], ["transformed"]))
        current = "transformed"
    elif variant == "rot90_ccw":
        initializers.extend([init_i("w_starts", [-1]), init_i("w_ends", [-(1 << 31)]), init_i("w_axes", [3]), init_i("neg_steps", [-1])])
        nodes.append(helper.make_node("Transpose", [current], ["transposed"], perm=[0, 1, 3, 2]))
        nodes.append(helper.make_node("Slice", ["transposed", "w_starts", "w_ends", "w_axes", "neg_steps"], ["transformed"]))
        current = "transformed"

    nodes.append(helper.make_node("Pad", [current], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30 - oh, 30 - ow], value=0.0))
    raw = make_model(nodes, initializers, f"{EXP_ID}_task{task_id:03d}_{variant}", opset_version=10)
    return Candidate(task_id, f"shape_aware_{variant}", "", raw, "generated", f"shape={ih}x{iw}->{oh}x{ow}")


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_current_tasks()

    inventories = [shape_inventory(task_id, variant) for task_id, variant in TARGETS]
    evals: list[CandidateEval] = []
    accepted: dict[int, bytes] = {}
    best_eval_by_task: dict[int, CandidateEval] = {}
    for task_id, variant in TARGETS:
        candidate = build_shape_aware_model(task_id, variant)
        base = base_tasks[task_id]
        candidate = Candidate(task_id, candidate.template_name, base.route, candidate.raw, candidate.status, candidate.reason)
        ev, raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
        evals.append(ev)
        if raw is not None and ev.status == "improved":
            accepted[task_id] = raw
            best_eval_by_task[task_id] = ev

    final_raw = {task_id: task.raw for task_id, task in base_tasks.items()}
    final_raw.update(accepted)
    write_zip(OUTPUT_ZIP, final_raw)

    with (EXP_DIR / "shape_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task_id", "variant", "n_examples", "input_shapes", "output_shapes", "fixed_input_shape", "fixed_output_shape", "python_exact"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in inventories:
            out = dict(row)
            out["input_shapes"] = ";".join(out["input_shapes"])
            out["output_shapes"] = ";".join(out["output_shapes"])
            writer.writerow(out)

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(CandidateEval.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([asdict(ev) for ev in evals])

    delta = sum(float(ev.candidate_points) - float(ev.baseline_points) for ev in best_eval_by_task.values())
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_bundle_candidate" if accepted else "no_gain",
        "targets": TARGETS,
        "evaluated_candidates": len(evals),
        "accepted_tasks": sorted(accepted),
        "accepted_count": len(accepted),
        "accepted_delta": delta,
        "base_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "new_local_estimate": CURRENT_LOCAL_ESTIMATE + delta,
        "candidate_eval": [asdict(ev) for ev in evals],
        "shape_inventory": inventories,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "local_estimate_delta": delta,
        "submission_decision": "submit_candidate_after_review" if accepted else "no_submit",
        "leakage_risk": "low: deterministic geometry transform validated on all available examples.",
        "overfitting_risk": "low for fixed-shape pure geometry; medium if shape constancy is arc-gen-specific.",
        "decision": "Use shape-aware crop/transform/pad only when it beats current submit-safe artifact and remains full-validation exact.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = f"""# {EXP_ID}

## 目的

`neurogolf_templates.zip` 由来のone-node template hitのうち、公式padding semanticsで失敗したflip/rot候補を、active area `Slice -> transform -> Pad` に変えて250〜600級へ入るか検証する。

## 結果

- evaluated candidates: `{len(evals)}`
- accepted tasks: `{sorted(accepted)}`
- accepted delta: `{delta:.6f}`
- new local estimate: `{CURRENT_LOCAL_ESTIMATE + delta:.6f}`

## 解釈

全gridを直接flip/rotするのではなく、固定active shapeを切り出してから変換すればpadding mismatchは避けられる。ただし、`Slice+transform+Pad` の中間tensor costが支配するため、250〜600級に入るかはshape面積に強く依存する。

## Decision

現行submit-safe artifactより安く、かつfull validationを通ったものだけ採用候補にする。改善がなければzip由来template routeは、さらに小さいshapeまたは既存artifact surgery対象に限定する。

## Risk

- leakage risk: low。
- overfitting risk: fixed shapeに依存するためlow〜medium。train/test/arc-gen full validationを通してから判断する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
