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
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp101_archetype_compiler_v1"
EXP_DIR = ROOT / "experiments" / EXP_ID
CURRENT_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"
CAMPAIGN_PROGRESS = ROOT / "experiments" / "compiler_campaign_30" / "progress.md"
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


def fit_global_colormap(examples: list[dict[str, Any]]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if x.shape != y.shape:
            return None
        for src, dst in zip(x.ravel(), y.ravel()):
            src_i, dst_i = int(src), int(dst)
            if src_i in mapping and mapping[src_i] != dst_i:
                return None
            mapping[src_i] = dst_i
    return mapping


def build_channel_gather(task_id: int, route: str, mapping: dict[int, int]) -> Candidate | None:
    inv: dict[int, int] = {}
    for src, dst in mapping.items():
        if dst in inv and inv[dst] != src:
            return None
        inv[dst] = src
    idx = np.asarray([inv.get(c, c) for c in range(10)], dtype=np.int64)
    raw = make_model(
        [helper.make_node("Gather", ["input", "idx"], ["output"], axis=1)],
        [init_i("idx", idx)],
        f"{EXP_ID}_task{task_id:03d}_channel_gather",
        opset_version=11,
    )
    return Candidate(task_id, "archetype_channel_gather", route, raw, "generated", f"mapping={mapping}")


def crop_color_map_for(examples: list[dict[str, Any]], top: int, left: int, oh: int, ow: int) -> tuple[dict[int, int] | None, str]:
    mapping: dict[int, int] = {}
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if y.shape != (oh, ow):
            return None, "output shape mismatch"
        if top + oh > x.shape[0] or left + ow > x.shape[1]:
            return None, "crop outside input"
        crop = x[top : top + oh, left : left + ow]
        for src, dst in zip(crop.ravel(), y.ravel()):
            src_i, dst_i = int(src), int(dst)
            if src_i in mapping and mapping[src_i] != dst_i:
                return None, "conflicting mapping"
            mapping[src_i] = dst_i
    return mapping, "ok"


def find_common_crop(examples: list[dict[str, Any]], allow_colormap: bool) -> tuple[int, int, int, int, dict[int, int] | None] | None:
    candidates: set[tuple[int, int, int, int]] | None = None
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        ih, iw = x.shape
        oh, ow = y.shape
        if oh > ih or ow > iw:
            return None
        local: set[tuple[int, int, int, int]] = set()
        for top in range(ih - oh + 1):
            for left in range(iw - ow + 1):
                crop = x[top : top + oh, left : left + ow]
                if allow_colormap:
                    mapping, _ = crop_color_map_for([ex], top, left, oh, ow)
                    if mapping is not None:
                        local.add((top, left, oh, ow))
                elif np.array_equal(crop, y):
                    local.add((top, left, oh, ow))
        candidates = local if candidates is None else candidates.intersection(local)
        if not candidates:
            return None
    if not candidates:
        return None
    for top, left, oh, ow in sorted(candidates, key=lambda item: (item[2] * item[3], item[0], item[1])):
        if not allow_colormap:
            return top, left, oh, ow, None
        mapping, reason = crop_color_map_for(examples, top, left, oh, ow)
        if mapping is not None:
            return top, left, oh, ow, mapping
    return None


def build_crop_candidate(task_id: int, route: str, top: int, left: int, oh: int, ow: int, mapping: dict[int, int] | None) -> Candidate | None:
    nodes = [helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["crop"])]
    initializers = [
        init_i("starts", [0, 0, top, left]),
        init_i("ends", [1, 10, top + oh, left + ow]),
        init_i("axes", [0, 1, 2, 3]),
    ]
    current = "crop"
    name = "archetype_static_slice_pad"
    reason = f"top={top},left={left},h={oh},w={ow}"
    if mapping is not None:
        inv: dict[int, int] = {}
        for src, dst in mapping.items():
            if dst in inv and inv[dst] != src:
                return None
            inv[dst] = src
        idx = np.asarray([inv.get(c, c) for c in range(10)], dtype=np.int64)
        initializers.append(init_i("idx", idx))
        nodes.append(helper.make_node("Gather", [current, "idx"], ["colored"], axis=1))
        current = "colored"
        name = "archetype_crop_channel_gather_pad"
        reason += f",mapping={mapping}"
    nodes.append(helper.make_node("Pad", [current], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 30 - oh, 30 - ow], value=0.0))
    raw = make_model(nodes, initializers, f"{EXP_ID}_task{task_id:03d}_{name}", opset_version=10)
    return Candidate(task_id, name, route, raw, "generated", reason)


def build_candidates(task_id: int, base: BaseTask) -> list[Candidate]:
    examples = examples_for(load_task(task_id), -1)
    candidates: list[Candidate] = []
    mapping = fit_global_colormap(examples)
    if mapping is not None:
        candidate = build_channel_gather(task_id, base.route, mapping)
        if candidate is not None:
            candidates.append(candidate)
    exact_crop = find_common_crop(examples, allow_colormap=False)
    if exact_crop is not None:
        top, left, oh, ow, _ = exact_crop
        candidate = build_crop_candidate(task_id, base.route, top, left, oh, ow, None)
        if candidate is not None:
            candidates.append(candidate)
    mapped_crop = find_common_crop(examples, allow_colormap=True)
    if mapped_crop is not None:
        top, left, oh, ow, crop_mapping = mapped_crop
        if crop_mapping is not None:
            candidate = build_crop_candidate(task_id, base.route, top, left, oh, ow, crop_mapping)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_current_tasks()
    evals: list[CandidateEval] = []
    accepted: dict[int, bytes] = {}
    best_eval_by_task: dict[int, CandidateEval] = {}
    generated_counts: dict[int, int] = {}

    # These archetypes are cheap enough to scan all tasks and are full-validation gated.
    for task_id in sorted(base_tasks):
        base = base_tasks[task_id]
        candidates = build_candidates(task_id, base)
        generated_counts[task_id] = len(candidates)
        for candidate in candidates:
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

    accepted_rows = [asdict(best_eval_by_task[t]) for t in sorted(best_eval_by_task)]
    delta = sum(float(row["candidate_points"]) - float(row["baseline_points"]) for row in accepted_rows)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_bundle_candidate" if accepted else "no_gain",
        "campaign_index": 3,
        "base_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "local_estimate_delta": delta,
        "new_local_estimate": CURRENT_LOCAL_ESTIMATE + delta,
        "generated_candidate_count": len(evals),
        "tasks_with_candidates": sum(1 for n in generated_counts.values() if n),
        "accepted_tasks": sorted(accepted),
        "accepted_count": len(accepted),
        "accepted": accepted_rows,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_candidate_after_review" if accepted else "no_submit",
        "decision": "Static archetype compiler v1 either contributes accepted low-risk deltas or proves that current best already covers simple channel/crop archetypes.",
        "leakage_risk": "low-to-medium: generated from simple global/crop relations fit on all available examples; avoid raw lookup and require full validation.",
        "overfitting_risk": "medium: all-example fitting may still overfit task-specific constants; LB calibration required for any accepted bundle.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp100で抽出した安全寄りarchetype (`channel_gather`, `static_slice_pad`, `crop+channel_gather+pad`) を全taskに流し、current submit-safe bundleより低costな候補が出るか測る。

## 結果

- generated candidates: `{len(evals)}`
- tasks with candidates: `{sum(1 for n in generated_counts.values() if n)}`
- accepted tasks: `{sorted(accepted)}`
- local delta: `{delta:.6f}`
- new local estimate: `{CURRENT_LOCAL_ESTIMATE + delta:.6f}`

## Decision

acceptedが空なら、単純global color-map/common cropは現行bestにほぼ吸収済み。次は `one_node_conv_kernel` と `computed_slice_pad` の重み/shape miningへ進む。

## Risk

- leakage risk: low-to-medium。
- overfitting risk: medium。採用候補が出た場合はLB較正対象。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")

    if CAMPAIGN_PROGRESS.exists():
        with CAMPAIGN_PROGRESS.open("a", encoding="utf-8") as f:
            f.write(f"\n| 3 | exp101_archetype_compiler_v1 | channel_gather/static_slice_pad/crop+colormapを全taskへ適用 | {delta:.6f} | {CURRENT_LOCAL_ESTIMATE + delta:.6f} | accepted {sorted(accepted)} |\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
