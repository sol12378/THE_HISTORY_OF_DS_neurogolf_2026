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

import onnx
from onnx import numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    CandidateEval,
    evaluate_candidate,
    load_neurogolf_utils,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp109_task048_artifact_surgery_sweep"
EXP_DIR = ROOT / "experiments" / EXP_ID
CURRENT_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
CURRENT_LOCAL_ESTIMATE = 6282.812218
TASK_ID = 48


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


def graph_inventory(raw: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(raw)
    init_hashes: dict[str, list[str]] = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        key = f"{arr.dtype}:{arr.shape}:{sha256(arr.tobytes())}"
        init_hashes.setdefault(key, []).append(init.name)
    return {
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "file_bytes": len(raw),
        "op_counts": dict(sorted(__import__("collections").Counter(n.op_type for n in model.graph.node).items())),
        "duplicate_initializer_group_count": sum(1 for names in init_hashes.values() if len(names) > 1),
    }


def value_consumers(model: onnx.ModelProto) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        for inp in node.input:
            counts[inp] = counts.get(inp, 0) + 1
    return counts


def bypass_value(raw: bytes, old: str, new: str, remove_node_idx: int) -> bytes:
    model = onnx.load_model_from_string(raw)
    graph_outputs = {out.name for out in model.graph.output}
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == old:
                node.input[i] = new
    if old not in graph_outputs:
        del model.graph.node[remove_node_idx]
    return model.SerializeToString()


def build_bypass_candidates(base: BaseTask, max_candidates: int = 220) -> list[Candidate]:
    model = onnx.load_model_from_string(base.raw)
    consumers = value_consumers(model)
    candidate_ops = {"And", "Or", "Not", "Where", "Equal", "Greater", "Mul", "Add", "Sub", "Cast"}
    candidates: list[Candidate] = []
    for idx, node in enumerate(model.graph.node):
        if len(node.output) != 1 or node.op_type not in candidate_ops:
            continue
        old = node.output[0]
        if consumers.get(old, 0) == 0:
            continue
        for input_idx, inp in enumerate(node.input):
            if not inp:
                continue
            raw = bypass_value(base.raw, old, inp, idx)
            name = f"bypass_{node.op_type}_{idx:03d}_in{input_idx}"
            reason = f"replace {old} ({node.op_type} node {idx}) with input {input_idx}:{inp}"
            candidates.append(Candidate(TASK_ID, name, base.route, raw, "generated", reason))
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_current_tasks()
    base = base_tasks[TASK_ID]
    inventory = graph_inventory(base.raw)

    candidates = build_bypass_candidates(base)
    evals: list[CandidateEval] = []
    accepted: dict[int, bytes] = {}
    best_eval: CandidateEval | None = None
    for candidate in candidates:
        ev, raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
        evals.append(ev)
        if raw is not None and ev.status == "improved" and (best_eval is None or int(ev.candidate_cost) < int(best_eval.candidate_cost)):
            accepted[TASK_ID] = raw
            best_eval = ev

    final_raw = {task_id: task.raw for task_id, task in base_tasks.items()}
    final_raw.update(accepted)
    write_zip(OUTPUT_ZIP, final_raw)

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(CandidateEval.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([asdict(ev) for ev in evals])

    status_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}
    for ev in evals:
        status_counts[ev.status] = status_counts.get(ev.status, 0) + 1
        validation_counts[ev.validation_status] = validation_counts.get(ev.validation_status, 0) + 1

    delta = 0.0 if best_eval is None else float(best_eval.candidate_points) - float(best_eval.baseline_points)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_bundle_candidate" if accepted else "no_gain",
        "campaign_index": 11,
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "inventory": inventory,
        "generated_candidate_count": len(candidates),
        "evaluated_candidate_count": len(evals),
        "status_counts": status_counts,
        "validation_counts": validation_counts,
        "accepted_tasks": sorted(accepted),
        "accepted_count": len(accepted),
        "best_eval": asdict(best_eval) if best_eval is not None else None,
        "base_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "local_estimate_delta": delta,
        "new_local_estimate": CURRENT_LOCAL_ESTIMATE + delta,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_candidate_after_review" if accepted else "no_submit",
        "decision": "Task048 existing artifact surgery is tested because naive connectivity lowering is above baseline.",
        "leakage_risk": "low: graph surgery only, full validation gated.",
        "overfitting_risk": "low-to-medium: accepted surgery would still need LB calibration.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## Hypothesis

task048 is solved by a bridge connectivity rule, but naive 8x8 connectivity lowering is more expensive than baseline. The current artifact may already contain the right structure with removable generated guards or casts. A focused bypass surgery sweep may produce a safe micro-delta.

## Result

- baseline cost: `{base.cost}`
- inventory: `{inventory}`
- generated candidates: `{len(candidates)}`
- status counts: `{status_counts}`
- accepted tasks: `{sorted(accepted)}`
- local delta: `{delta:.6f}`
- new local estimate: `{CURRENT_LOCAL_ESTIMATE + delta:.6f}`

## Interpretation

If no accepted candidate appears, task048 needs a closed-form bridge feature or a larger subgraph extraction rather than local node bypass.

## Risk

- leakage risk: low。
- overfitting risk: low-to-medium。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
