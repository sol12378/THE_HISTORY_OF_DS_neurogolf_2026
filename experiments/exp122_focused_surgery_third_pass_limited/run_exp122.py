from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from dataclasses import asdict

import onnx

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    CandidateEval,
    evaluate_candidate,
    load_neurogolf_utils,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp122_focused_surgery_third_pass_limited"
EXP_DIR = ROOT / "experiments" / EXP_ID
SEED_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"
PREV_EXPS = [
    ROOT / "experiments" / "exp109_task048_artifact_surgery_sweep",
    ROOT / "experiments" / "exp110_focused_surgery_rulehit_cropish_sweep",
    ROOT / "experiments" / "exp111_focused_surgery_second_pass",
]
BASE_ZIP_EXP = PREV_EXPS[-1]
OUTPUT_ZIP = EXP_DIR / "submission.zip"
CURRENT_LOCAL_ESTIMATE = 6282.93615685804


def load_seed_tasks() -> dict[int, BaseTask]:
    rows: dict[int, dict[str, str]] = {}
    with (SEED_EXP / "selected_manifest.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["task_id"])] = row
    with zipfile.ZipFile(BASE_ZIP_EXP / "submission.zip") as zf:
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


def overlay_previous_accepts(tasks: dict[int, BaseTask]) -> list[int]:
    accepted_ids: list[int] = []
    for exp_dir in PREV_EXPS:
        result = json.loads((exp_dir / "result.json").read_text(encoding="utf-8"))
        accepted_rows = []
        if result.get("best_eval"):
            accepted_rows.append(result["best_eval"])
        accepted_rows.extend(result.get("accepted", []))
        with zipfile.ZipFile(exp_dir / "submission.zip") as zf:
            for row in accepted_rows:
                task_id = int(row["task_id"])
                raw = zf.read(f"task{task_id:03d}.onnx")
                old = tasks[task_id]
                tasks[task_id] = BaseTask(
                    task_id=task_id,
                    cost=int(row["candidate_cost"]),
                    points=float(row["candidate_points"]),
                    source=f"{old.source}+{exp_dir.name}",
                    template_name=str(row["template_name"]),
                    route=old.route,
                    raw=raw,
                )
                if task_id not in accepted_ids:
                    accepted_ids.append(task_id)
    return sorted(accepted_ids)


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


def build_bypass_candidates(base: BaseTask, max_candidates: int = 160) -> list[Candidate]:
    model = onnx.load_model_from_string(base.raw)
    consumers = value_consumers(model)
    candidate_ops = {"Cast", "And", "Or", "Not", "Where", "Equal", "Greater", "Mul", "Add", "Sub"}
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
            name = f"third_bypass_{node.op_type}_{idx:03d}_in{input_idx}"
            reason = f"third pass replace {old} ({node.op_type} node {idx}) with input {input_idx}:{inp}"
            candidates.append(Candidate(base.task_id, name, base.route, raw, "generated", reason))
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_seed_tasks()
    previous_accepted_ids = overlay_previous_accepts(base_tasks)
    targets = previous_accepted_ids

    evals: list[CandidateEval] = []
    accepted: dict[int, bytes] = {}
    best_eval_by_task: dict[int, CandidateEval] = {}
    generated_by_task: dict[int, int] = {}
    for task_id in targets:
        base = base_tasks[task_id]
        candidates = build_bypass_candidates(base)
        generated_by_task[task_id] = len(candidates)
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

    status_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}
    for ev in evals:
        status_counts[ev.status] = status_counts.get(ev.status, 0) + 1
        validation_counts[ev.validation_status] = validation_counts.get(ev.validation_status, 0) + 1

    accepted_rows = [asdict(best_eval_by_task[t]) for t in sorted(best_eval_by_task)]
    delta = sum(float(row["candidate_points"]) - float(row["baseline_points"]) for row in accepted_rows)
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-07",
        "status": "improved_bundle_candidate" if accepted else "no_gain",
        "campaign_index": 24,
        "base_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "local_estimate_delta": delta,
        "new_local_estimate": CURRENT_LOCAL_ESTIMATE + delta,
        "target_count": len(targets),
        "targets": targets,
        "previous_accepted_ids": previous_accepted_ids,
        "generated_by_task": generated_by_task,
        "generated_candidate_count": len(evals),
        "status_counts": status_counts,
        "validation_counts": validation_counts,
        "accepted_tasks": sorted(accepted),
        "accepted_count": len(accepted),
        "accepted": accepted_rows,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_candidate_after_review" if accepted else "no_submit",
        "decision": "Limited third-pass focused bypass surgery tests whether accepted-task compounding remains score-positive after exp111.",
        "leakage_risk": "low: graph surgery only and full-validation gated.",
        "overfitting_risk": "low-to-medium: repeated edits on the same public-valid tasks should be LB-calibrated before trusting the gain.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## Hypothesis

exp109〜exp111で受理されたtaskだけに絞れば、focused bypass surgeryの三巡目でも探索ノイズを抑えつつ、残った冗長guard/cast/arithmeticを追加で削れる可能性がある。

## Result

- targets: `{len(targets)}`
- generated candidates: `{len(evals)}`
- status counts: `{status_counts}`
- accepted tasks: `{sorted(accepted)}`
- local delta: `{delta:.6f}`
- new local estimate: `{CURRENT_LOCAL_ESTIMATE + delta:.6f}`

## Decision

三巡目でdeltaが出れば、focused surgeryは提出候補bundleのpost-passとして継続する。deltaが0なら、#25 reviewでこのlaneを補助へ下げ、別familyのfresh compilerへ戻す。

## Risk

- leakage risk: low。
- overfitting risk: low-to-medium。既存artifactの意味保存をfull validationで確認しているが、同一taskへの反復surgeryなのでLB較正が必要。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
