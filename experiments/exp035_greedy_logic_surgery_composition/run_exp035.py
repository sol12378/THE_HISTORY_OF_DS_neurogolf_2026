from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import sys
import zipfile
from dataclasses import asdict
from typing import Iterable

import onnx

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "experiments"))

from phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    candidate_fields,
    evaluate_candidate,
    load_base_tasks,
    load_neurogolf_utils,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp035_greedy_logic_surgery_composition"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp034_redundant_logic_surgery_sweep"
FALLBACK_BASE_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
ARC_GEN_SAMPLE = 20
TOP_K = 120
MAX_PASSES_PER_TASK = 4
BINARY_OPS = {"And", "Or", "Greater", "Less", "LessOrEqual", "GreaterOrEqual", "Equal"}


def source_exp_dir() -> pathlib.Path:
    return BASE_EXP if (BASE_EXP / "submission.zip").exists() else FALLBACK_BASE_EXP


def load_raws(exp_dir: pathlib.Path) -> dict[int, bytes]:
    raws: dict[int, bytes] = {}
    with zipfile.ZipFile(exp_dir / "submission.zip") as zf:
        for name in zf.namelist():
            task_id = int(pathlib.Path(name).stem.replace("task", ""))
            raws[task_id] = zf.read(name)
    return raws


def top_task_ids(base: dict[int, BaseTask], top_k: int) -> list[int]:
    return [item.task_id for item in sorted(base.values(), key=lambda x: (-x.cost, x.task_id))[:top_k]]


def replace_all_inputs(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for idx, input_name in enumerate(node.input):
            if input_name == old:
                node.input[idx] = new
    for output in model.graph.output:
        if output.name == old:
            output.name = new


def remove_node_by_output(model: onnx.ModelProto, output_name: str) -> None:
    kept = [node for node in model.graph.node if output_name not in set(node.output)]
    del model.graph.node[:]
    model.graph.node.extend(kept)


def load_base_with_overlay(source_exp: pathlib.Path, raws: dict[int, bytes]) -> dict[int, BaseTask]:
    base = load_base_tasks(FALLBACK_BASE_EXP)
    overlay_chain = [
        ROOT / "experiments" / "exp032_task187_component_lowering",
        ROOT / "experiments" / "exp033_redundant_and_surgery_sweep",
        ROOT / "experiments" / "exp034_redundant_logic_surgery_sweep",
    ]
    for exp_dir in overlay_chain:
        if source_exp == FALLBACK_BASE_EXP:
            break
        if not exp_dir.exists():
            continue
        apply_selected_overlay(base, exp_dir, raws)
        if exp_dir == source_exp:
            break
    return base


def selected_rows_for_overlay(exp_dir: pathlib.Path) -> list[dict]:
    manifest_path = exp_dir / "selected_manifest.csv"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    result_path = exp_dir / "result.json"
    if not result_path.exists():
        return []
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = result.get("top_selected", [])
    if rows:
        return rows
    evaluation = result.get("evaluation")
    return [evaluation] if evaluation else []


def apply_selected_overlay(base: dict[int, BaseTask], exp_dir: pathlib.Path, raws: dict[int, bytes]) -> None:
    for row in selected_rows_for_overlay(exp_dir):
        task_id = int(row["task_id"])
        if task_id not in base:
            continue
        old = base[task_id]
        base[task_id] = BaseTask(
            task_id=task_id,
            cost=int(row["candidate_cost"]),
            points=float(row["candidate_points"]),
            source=str(exp_dir.relative_to(ROOT)),
            template_name=str(row["template_name"]),
            route=old.route,
            raw=raws[task_id],
        )


def surgery_candidates(task_id: int, raw: bytes, route: str, pass_idx: int) -> Iterable[Candidate]:
    model = onnx.load_model_from_string(raw)
    for idx, node in enumerate(model.graph.node):
        if len(node.output) != 1:
            continue
        output_name = node.output[0]
        replacements: list[tuple[int, str]] = []
        if node.op_type in BINARY_OPS and len(node.input) == 2:
            replacements = [(0, node.input[0]), (1, node.input[1])]
        elif node.op_type == "Where" and len(node.input) == 3:
            replacements = [(1, node.input[1]), (2, node.input[2])]
        else:
            continue
        for input_idx, input_name in replacements:
            candidate_model = onnx.load_model_from_string(raw)
            replace_all_inputs(candidate_model, output_name, input_name)
            remove_node_by_output(candidate_model, output_name)
            yield Candidate(
                task_id,
                f"greedy{pass_idx}_{node.op_type}_node{idx}_to_input{input_idx}",
                route,
                candidate_model.SerializeToString(),
                "generated",
                f"pass={pass_idx}; replace {output_name}={node.op_type}(...) with {input_name}",
            )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    source_exp = source_exp_dir()
    utils = load_neurogolf_utils()
    raws = load_raws(source_exp)
    base = load_base_with_overlay(source_exp, raws)

    eval_rows = []
    final_selected: dict[int, bytes] = {}
    final_eval = {}
    accepted_steps: list[dict] = []
    generated = 0

    for task_id in top_task_ids(base, TOP_K):
        task = base[task_id]
        current_raw = raws[task_id]
        current_base = task
        task_improved = False
        for pass_idx in range(1, MAX_PASSES_PER_TASK + 1):
            best_eval = None
            best_raw = None
            for candidate in surgery_candidates(task_id, current_raw, task.route, pass_idx):
                generated += 1
                evaluation, accepted_raw = evaluate_candidate(utils, candidate, current_base, ARC_GEN_SAMPLE, EXP_DIR)
                eval_rows.append(asdict(evaluation))
                if accepted_raw is None:
                    continue
                if best_eval is None or float(evaluation.candidate_points) > float(best_eval.candidate_points):
                    best_eval = evaluation
                    best_raw = accepted_raw
            if best_eval is None or best_raw is None:
                break
            accepted_steps.append(asdict(best_eval))
            current_raw = best_raw
            current_base = BaseTask(
                task_id=task_id,
                cost=int(best_eval.candidate_cost),
                points=float(best_eval.candidate_points),
                source=EXP_ID,
                template_name=best_eval.template_name,
                route=task.route,
                raw=current_raw,
            )
            task_improved = True
        if task_improved:
            final_selected[task_id] = current_raw
            final_eval[task_id] = current_base

    bundle_raws = dict(raws)
    bundle_raws.update(final_selected)
    sanity = {}
    if final_selected:
        write_zip(EXP_DIR / "submission.zip", bundle_raws)
        sanity = zip_sanity(EXP_DIR / "submission.zip")

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(eval_rows)

    selected_rows = []
    for task_id, item in sorted(final_eval.items()):
        base_item = base[task_id]
        selected_rows.append(
            {
                "task_id": task_id,
                "route": item.route,
                "template_name": item.template_name,
                "baseline_cost": base_item.cost,
                "candidate_cost": item.cost,
                "baseline_points": base_item.points,
                "candidate_points": item.points,
                "file_bytes": len(final_selected[task_id]),
                "validation_status": "composed_sample20_pass",
                "status": "improved",
                "reason": "greedy composed surgery",
                "sha256": hashlib.sha256(final_selected[task_id]).hexdigest(),
            }
        )
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(selected_rows)

    base_local = sum(task.points for task in base.values())
    delta = sum(float(row["candidate_points"]) - float(row["baseline_points"]) for row in selected_rows)
    local_estimate = base_local + delta
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-06",
        "status": "improved" if selected_rows else "no_gain",
        "base_exp": str(source_exp.relative_to(ROOT)),
        "base_local_estimate": base_local,
        "top_k": TOP_K,
        "max_passes_per_task": MAX_PASSES_PER_TASK,
        "arc_gen_sample": ARC_GEN_SAMPLE,
        "candidate_rows": len(eval_rows),
        "generated_candidate_count": generated,
        "accepted_step_count": len(accepted_steps),
        "improved_task_count": len(selected_rows),
        "local_estimate_delta": delta,
        "local_estimate": local_estimate,
        "gap_to_6500": 6500.0 - local_estimate,
        "submission_decision": "submit_threshold_reached" if local_estimate >= 6500.0 else "no_submit: below 6500 threshold",
        "selected_tasks": [row["task_id"] for row in selected_rows],
        "top_selected": sorted(selected_rows, key=lambda row: float(row["candidate_points"]) - float(row["baseline_points"]), reverse=True)[:20],
        "accepted_steps": accepted_steps[:50],
        "zip_sanity": sanity,
        "leakage_risk": "medium: greedy graph surgery is accepted only after sample20 validation, but base includes high-risk lookup artifacts.",
        "overfitting_risk": "medium-high: sequential edits can compound sample-specific risk without full private-like validation.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## Hypothesis

exp033/034の単発graph surgeryは、同じtaskに対して逐次合成できる。改善候補を1つ採用した後に再度候補生成すれば、さらに冗長nodeを削れる可能性がある。

## Result

- base: `{result["base_exp"]}`
- top_k: `{TOP_K}`
- max passes per task: `{MAX_PASSES_PER_TASK}`
- candidate rows: `{result["candidate_rows"]}`
- accepted steps: `{result["accepted_step_count"]}`
- improved tasks: `{result["improved_task_count"]}`
- local estimate: `{result["local_estimate"]:.6f}`
- delta: `{result["local_estimate_delta"]:.6f}`
- gap to 6500: `{result["gap_to_6500"]:.6f}`
- submission decision: `{result["submission_decision"]}`

## Risks

- leakage risk: {result["leakage_risk"]}
- overfitting risk: {result["overfitting_risk"]}
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
