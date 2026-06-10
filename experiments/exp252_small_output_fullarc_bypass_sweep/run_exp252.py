from __future__ import annotations

import csv
import json
import pathlib
import sys
from dataclasses import asdict
from datetime import date

import onnx

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "exp036_noop_bypass_and_prune"))

import run_exp036 as e36  # noqa: E402
from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    candidate_fields,
    evaluate_candidate,
    grid_to_array,
    load_base_tasks,
    load_neurogolf_utils,
    load_task,
    validate_examples,
)


EXP_ID = "exp252_small_output_fullarc_bypass_sweep"
EXP_DIR = ROOT / "experiments" / EXP_ID
TOP_N = 20
MAX_CANDIDATES_PER_TASK = 80


def output_area(task_id: int) -> int:
    task = load_task(task_id)
    shapes = set()
    for ex in task["train"] + task["test"] + task["arc-gen"]:
        y = grid_to_array(ex["output"])
        shapes.add((y.shape[0], y.shape[1]))
    if len(shapes) != 1:
        return 9999
    h, w = next(iter(shapes))
    return h * w


def target_tasks(base: dict[int, BaseTask]) -> list[int]:
    rows = []
    for task_id, item in base.items():
        area = output_area(task_id)
        if area <= 16 and item.cost >= 8000:
            rows.append((item.cost, task_id, area))
    return [task_id for _, task_id, _ in sorted(rows, reverse=True)[:TOP_N]]


def fullarc_ok(utils, raw: bytes, task_id: int) -> tuple[bool, str, int, int]:
    return validate_examples(utils, raw, task_id, -1)


def op_counts(raw: bytes) -> dict[str, int]:
    model = onnx.load_model_from_string(raw)
    out: dict[str, int] = {}
    for node in model.graph.node:
        out[node.op_type] = out.get(node.op_type, 0) + 1
    return out


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()
    targets = target_tasks(base)

    eval_rows = []
    fullarc_rows = []
    selected_rows = []
    improved_raws: dict[int, bytes] = {}
    generated = 0

    for task_id in targets:
        task = base[task_id]
        best_eval = None
        best_raw = None
        cand_count = 0
        for candidate in e36.bypass_candidates(task_id, task.raw, task.route, 1):
            cand_count += 1
            if cand_count > MAX_CANDIDATES_PER_TASK:
                break
            generated += 1
            evaluation, accepted_raw = evaluate_candidate(utils, candidate, task, 20, EXP_DIR)
            eval_rows.append(asdict(evaluation))
            if accepted_raw is None:
                continue
            ok, reason, passed, failed = fullarc_ok(utils, accepted_raw, task_id)
            fullarc_rows.append(
                {
                    "task_id": task_id,
                    "template_name": candidate.template_name,
                    "sample20_cost": evaluation.candidate_cost,
                    "sample20_points": evaluation.candidate_points,
                    "full_arc_ok": ok,
                    "passed": passed,
                    "failed": failed,
                    "reason": reason,
                    "sha256": evaluation.sha256,
                }
            )
            if not ok:
                continue
            if best_eval is None or float(evaluation.candidate_points) > float(best_eval.candidate_points):
                best_eval = evaluation
                best_raw = accepted_raw
        if best_eval is not None and best_raw is not None:
            row = asdict(best_eval)
            row["validation_status"] = "full_arc_pass"
            selected_rows.append(row)
            improved_raws[task_id] = best_raw

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(eval_rows)
    with (EXP_DIR / "fullarc_candidate_audit.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["task_id", "template_name", "sample20_cost", "sample20_points", "full_arc_ok", "passed", "failed", "reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fullarc_rows)
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(selected_rows)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "targets": targets,
        "top_n": TOP_N,
        "max_candidates_per_task": MAX_CANDIDATES_PER_TASK,
        "generated_candidate_count": generated,
        "sample20_improved_count": len(fullarc_rows),
        "fullarc_selected_count": len(selected_rows),
        "selected_tasks": [int(r["task_id"]) for r in selected_rows],
        "selected_rows": selected_rows,
        "target_op_counts": {str(t): op_counts(base[t].raw) for t in targets},
        "decision": "If selected_rows non-empty, bundle on current best and consider single submission. Otherwise bypass lane has no quick gain here.",
        "submission_decision": "no_submit: sweep only",
        "leakage_risk": "low-medium: full-arc gated surgery on existing artifacts.",
        "overfitting_risk": "medium-low: full available validation, but private risk remains.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

small-output高cost候補に対し、既存artifactの1-pass node bypass surgeryをfull-arc gate付きで試し、即提出可能なcost改善があるか確認する。

## 結果

- targets: `{targets}`
- generated_candidate_count: `{generated}`
- sample20_improved_count: `{len(fullarc_rows)}`
- fullarc_selected_count: `{len(selected_rows)}`
- selected_tasks: `{[int(r["task_id"]) for r in selected_rows]}`

## 判断

selectedがあればbundle/submission候補。なければこの範囲のquick bypass laneは不発。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
