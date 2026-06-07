from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
from dataclasses import asdict
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    build_boundary_flood_fill_candidate,
    candidate_fields,
    evaluate_candidate,
    examples_for,
    load_base_tasks,
    load_neurogolf_utils,
    load_task,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp_b028_task251_closed_component_lowering_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"
TASK_ID = 251
BASE_SCORE = 6282.812217709089
OUTPUT_ZIP = EXP_DIR / "submission.zip"


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks(BASE_EXP)
    base_task = base[TASK_ID]
    task_examples = examples_for(load_task(TASK_ID), -1)
    eval_rows = []
    best_row = None
    best_raw = None

    for steps in [6, 8, 10, 12, 14, 18, 24, 30]:
        candidate = build_boundary_flood_fill_candidate(TASK_ID, task_examples, base_task.route, steps)
        row, raw = evaluate_candidate(utils, candidate, base_task, -1, EXP_DIR)
        eval_rows.append(row)
        if raw is not None and row.status == "improved" and isinstance(row.candidate_cost, int):
            if best_row is None or int(row.candidate_cost) < int(best_row.candidate_cost):
                best_row = row
                best_raw = raw

    final_raws = {task_id: task.raw for task_id, task in base.items()}
    selected_rows: list[dict[str, Any]] = []
    local_delta = 0.0
    if best_row is not None and best_raw is not None:
        final_raws[TASK_ID] = best_raw
        local_delta = float(best_row.candidate_points) - base_task.points
    write_zip(OUTPUT_ZIP, final_raws)

    for task_id in sorted(base):
        task = base[task_id]
        if task_id == TASK_ID and best_row is not None and best_raw is not None:
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": EXP_ID,
                    "template_name": best_row.template_name,
                    "route": task.route,
                    "cost": best_row.candidate_cost,
                    "local_points": best_row.candidate_points,
                    "file_bytes": len(best_raw),
                    "status": "improved",
                    "reason": best_row.reason,
                    "sha256": sha256(best_raw),
                }
            )
        else:
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": task.source,
                    "template_name": task.template_name,
                    "route": task.route,
                    "cost": task.cost,
                    "local_points": task.points,
                    "file_bytes": len(task.raw),
                    "status": "baseline",
                    "reason": "base exp_b025",
                    "sha256": sha256(task.raw),
                }
            )

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows([asdict(row) for row in eval_rows])
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "submit_candidate" if best_row is not None else "no_cost_gain",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "task_id": TASK_ID,
        "base_cost": base_task.cost,
        "base_points": base_task.points,
        "candidate_rows": [asdict(row) for row in eval_rows],
        "best_candidate": asdict(best_row) if best_row is not None else None,
        "local_estimate_delta": local_delta,
        "new_local_estimate": BASE_SCORE + local_delta,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_if_improved" if best_row is not None else "no_submit",
        "leakage_risk": "low-to-medium: lowering implements explicit task251 component rule; no output lookup table.",
        "overfitting_risk": "medium: boundary flood unroll is task-specific and must be LB-calibrated if improved.",
        "decision": "If cost improves exp_b025 task251, submit single-task delta; otherwise keep rule and design cheaper closed-form lowering.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`exp_b027` の task251 closed zero-component rule を、既存 `boundary_flood_fill` ONNX loweringで実cost評価する。

## 結果

- base cost: `{base_task.cost}`
- best candidate: `{best_row.template_name if best_row else 'none'}`
- local delta: `{local_delta:.9f}`
- submission decision: `{result["submission_decision"]}`

## 判断

改善があればsingle-task deltaとして提出する。改善がなければ、naive flood-fill系は避け、rectangle-specific closed maskまたはborder reachabilityの低cost化へ進む。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
