from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
from dataclasses import asdict
from datetime import date
from typing import Any

import onnx


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    Candidate,
    candidate_fields,
    evaluate_candidate,
    load_base_tasks,
    load_neurogolf_utils,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp074_task251_reachability_depth_surgery"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"
BASE_SCORE = 6282.812217709089
TASK_ID = 251
OUTPUT_ZIP = EXP_DIR / "submission.zip"


def replace_input(raw: bytes, old: str, new: str) -> bytes:
    model = onnx.load_model_from_string(raw)
    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name == old:
                node.input[idx] = new
    return model.SerializeToString()


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks(BASE_EXP)
    base_task = base[TASK_ID]

    # safe_name_61 is the final reachability mask used by the current artifact.
    # Earlier Max outputs correspond to shallower propagation depths.
    depth_outputs = [
        ("depth0_safe_name_40", "safe_name_40"),
        ("depth1_safe_name_43", "safe_name_43"),
        ("depth2_safe_name_46", "safe_name_46"),
        ("depth3_safe_name_49", "safe_name_49"),
        ("depth4_safe_name_52", "safe_name_52"),
        ("depth5_safe_name_55", "safe_name_55"),
        ("depth6_safe_name_58", "safe_name_58"),
    ]
    eval_rows = []
    best_row = None
    best_raw = None
    for label, output_name in depth_outputs:
        raw = replace_input(base_task.raw, "safe_name_61", output_name)
        candidate = Candidate(TASK_ID, f"reachability_{label}", base_task.route, raw, "generated", f"replace safe_name_61 with {output_name}")
        row, improved_raw = evaluate_candidate(utils, candidate, base_task, -1, EXP_DIR)
        eval_rows.append(row)
        if improved_raw is not None and row.status == "improved" and isinstance(row.candidate_cost, int):
            if best_row is None or int(row.candidate_cost) < int(best_row.candidate_cost):
                best_row = row
                best_raw = improved_raw

    final_raws = {task_id: task.raw for task_id, task in base.items()}
    local_delta = 0.0
    if best_row is not None and best_raw is not None:
        final_raws[TASK_ID] = best_raw
        local_delta = float(best_row.candidate_points) - base_task.points
    write_zip(OUTPUT_ZIP, final_raws)

    selected_rows: list[dict[str, Any]] = []
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
        "status": "submit_candidate" if best_row is not None else "no_gain",
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
        "leakage_risk": "low: graph surgery only shortens an existing explicit reachability computation and is full-arc gated.",
        "overfitting_risk": "medium: task-specific depth reduction may still fail hidden generation; submit if improved to calibrate.",
        "decision": "If a shallower reachability output full-passes and reduces cost, submit as single-task delta. Otherwise keep existing artifact and seek different closed-form lowering.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`task251` の既存artifactが持つreachability propagation depthを浅くし、full-arc validationを保ったままcostを削れるか確認する。

## 結果

- base cost: `{base_task.cost}`
- best candidate: `{best_row.template_name if best_row else 'none'}`
- local delta: `{local_delta:.9f}`
- submission decision: `{result["submission_decision"]}`

## 判断

浅いreachabilityで通るならsingle-task deltaとして提出候補。通らなければ、既存artifactの深さは必要であり、別のclosed-form loweringが必要。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
