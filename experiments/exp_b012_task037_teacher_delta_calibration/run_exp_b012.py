from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from datetime import date
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    load_neurogolf_utils,
    load_routes,
    point,
    sha256,
    validate_examples,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp_b012_task037_teacher_delta_calibration"
EXP_DIR = ROOT / "experiments" / EXP_ID
STRICT_EXP = ROOT / "experiments" / "exp005_top_cost_rewrite_strict"
TEACHER_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
STRICT_SEED_SCORE = 6282.230228
TASK_ID = 37


def load_strict_seed_tasks() -> dict[int, BaseTask]:
    routes = load_routes()
    rows: dict[int, dict[str, str]] = {}
    with (STRICT_EXP / "rewrite_manifest.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["task_id"])] = row
    with zipfile.ZipFile(STRICT_EXP / "submission.zip") as zf:
        raws = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}
    return {
        task_id: BaseTask(
            task_id=task_id,
            cost=int(float(row["new_cost"])),
            points=float(row["new_points"]),
            source=row["source"],
            template_name="strict_seed",
            route=routes.get(task_id, ""),
            raw=raws[task_id],
        )
        for task_id, row in rows.items()
    }


def load_teacher_task(task_id: int) -> tuple[bytes, dict[str, str]]:
    with (TEACHER_EXP / "selected_manifest.csv").open(encoding="utf-8", newline="") as f:
        rows = {int(row["task_id"]): row for row in csv.DictReader(f)}
    with zipfile.ZipFile(TEACHER_EXP / "submission.zip") as zf:
        raw = zf.read(f"task{task_id:03d}.onnx")
    return raw, rows[task_id]


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_strict_seed_tasks()
    teacher_raw, teacher_row = load_teacher_task(TASK_ID)
    final_raw = {task_id: task.raw for task_id, task in base_tasks.items()}
    final_raw[TASK_ID] = teacher_raw
    write_zip(OUTPUT_ZIP, final_raw)

    ok, reason, passed, failed = validate_examples(utils, teacher_raw, TASK_ID, -1)
    validation_status = f"{passed}_pass_{failed}_fail"
    base = base_tasks[TASK_ID]
    teacher_cost = int(float(teacher_row["cost"]))
    teacher_points = float(teacher_row["local_points"])
    delta = teacher_points - base.points
    selected_rows = []
    for task_id, task in sorted(base_tasks.items()):
        if task_id == TASK_ID:
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": f"{EXP_ID}_teacher_single_task_delta",
                    "template_name": teacher_row["template_name"],
                    "route": task.route,
                    "cost": teacher_cost,
                    "local_points": teacher_points,
                    "file_bytes": len(teacher_raw),
                    "status": "teacher_delta",
                    "reason": "LB calibration only: risky teacher artifact replacing strict seed for task037",
                    "sha256": sha256(teacher_raw),
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
                    "reason": "strict seed",
                    "sha256": sha256(task.raw),
                }
            )
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "submit_candidate" if ok else "rejected_validation_failed",
        "purpose": "single-task LB calibration for local/LB alignment, not a rule-based adoption",
        "task_id": TASK_ID,
        "strict_exp": str(STRICT_EXP.relative_to(ROOT)),
        "teacher_exp": str(TEACHER_EXP.relative_to(ROOT)),
        "strict_seed_local_estimate": STRICT_SEED_SCORE,
        "strict_task_cost": base.cost,
        "strict_task_points": base.points,
        "teacher_task_cost": teacher_cost,
        "teacher_task_points": teacher_points,
        "local_estimate_delta": delta,
        "new_local_estimate": STRICT_SEED_SCORE + delta,
        "validation_ok": ok,
        "validation_status": validation_status,
        "validation_reason": reason,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_for_calibration" if ok else "no_submit",
        "leakage_risk": "high: teacher artifact is signature/public-artifact style and is not a final strategy.",
        "overfitting_risk": "high: single-task hidden behavior must be measured by Kaggle LB.",
        "decision": "submit if validation passes; use LB delta to calibrate task037 teacher/rule gap.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

strict seedにtask037 teacher artifactだけを差し替え、local/LB差を単一taskで測る。これは説明可能rule採用ではなく、LB calibration用の高リスクdelta提出である。

## 結果

- strict task cost: {base.cost}
- teacher task cost: {teacher_cost}
- local estimate delta: {delta:.6f}
- new local estimate: {STRICT_SEED_SCORE + delta:.6f}
- validation: {validation_status}, ok={ok}, reason={reason}

## Risk

- leakage risk: high。teacher artifactはfinal戦略ではない。
- overfitting risk: high。Kaggle LB deltaでのみ較正可能。

## Decision

validationが通れば、single-task delta calibrationとして提出する。LBで崩れる場合、task037 teacher系はhidden非対応とみなす。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
