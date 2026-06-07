from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    load_routes,
    load_neurogolf_utils,
    point,
    sha256,
    validate_examples,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp063_task020_teacher_delta_calibration"
EXP_DIR = ROOT / "experiments" / EXP_ID
STRICT_EXP = ROOT / "experiments" / "exp005_top_cost_rewrite_strict"
TEACHER_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
TARGET_TASK = 20
OUTPUT_ZIP = EXP_DIR / "submission.zip"
STRICT_SEED_LOCAL = 6282.230228092811


@dataclass(frozen=True)
class DeltaRow:
    task_id: int
    strict_cost: int
    strict_points: float
    teacher_cost: int
    teacher_points: float
    local_delta: float
    validation_ok: bool
    validation_status: str
    validation_reason: str
    teacher_sha256: str


def manifest_by_task(exp_dir: pathlib.Path) -> dict[int, dict[str, str]]:
    candidates = [exp_dir / "selected_manifest.csv", exp_dir / "rewrite_manifest.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"no manifest in {exp_dir}")
    rows: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if "task_id" in row:
                rows[int(row["task_id"])] = row
    return rows


def zip_raws(exp_dir: pathlib.Path) -> dict[int, bytes]:
    with zipfile.ZipFile(exp_dir / "submission.zip") as zf:
        return {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}


def load_exp_tasks(exp_dir: pathlib.Path) -> dict[int, BaseTask]:
    manifest = manifest_by_task(exp_dir)
    raws = zip_raws(exp_dir)
    routes = load_routes()
    out: dict[int, BaseTask] = {}
    for task_id, row in manifest.items():
        cost, pts = row_cost_points(row)
        out[task_id] = BaseTask(
            task_id=task_id,
            cost=cost,
            points=pts,
            source=row.get("source", exp_dir.name),
            template_name=row.get("template_name", row.get("variant", "")),
            route=row.get("route", routes.get(task_id, "")),
            raw=raws[task_id],
        )
    return out


def row_cost_points(row: dict[str, str], cost_key: str = "cost", point_key: str = "local_points") -> tuple[int, float]:
    if cost_key in row and row[cost_key] != "":
        cost = int(float(row[cost_key]))
    elif "new_cost" in row and row["new_cost"] != "":
        cost = int(float(row["new_cost"]))
    elif "candidate_cost" in row and row["candidate_cost"] != "":
        cost = int(float(row["candidate_cost"]))
    elif "teacher_cost" in row and row["teacher_cost"] != "":
        cost = int(float(row["teacher_cost"]))
    else:
        raise KeyError(row)
    if point_key in row and row[point_key] != "":
        pts = float(row[point_key])
    elif "new_points" in row and row["new_points"] != "":
        pts = float(row["new_points"])
    elif "candidate_points" in row and row["candidate_points"] != "":
        pts = float(row["candidate_points"])
    elif "teacher_points" in row and row["teacher_points"] != "":
        pts = float(row["teacher_points"])
    else:
        pts = point(cost)
    return cost, pts


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()

    strict_tasks = load_exp_tasks(STRICT_EXP)
    strict_raws = {task_id: task.raw for task_id, task in strict_tasks.items()}
    teacher_manifest = manifest_by_task(TEACHER_EXP)
    teacher_raws = zip_raws(TEACHER_EXP)
    teacher_raw = teacher_raws[TARGET_TASK]
    strict_task = strict_tasks[TARGET_TASK]
    teacher_row = teacher_manifest[TARGET_TASK]
    teacher_cost, teacher_points = row_cost_points(teacher_row)

    ok, reason, passed, failed = validate_examples(utils, teacher_raw, TARGET_TASK, -1)
    validation_status = f"{passed}_pass_{failed}_fail"
    local_delta = teacher_points - strict_task.points if ok else 0.0
    new_local = STRICT_SEED_LOCAL + local_delta

    delta_row = DeltaRow(
        task_id=TARGET_TASK,
        strict_cost=strict_task.cost,
        strict_points=strict_task.points,
        teacher_cost=teacher_cost,
        teacher_points=teacher_points,
        local_delta=local_delta,
        validation_ok=ok,
        validation_status=validation_status,
        validation_reason=reason,
        teacher_sha256=sha256(teacher_raw),
    )

    final_raws = dict(strict_raws)
    if ok:
        final_raws[TARGET_TASK] = teacher_raw
    write_zip(OUTPUT_ZIP, final_raws)

    selected_rows: list[dict[str, Any]] = []
    for task_id in sorted(strict_tasks):
        task: BaseTask = strict_tasks[task_id]
        if task_id == TARGET_TASK and ok:
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": f"{EXP_ID}_teacher_task020",
                    "template_name": "teacher_task020_calibration",
                    "route": task.route,
                    "cost": teacher_cost,
                    "local_points": teacher_points,
                    "file_bytes": len(teacher_raw),
                    "status": "teacher_delta",
                    "reason": "single-task teacher delta calibration; high risk, not final strategy",
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
                    "status": "strict_seed",
                    "reason": "unchanged strict seed",
                    "sha256": sha256(task.raw),
                }
            )
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)
    with (EXP_DIR / "delta_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(DeltaRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerow(asdict(delta_row))

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "ready_to_submit" if ok else "rejected_validation_failed",
        "purpose": "single-task local/LB calibration for task020 teacher delta; not final rule adoption",
        "strict_exp": str(STRICT_EXP.relative_to(ROOT)),
        "teacher_exp": str(TEACHER_EXP.relative_to(ROOT)),
        "target_task": TARGET_TASK,
        "strict_seed_local_estimate": STRICT_SEED_LOCAL,
        "new_local_estimate": new_local,
        "local_estimate_delta": local_delta,
        "delta": asdict(delta_row),
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_for_calibration" if ok else "no_submit",
        "leakage_risk": "high: teacher artifact is signature/public-artifact style; this is a calibration probe only.",
        "overfitting_risk": "high: use Kaggle LB delta to measure whether task020 teacher behavior generalizes.",
        "decision": "Submit only because user requested faster local/LB alignment; do not treat as final low-risk rule replacement.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task020だけをteacher artifactへ差し替え、local/LB較正を早める。これは最終戦略ではなく、single-task calibration probe。

## 結果

- validation: {validation_status}, ok={ok}
- strict cost: {strict_task.cost}
- teacher cost: {teacher_cost}
- local delta: {local_delta:.6f}
- new local estimate: {new_local:.6f}

## Risk

- leakage risk: high。teacher artifactはsignature/public artifact系。
- overfitting risk: high。Kaggle LB deltaでlocalとの対応を見るためだけに使う。

## Decision

validation passならKaggleへ提出し、LB deltaを記録する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
