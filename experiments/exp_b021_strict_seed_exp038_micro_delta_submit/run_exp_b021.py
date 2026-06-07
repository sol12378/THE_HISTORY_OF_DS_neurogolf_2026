from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import date
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    load_neurogolf_utils,
    load_routes,
    sha256,
    validate_examples,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp_b021_strict_seed_exp038_micro_delta_submit"
EXP_DIR = ROOT / "experiments" / EXP_ID
STRICT_EXP = ROOT / "experiments" / "exp005_top_cost_rewrite_strict"
DELTA_EXP = ROOT / "experiments" / "exp038_fullarc_gated_noop_bypass"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
STRICT_SEED_SCORE = 6282.230228
DELTA_TASKS = [62, 145, 255, 268]


@dataclass(frozen=True)
class DeltaRow:
    task_id: int
    baseline_cost: int
    delta_cost: int
    baseline_points: float
    delta_points: float
    validation_status: str
    validation_ok: bool
    sha256: str


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


def load_delta_manifest() -> dict[int, dict[str, str]]:
    with (DELTA_EXP / "selected_manifest.csv").open(encoding="utf-8", newline="") as f:
        return {int(row["task_id"]): row for row in csv.DictReader(f)}


def load_delta_raws() -> dict[int, bytes]:
    with zipfile.ZipFile(DELTA_EXP / "submission.zip") as zf:
        return {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    strict = load_strict_seed_tasks()
    delta_manifest = load_delta_manifest()
    delta_raws = load_delta_raws()
    final_raw = {task_id: task.raw for task_id, task in strict.items()}
    rows: list[DeltaRow] = []
    selected_rows: list[dict[str, Any]] = []
    all_ok = True

    for task_id in DELTA_TASKS:
        manifest = delta_manifest[task_id]
        raw = delta_raws[task_id]
        ok, reason, passed, failed = validate_examples(utils, raw, task_id, -1)
        validation_status = f"{passed}_pass_{failed}_fail" if ok else f"{passed}_pass_{failed}_fail:{reason}"
        all_ok = all_ok and ok
        rows.append(
            DeltaRow(
                task_id=task_id,
                baseline_cost=strict[task_id].cost,
                delta_cost=int(float(manifest["candidate_cost"])),
                baseline_points=strict[task_id].points,
                delta_points=float(manifest["candidate_points"]),
                validation_status=validation_status,
                validation_ok=ok,
                sha256=sha256(raw),
            )
        )
        if ok:
            final_raw[task_id] = raw

    write_zip(OUTPUT_ZIP, final_raw)
    delta_points = sum(row.delta_points - row.baseline_points for row in rows if row.validation_ok)
    for task_id, task in sorted(strict.items()):
        match = next((r for r in rows if r.task_id == task_id and r.validation_ok), None)
        if match:
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": EXP_ID,
                    "template_name": delta_manifest[task_id]["template_name"],
                    "route": task.route,
                    "cost": match.delta_cost,
                    "local_points": match.delta_points,
                    "file_bytes": len(final_raw[task_id]),
                    "status": "micro_delta",
                    "reason": "exp038 full-arc-safe single/micro delta for LB calibration",
                    "sha256": sha256(final_raw[task_id]),
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

    with (EXP_DIR / "delta_validation.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(DeltaRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([row.__dict__ for row in rows])
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "submit_candidate" if all_ok and delta_points > 0 else ("partial_submit_candidate" if delta_points > 0 else "no_submit"),
        "strict_exp": str(STRICT_EXP.relative_to(ROOT)),
        "delta_exp": str(DELTA_EXP.relative_to(ROOT)),
        "delta_tasks": DELTA_TASKS,
        "validation_all_ok": all_ok,
        "delta_rows": [row.__dict__ for row in rows],
        "strict_seed_local_estimate": STRICT_SEED_SCORE,
        "local_estimate_delta": delta_points,
        "new_local_estimate": STRICT_SEED_SCORE + delta_points,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_for_micro_delta_lb_calibration" if all_ok and delta_points > 0 else "no_submit",
        "leakage_risk": "low-to-medium: graph surgery deltas are full-arc validated and based on strict seed, but still task-specific.",
        "overfitting_risk": "medium: small delta should be Kaggle-calibrated before scaling.",
        "decision": "submit if all deltas validate and local delta is positive.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

strict seedにexp038 full-arc-safe graph surgeryの4 taskだけを載せ、小さいlocal/LB較正提出候補を作る。

## 結果

- delta tasks: {DELTA_TASKS}
- validation all ok: {all_ok}
- local estimate delta: {delta_points:.6f}
- new local estimate: {STRICT_SEED_SCORE + delta_points:.6f}
- submission decision: {result["submission_decision"]}

## Risk

- leakage risk: {result["leakage_risk"]}
- overfitting risk: {result["overfitting_risk"]}
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
