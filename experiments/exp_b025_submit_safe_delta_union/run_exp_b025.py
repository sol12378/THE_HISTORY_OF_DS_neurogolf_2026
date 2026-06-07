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
    load_neurogolf_utils,
    point,
    sha256,
    validate_examples,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp_b025_submit_safe_delta_union"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp068_seddik_style_strict_scalarization"
DELTA_EXP = ROOT / "experiments" / "exp_b021_strict_seed_exp038_micro_delta_submit"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
BASE_SCORE = 6282.610350374699
DELTA_TASKS = [62, 145, 255, 268]


def read_manifest(exp_dir: pathlib.Path) -> dict[int, dict[str, str]]:
    with (exp_dir / "selected_manifest.csv").open(encoding="utf-8", newline="") as f:
        return {int(row["task_id"]): row for row in csv.DictReader(f)}


def read_zip(exp_dir: pathlib.Path) -> dict[int, bytes]:
    with zipfile.ZipFile(exp_dir / "submission.zip") as zf:
        return {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}


def row_cost(row: dict[str, str]) -> int:
    for key in ("cost", "candidate_cost", "new_cost"):
        if row.get(key):
            return int(float(row[key]))
    raise KeyError(f"no cost column in {row}")


def row_points(row: dict[str, str]) -> float:
    for key in ("local_points", "candidate_points", "new_points"):
        if row.get(key):
            return float(row[key])
    return point(row_cost(row))


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()

    base_manifest = read_manifest(BASE_EXP)
    base_raws = read_zip(BASE_EXP)
    delta_manifest = read_manifest(DELTA_EXP)
    delta_raws = read_zip(DELTA_EXP)

    final_raws = dict(base_raws)
    validation_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    local_delta = 0.0
    all_ok = True

    for task_id in DELTA_TASKS:
        raw = delta_raws[task_id]
        ok, reason, passed, failed = validate_examples(utils, raw, task_id, -1)
        base_cost = row_cost(base_manifest[task_id])
        base_pts = row_points(base_manifest[task_id])
        delta_cost = row_cost(delta_manifest[task_id])
        delta_pts = row_points(delta_manifest[task_id])
        improved = ok and delta_pts > base_pts and delta_cost < base_cost
        all_ok = all_ok and ok
        if improved:
            final_raws[task_id] = raw
            local_delta += delta_pts - base_pts
        validation_rows.append(
            {
                "task_id": task_id,
                "base_cost": base_cost,
                "delta_cost": delta_cost,
                "base_points": base_pts,
                "delta_points": delta_pts,
                "point_delta": delta_pts - base_pts,
                "validation_status": f"{passed}_pass_{failed}_fail",
                "validation_ok": ok,
                "accepted": improved,
                "reason": reason if not ok else "ok",
                "sha256": sha256(raw),
            }
        )

    write_zip(OUTPUT_ZIP, final_raws)

    accepted = {row["task_id"] for row in validation_rows if row["accepted"]}
    for task_id in sorted(base_manifest):
        if task_id in accepted:
            row = delta_manifest[task_id]
            raw = final_raws[task_id]
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": EXP_ID,
                    "template_name": row.get("template_name", ""),
                    "route": row.get("route", base_manifest[task_id].get("route", "")),
                    "cost": row_cost(row),
                    "local_points": row_points(row),
                    "file_bytes": len(raw),
                    "status": "union_delta",
                    "reason": "exp_b021 full-arc-safe delta over exp068 submit-safe base",
                    "sha256": sha256(raw),
                }
            )
        else:
            row = base_manifest[task_id]
            raw = final_raws[task_id]
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": row.get("source", BASE_EXP.name),
                    "template_name": row.get("template_name", ""),
                    "route": row.get("route", ""),
                    "cost": row_cost(row),
                    "local_points": row_points(row),
                    "file_bytes": len(raw),
                    "status": row.get("status", "baseline"),
                    "reason": row.get("reason", "base exp068"),
                    "sha256": sha256(raw),
                }
            )

    with (EXP_DIR / "delta_validation.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "task_id",
            "base_cost",
            "delta_cost",
            "base_points",
            "delta_points",
            "point_delta",
            "validation_status",
            "validation_ok",
            "accepted",
            "reason",
            "sha256",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(validation_rows)
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "submit_candidate" if all_ok and local_delta > 0 else "no_submit",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "delta_exp": str(DELTA_EXP.relative_to(ROOT)),
        "base_local_estimate": BASE_SCORE,
        "delta_tasks": DELTA_TASKS,
        "accepted_tasks": sorted(accepted),
        "validation_all_ok": all_ok,
        "local_estimate_delta": local_delta,
        "new_local_estimate": BASE_SCORE + local_delta,
        "delta_rows": validation_rows,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_for_union_delta_lb_calibration" if all_ok and local_delta > 0 else "no_submit",
        "leakage_risk": "low-to-medium: union of two already submitted full-arc-safe strict-derived deltas.",
        "overfitting_risk": "medium: task-specific graph surgery remains small but should be LB-calibrated after union.",
        "decision": "Submit if all overlay tasks validate and local delta is positive.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

Kaggle転移が確認済みのsubmit-safe deltaを合成する。

- base: `exp068_seddik_style_strict_scalarization`
- overlay: `exp_b021_strict_seed_exp038_micro_delta_submit`

## 結果

- accepted tasks: {sorted(accepted)}
- local delta over exp068: {local_delta:.9f}
- new local estimate: {BASE_SCORE + local_delta:.9f}
- submission decision: {result["submission_decision"]}

## Risk

- leakage risk: {result["leakage_risk"]}
- overfitting risk: {result["overfitting_risk"]}
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
