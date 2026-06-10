from __future__ import annotations

import csv
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import onnx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (
    build_boundary_flood_fill_candidate,
    examples_for,
    infer_static_ok,
    load_neurogolf_utils,
    load_task,
    score_model,
    sha256,
    validate_examples,
    zip_sanity,
)


EXP_ID = "exp317_task187_zero_component_onnx_repair"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp297_exp262_skip_task048_336_fresh_candidates"
TASK_ID = 187
BASE_PUBLIC_LB = 6008.96
STEPS_TO_TRY = [30, 45, 60]


def point(cost: int | float) -> float:
    import math

    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def build_repair_zip(candidate_raw: bytes | None) -> dict[str, object] | None:
    if candidate_raw is None:
        return None
    base_zip = BASE_EXP / "submission.zip"
    out_zip = EXP_DIR / "submission.zip"
    with zipfile.ZipFile(base_zip) as zin, zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name in sorted(zin.namelist()):
            tid = int(Path(name).stem.replace("task", ""))
            zout.writestr(name, candidate_raw if tid == TASK_ID else zin.read(name))
    return zip_sanity(out_zip)


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    task = load_task(TASK_ID)
    task_examples = examples_for(task, -1)
    rows: list[dict[str, object]] = []
    best_raw: bytes | None = None
    best_row: dict[str, object] | None = None

    for steps in STEPS_TO_TRY:
        candidate = build_boundary_flood_fill_candidate(TASK_ID, task_examples, "public_zero_repair", steps)
        row: dict[str, object] = {
            "steps": steps,
            "template_name": candidate.template_name,
            "generated_status": candidate.status,
            "generated_reason": candidate.reason,
            "raw_bytes": len(candidate.raw) if candidate.raw is not None else "",
            "static_ok": False,
            "static_reason": "not generated",
            "validation_status": "not_run",
            "validation_reason": "not_run",
            "memory": "",
            "params": "",
            "cost": "",
            "points_if_public_alive": "",
            "sha256": "",
        }
        if candidate.raw is None:
            rows.append(row)
            continue
        model = onnx.load_model_from_string(candidate.raw)
        static_ok, static_reason = infer_static_ok(model)
        row["static_ok"] = static_ok
        row["static_reason"] = static_reason
        if not static_ok:
            rows.append(row)
            continue
        sanitized = utils.sanitize_model(model)
        if sanitized is None:
            row["static_reason"] = "sanitize failed"
            rows.append(row)
            continue
        raw = sanitized.SerializeToString()
        ok, reason, passed, failed = validate_examples(utils, raw, TASK_ID, arc_gen_sample=-1)
        row["validation_status"] = f"{passed}_pass_{failed}_fail"
        row["validation_reason"] = reason
        row["sha256"] = sha256(raw)
        if not ok:
            rows.append(row)
            continue
        memory, params, score_reason = score_model(utils, raw, TASK_ID, f"{EXP_ID}_{steps}", EXP_DIR)
        row["validation_reason"] = score_reason
        if memory is not None and params is not None:
            cost = int(memory + params)
            row["memory"] = int(memory)
            row["params"] = int(params)
            row["cost"] = cost
            row["points_if_public_alive"] = point(cost)
        rows.append(row)
        if memory is not None and params is not None:
            if best_row is None or int(row["cost"]) < int(best_row["cost"]):
                best_row = row
                best_raw = raw

    zip_info = build_repair_zip(best_raw)
    candidate_points = float(best_row["points_if_public_alive"]) if best_row is not None else 0.0
    result: dict[str, Any] = {
        "exp_id": EXP_ID,
        "date": "2026-06-11",
        "status": "repair_zip_ready" if best_row is not None and zip_info is not None else "no_fullarc_onnx_candidate",
        "hypothesis": "task187 zero-component rule can be repaired with correctness-first boundary flood-fill ONNX even if cost is high",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_public_lb": BASE_PUBLIC_LB,
        "task_id": TASK_ID,
        "candidate_rows": rows,
        "best_candidate": best_row,
        "expected_public_lb_if_pass": BASE_PUBLIC_LB + candidate_points if best_row is not None else None,
        "zip_sanity": zip_info,
        "outputs": {"submission_zip": str((EXP_DIR / "submission.zip").relative_to(ROOT))} if zip_info else {},
        "elapsed_s": round(time.time() - started, 3),
        "submission_decision": "submit_publiczero_repair_probe" if best_row is not None and zip_info else "no_submit_validation_or_score_failed",
        "leakage_risk": "low-medium: task187 was selected via public-zero probe, but the candidate is an input-derived zero-component rule inferred from train and full-arc validated.",
        "overfitting_risk": "medium: correctness-first unrolled flood-fill is task-specific and still needs LB probe to verify public repair.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    notes = [
        f"# {EXP_ID}",
        "",
        "## Plan",
        "",
        "task187 の public-zero を、exp316 で full-arc pass した zero-component rule の correctness-first ONNX lowering で修復する。",
        "",
        "## Result",
        "",
        f"- status: `{result['status']}`",
        f"- best_candidate: `{best_row}`",
        f"- expected_public_lb_if_pass: `{result['expected_public_lb_if_pass']}`",
        f"- submission_decision: `{result['submission_decision']}`",
        "",
        "## Risk",
        "",
        f"- leakage risk: {result['leakage_risk']}",
        f"- overfitting risk: {result['overfitting_risk']}",
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
