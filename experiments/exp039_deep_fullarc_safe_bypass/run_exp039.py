from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import sys
import zipfile
from dataclasses import asdict
from typing import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "experiments"))
sys.path.append(str(ROOT / "experiments" / "exp036_noop_bypass_and_prune"))
sys.path.append(str(ROOT / "experiments" / "exp038_fullarc_gated_noop_bypass"))

import run_exp036 as e36  # noqa: E402
import run_exp038 as e38  # noqa: E402
from phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    candidate_fields,
    evaluate_candidate,
    load_neurogolf_utils,
    validate_examples,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp039_deep_fullarc_safe_bypass"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp038_fullarc_gated_noop_bypass"
ARC_GEN_SAMPLE = 20
MAX_PASSES_PER_TASK = 8
SAFE_OPS = {"Mul", "Reshape", "And", "Transpose", "Greater", "Cast", "Squeeze", "Unsqueeze", "Slice"}


def load_raws(exp_dir: pathlib.Path) -> dict[int, bytes]:
    raws: dict[int, bytes] = {}
    with zipfile.ZipFile(exp_dir / "submission.zip") as zf:
        for name in zf.namelist():
            task_id = int(pathlib.Path(name).stem.replace("task", ""))
            raws[task_id] = zf.read(name)
    return raws


def read_selected_tasks(exp_dir: pathlib.Path) -> set[int]:
    out: set[int] = set()
    manifest_path = exp_dir / "selected_manifest.csv"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            out.update(int(row["task_id"]) for row in csv.DictReader(f))
    return out


def read_fullarc_ok_tasks(exp_dir: pathlib.Path) -> set[int]:
    audit_path = exp_dir / "fullarc_candidate_audit.csv"
    if not audit_path.exists():
        return set()
    with audit_path.open("r", encoding="utf-8", newline="") as f:
        return {int(row["task_id"]) for row in csv.DictReader(f) if row["full_arc_ok"] == "True"}


def load_base(raws: dict[int, bytes]) -> dict[int, BaseTask]:
    exp037 = ROOT / "experiments" / "exp037_fullarc_filter_exp036"
    exp035 = ROOT / "experiments" / "exp035_greedy_logic_surgery_composition"
    base = e36.load_base_with_overlay(exp035, raws)
    e38.apply_manifest_overlay(base, exp037, raws)
    e38.apply_manifest_overlay(base, BASE_EXP, raws)
    return base


def candidate_op(template_name: str) -> str:
    parts = template_name.split("_")
    return parts[1] if len(parts) > 1 else ""


def safe_candidates(task_id: int, raw: bytes, route: str, pass_idx: int) -> Iterable[Candidate]:
    for candidate in e36.bypass_candidates(task_id, raw, route, pass_idx):
        if candidate_op(candidate.template_name) in SAFE_OPS:
            yield candidate


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    raws = load_raws(BASE_EXP)
    base = load_base(raws)
    exp037 = ROOT / "experiments" / "exp037_fullarc_filter_exp036"
    task_ids = sorted(read_selected_tasks(exp037) | read_selected_tasks(BASE_EXP) | read_fullarc_ok_tasks(BASE_EXP))

    eval_rows = []
    fullarc_rows = []
    final_selected: dict[int, bytes] = {}
    final_eval: dict[int, BaseTask] = {}
    accepted_steps: list[dict] = []
    generated = 0

    for task_id in task_ids:
        task = base[task_id]
        current_raw = raws[task_id]
        current_base = task
        task_improved = False
        for pass_idx in range(1, MAX_PASSES_PER_TASK + 1):
            best_eval = None
            best_raw = None
            best_fullarc = None
            for candidate in safe_candidates(task_id, current_raw, task.route, pass_idx):
                generated += 1
                evaluation, accepted_raw = evaluate_candidate(utils, candidate, current_base, ARC_GEN_SAMPLE, EXP_DIR)
                eval_rows.append(asdict(evaluation))
                if accepted_raw is None:
                    continue
                ok, reason, passed, failed = validate_examples(utils, accepted_raw, task_id, -1)
                fullarc_row = {
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
                fullarc_rows.append(fullarc_row)
                if not ok:
                    continue
                if best_eval is None or float(evaluation.candidate_points) > float(best_eval.candidate_points):
                    best_eval = evaluation
                    best_raw = accepted_raw
                    best_fullarc = fullarc_row
            if best_eval is None or best_raw is None or best_fullarc is None:
                break
            accepted = asdict(best_eval)
            accepted["validation_status"] = f"full_arc_{best_fullarc['passed']}_pass_{best_fullarc['failed']}_fail"
            accepted_steps.append(accepted)
            current_raw = best_raw
            current_base = BaseTask(task_id, int(best_eval.candidate_cost), float(best_eval.candidate_points), EXP_ID, best_eval.template_name, task.route, current_raw)
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

    with (EXP_DIR / "fullarc_candidate_audit.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["task_id", "template_name", "sample20_cost", "sample20_points", "full_arc_ok", "passed", "failed", "reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fullarc_rows)

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
                "validation_status": "full_arc_pass",
                "status": "improved",
                "reason": "deep full-arc safe bypass",
                "sha256": hashlib.sha256(final_selected[task_id]).hexdigest(),
            }
        )
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(selected_rows)

    base_result = json.loads((BASE_EXP / "result.json").read_text(encoding="utf-8"))
    base_local = float(base_result["local_estimate"])
    delta = sum(float(row["candidate_points"]) - float(row["baseline_points"]) for row in selected_rows)
    local_estimate = base_local + delta
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-06",
        "status": "improved" if selected_rows else "no_gain",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_local_estimate": base_local,
        "task_ids": task_ids,
        "safe_ops": sorted(SAFE_OPS),
        "max_passes_per_task": MAX_PASSES_PER_TASK,
        "candidate_rows": len(eval_rows),
        "fullarc_audited_candidate_count": len(fullarc_rows),
        "accepted_step_count": len(accepted_steps),
        "improved_task_count": len(selected_rows),
        "local_estimate_delta": delta,
        "local_estimate": local_estimate,
        "gap_to_6500": 6500.0 - local_estimate,
        "submission_decision": "submit_threshold_reached" if local_estimate >= 6500.0 else "no_submit: below 6500 threshold",
        "selected_tasks": [row["task_id"] for row in selected_rows],
        "top_selected": sorted(selected_rows, key=lambda row: float(row["candidate_points"]) - float(row["baseline_points"]), reverse=True)[:20],
        "accepted_steps": accepted_steps[:120],
        "zip_sanity": sanity,
        "leakage_risk": "medium: full-arc gated edits, but base still includes high-risk lookup artifacts.",
        "overfitting_risk": "medium: targeted by observed safe patterns; private distribution risk remains.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## Hypothesis

exp037/038で全arc-gen通過実績があるtaskとopに絞って深いpassを回せば、広い探索より安全かつ高速に追加cost削減を得られる。

## Result

- base: `{result["base_exp"]}`
- task count: `{len(task_ids)}`
- safe ops: `{", ".join(result["safe_ops"])}`
- candidate rows: `{result["candidate_rows"]}`
- full-arc audited candidates: `{result["fullarc_audited_candidate_count"]}`
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
