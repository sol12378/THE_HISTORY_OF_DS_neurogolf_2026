from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from collections import Counter
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from experiments.phase1_rewrite_utils import (
    ROOT,
    CandidateEval,
    candidate_fields,
    crop_candidates,
    evaluate_candidate,
    examples_for,
    load_base_tasks,
    load_neurogolf_utils,
    load_task,
    sha256,
    sparse_candidates,
    top_task_ids,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp019_boundary_flood_fill_campaign"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp016_top100_rewrite_campaign"
OUTPUT_ZIP = EXP_DIR / "submission.zip"


def candidate_set(task_id: int, route: str, examples: list[dict]) -> list:
    if route == "crop_or_resize":
        return crop_candidates(task_id, examples, route)
    return sparse_candidates(task_id, examples, route)


def parse_task_ids(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--arc-gen-sample", type=int, default=20)
    parser.add_argument("--task-ids", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks(BASE_EXP)
    explicit_task_ids = parse_task_ids(args.task_ids)
    if explicit_task_ids is not None:
        task_ids = explicit_task_ids
    else:
        task_ids = top_task_ids(base, min(args.top_k, 10) if args.dry_run else args.top_k)

    final_raw = {task_id: task.raw for task_id, task in base.items()}
    candidate_rows: list[dict] = []
    selected_rows: list[dict] = []
    improved: dict[int, CandidateEval] = {}
    started = time.time()

    for idx, task_id in enumerate(task_ids, start=1):
        print(f"{EXP_ID} {idx}/{len(task_ids)} task{task_id:03d}", flush=True)
        task = base[task_id]
        task_examples = examples_for(load_task(task_id), args.arc_gen_sample)
        best_record: CandidateEval | None = None
        best_raw: bytes | None = None
        for candidate in candidate_set(task_id, task.route, task_examples):
            record, raw = evaluate_candidate(utils, candidate, task, args.arc_gen_sample, EXP_DIR)
            candidate_rows.append(record.__dict__)
            if raw is None or record.status != "improved" or not isinstance(record.candidate_cost, int):
                continue
            if best_record is None or int(record.candidate_cost) < int(best_record.candidate_cost):
                best_record = record
                best_raw = raw
        if best_record is not None and best_raw is not None:
            improved[task_id] = best_record
            final_raw[task_id] = best_raw

    write_zip(OUTPUT_ZIP, final_raw)
    for task_id in sorted(base):
        task = base[task_id]
        if task_id in improved:
            record = improved[task_id]
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": f"{EXP_ID}_{record.template_name}",
                    "template_name": record.template_name,
                    "route": record.route,
                    "cost": record.candidate_cost,
                    "local_points": record.candidate_points,
                    "file_bytes": record.file_bytes,
                    "status": "improved",
                    "reason": record.reason,
                    "sha256": record.sha256,
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
                    "reason": "no accepted lower-cost boundary flood candidate",
                    "sha256": sha256(task.raw),
                }
            )

    with (EXP_DIR / "candidate_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(candidate_rows)
    selected_fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected_fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    baseline_score = sum(task.points for task in base.values())
    new_score = sum(float(row["local_points"]) for row in selected_rows)
    status_counts = Counter(row["status"] for row in candidate_rows)
    reject_counts = Counter(row["reason"] for row in candidate_rows if row["status"] not in {"improved", "no_cost_gain"})
    template_counts = Counter(row["template_name"] for row in selected_rows if row["status"] == "improved")
    remaining_top = sorted(selected_rows, key=lambda row: (-int(float(row["cost"])), int(row["task_id"])))[:20]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "target_reached" if new_score >= 6500 else "below_target",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "task_ids": task_ids,
        "top_k": args.top_k,
        "arc_gen_sample": args.arc_gen_sample,
        "baseline_local_estimate": baseline_score,
        "new_local_estimate": new_score,
        "delta": new_score - baseline_score,
        "gap_to_6500": 6500 - new_score,
        "gap_to_6750": 6750 - new_score,
        "gap_to_7000": 7000 - new_score,
        "gap_to_7400": 7400 - new_score,
        "gap_to_7600": 7600 - new_score,
        "improved_task_count": len(improved),
        "improved_tasks": sorted(improved),
        "improved_by_template": dict(template_counts),
        "candidate_status_counts": dict(status_counts),
        "top_reject_reasons": dict(reject_counts.most_common(20)),
        "remaining_top_cost_tasks": remaining_top,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "pipeline_role": "boundary flood fill campaign for region/line high-cost tasks.",
        "leakage_risk": "high: base exp016 includes signature lookup local-upper-bound candidates.",
        "overfitting_risk": "high until full arc-gen and private-like validation are run.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = [
        f"# {EXP_ID}",
        "",
        "## Hypothesis",
        "",
        "背景0を壁で分割するtaskは、境界から到達可能な外側背景と閉領域背景をunrolled ONNXで分ければ、",
        "exp016の高cost public artifactより低costに置換できる。",
        "",
        "## Result",
        "",
        f"- baseline local estimate: `{baseline_score:.6f}`",
        f"- new local estimate: `{new_score:.6f}`",
        f"- delta: `{new_score - baseline_score:.6f}`",
        f"- gap to 6500: `{6500 - new_score:.6f}`",
        f"- improved tasks: `{sorted(improved)}`",
        "",
        "## Risks",
        "",
        "- leakage risk: high。baseはexp016でsignature lookupを含む。",
        "- overfitting risk: high。sample-localであり、full arc-gen前の評価。",
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
