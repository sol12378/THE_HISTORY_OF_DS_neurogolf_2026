from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from collections import Counter, defaultdict

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
    point,
    sha256,
    sparse_candidates,
    top_task_ids,
    write_zip,
    zip_sanity,
)


EXP_DIR = ROOT / "experiments" / "exp016_top100_rewrite_campaign"
OUTPUT_ZIP = EXP_DIR / "submission.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--arc-gen-sample", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()
    task_ids = top_task_ids(base, min(args.top_k, 10) if args.dry_run else args.top_k)
    final_raw = {task_id: task.raw for task_id, task in base.items()}
    selected_rows = []
    candidate_rows = []
    improved: dict[int, CandidateEval] = {}
    improved_raw: dict[int, bytes] = {}
    started = time.time()

    for idx, task_id in enumerate(task_ids, start=1):
        if idx == 1 or idx % 10 == 0:
            print(f"Campaign {idx}/{len(task_ids)} task{task_id:03d}", flush=True)
        task = base[task_id]
        task_examples = examples_for(load_task(task_id), args.arc_gen_sample)
        candidates = crop_candidates(task_id, task_examples, task.route) if task.route == "crop_or_resize" else sparse_candidates(task_id, task_examples, task.route)
        best_record: CandidateEval | None = None
        best_raw: bytes | None = None
        for candidate in candidates:
            record, raw = evaluate_candidate(utils, candidate, task, args.arc_gen_sample, EXP_DIR)
            candidate_rows.append(record.__dict__)
            if raw is None or record.status != "improved" or not isinstance(record.candidate_cost, int):
                continue
            if best_record is None or int(record.candidate_cost) < int(best_record.candidate_cost):
                best_record = record
                best_raw = raw
        if best_record is not None and best_raw is not None:
            improved[task_id] = best_record
            improved_raw[task_id] = best_raw
            final_raw[task_id] = best_raw

    write_zip(OUTPUT_ZIP, final_raw)
    for task_id in sorted(base):
        if task_id in improved:
            record = improved[task_id]
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": f"exp016_{record.template_name}",
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
            task = base[task_id]
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
                    "reason": "no accepted lower-cost candidate",
                    "sha256": sha256(task.raw),
                }
            )

    with (EXP_DIR / "candidate_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(candidate_rows)
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    baseline_score = sum(task.points for task in base.values())
    new_score = sum(float(row["local_points"]) for row in selected_rows)
    route_counts = Counter(row["route"] for row in selected_rows if row["status"] == "improved")
    template_counts = Counter(row["template_name"] for row in selected_rows if row["status"] == "improved")
    reject_counts = Counter(row["reason"] for row in candidate_rows if row["status"] not in {"improved", "no_cost_gain"})
    status_counts = Counter(row["status"] for row in candidate_rows)
    remaining_top = sorted(selected_rows, key=lambda row: (-int(float(row["cost"])), int(row["task_id"])))[:20]
    score_if_top100_cap_9000 = new_score
    selected_by_task = {int(row["task_id"]): row for row in selected_rows}
    for task_id in task_ids:
        row = selected_by_task[task_id]
        capped_cost = min(int(float(row["cost"])), 9000)
        score_if_top100_cap_9000 += point(capped_cost) - float(row["local_points"])
    result = {
        "exp_id": "exp016_top100_rewrite_campaign",
        "date": "2026-06-05",
        "status": "target_reached" if new_score >= 6500 else "below_target",
        "base_exp": "exp012_template_factory_core",
        "top_k": args.top_k,
        "arc_gen_sample": args.arc_gen_sample,
        "baseline_local_estimate": baseline_score,
        "new_local_estimate": new_score,
        "delta": new_score - baseline_score,
        "gap_to_6500": 6500 - new_score,
        "score_if_top100_cap_9000": score_if_top100_cap_9000,
        "improved_task_count": len(improved),
        "improved_tasks": sorted(improved),
        "improved_by_route": dict(route_counts),
        "improved_by_template": dict(template_counts),
        "candidate_status_counts": dict(status_counts),
        "top_reject_reasons": dict(reject_counts.most_common(20)),
        "remaining_top_cost_tasks": remaining_top,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "leakage_risk": "high: base exp012 includes signature lookup over arc-gen sample.",
        "overfitting_risk": "high: Phase 1 optimizes local estimate; no Kaggle submit in this phase.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXP_DIR / "notes.md").write_text(
        "# exp016_top100_rewrite_campaign notes\n\n"
        f"- local estimate: `{new_score}`\n"
        f"- status: `{result['status']}`\n"
        "- submitは行わない。6500未達の場合はreject理由とremaining top costを次PDCAに使う。\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
