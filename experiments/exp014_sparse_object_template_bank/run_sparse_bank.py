from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from experiments.phase1_rewrite_utils import (
    ROOT,
    candidate_fields,
    evaluate_candidate,
    examples_for,
    load_base_tasks,
    load_neurogolf_utils,
    load_task,
    sparse_candidates,
    top_task_ids,
)


EXP_DIR = ROOT / "experiments" / "exp014_sparse_object_template_bank"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--arc-gen-sample", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()
    task_ids = [tid for tid in top_task_ids(base, min(args.top_k, 10) if args.dry_run else args.top_k) if base[tid].route != "crop_or_resize"]
    records = []
    best_by_task = {}
    started = time.time()
    for idx, task_id in enumerate(task_ids, start=1):
        if idx == 1 or idx % 10 == 0:
            print(f"Sparse bank {idx}/{len(task_ids)} task{task_id:03d}", flush=True)
        task_examples = examples_for(load_task(task_id), args.arc_gen_sample)
        best = None
        for candidate in sparse_candidates(task_id, task_examples, base[task_id].route):
            record, raw = evaluate_candidate(utils, candidate, base[task_id], args.arc_gen_sample, EXP_DIR)
            records.append(record.__dict__)
            if raw is not None and record.status == "improved":
                if best is None or int(record.candidate_cost) < int(best["candidate_cost"]):
                    best = record.__dict__.copy()
        if best is not None:
            best_by_task[task_id] = best

    with (EXP_DIR / "candidate_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(records)
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields = candidate_fields()
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(best_by_task.values())
    status_counts = Counter(row["status"] for row in records)
    result = {
        "exp_id": "exp014_sparse_object_template_bank",
        "date": "2026-06-05",
        "status": "bank_complete",
        "top_k": args.top_k,
        "arc_gen_sample": args.arc_gen_sample,
        "target_task_count": len(task_ids),
        "improved_task_count": len(best_by_task),
        "candidate_status_counts": dict(status_counts),
        "runtime_seconds": time.time() - started,
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXP_DIR / "notes.md").write_text(
        "# exp014_sparse_object_template_bank notes\n\n"
        "sparse/object completion routeに対して、固定編集・色写像・近傍fill templateを検証した。\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
