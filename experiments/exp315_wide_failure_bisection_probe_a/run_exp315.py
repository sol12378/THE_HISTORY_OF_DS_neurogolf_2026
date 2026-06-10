from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.public_zero_probe_utils import build_probe


EXP_ID = "exp315_wide_failure_bisection_probe_a"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp297_exp262_skip_task048_336_fresh_candidates"
BASE_PUBLIC_LB = 6008.96

TARGETS = [97, 193, 192, 197, 324, 137, 335, 224, 213, 338, 131, 275, 138, 243, 359, 101]


def load_task_points() -> dict[int, float]:
    rows: dict[int, float] = {}
    path = ROOT / "experiments" / "exp136_private_failure_subset_inventory" / "task_risk_inventory.csv"
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["task_id"])] = float(row["points"])
    return rows


def write_subset_sum_collision_audit(target_points: dict[int, float]) -> dict[str, object]:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    target_list = list(target_points)
    sums: dict[float, list[int]] = defaultdict(list)
    for mask in range(1, 1 << len(target_list)):
        total = round(sum(target_points[target_list[i]] for i in range(len(target_list)) if mask >> i & 1), 2)
        if len(sums[total]) < 4:
            sums[total].append(mask)

    collision_rows = []
    for total, masks in sorted(sums.items()):
        if len(masks) <= 1:
            continue
        collision_rows.append(
            {
                "rounded_missing_drop": total,
                "candidate_subsets": ";".join(
                    " ".join(f"{target_list[i]:03d}" for i in range(len(target_list)) if mask >> i & 1)
                    for mask in masks
                ),
                "shown_subset_count": len(masks),
            }
        )

    out = EXP_DIR / "subset_sum_collisions.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rounded_missing_drop", "candidate_subsets", "shown_subset_count"])
        writer.writeheader()
        writer.writerows(collision_rows)
    return {
        "target_count": len(target_list),
        "rounded_subset_sum_count": len(sums),
        "rounded_collision_count": len(collision_rows),
        "collision_audit_path": str(out.relative_to(ROOT)),
    }


def main() -> None:
    all_points = load_task_points()
    target_points = {tid: all_points[tid] for tid in TARGETS}
    collision_audit = write_subset_sum_collision_audit(target_points)
    result = build_probe(
        exp_dir=EXP_DIR,
        exp_id=EXP_ID,
        purpose=(
            "A-3 wide bisection probe: intentionally fail 16 unprobed high-risk tasks on the current exp297 "
            "best bundle. If the observed drop is smaller than the all-alive drop, the missing mass identifies "
            "one or more remaining public-zero tasks; subset ambiguity is recorded for follow-up split probes."
        ),
        base_exp=BASE_EXP,
        base_public_lb=BASE_PUBLIC_LB,
        target_points=target_points,
        submission_decision="submit_probe_after_sanity; do not submit another bisection probe until this score completes",
        overfitting_risk=(
            "medium: this is a public diagnostic probe; target selection comes from public-zero risk inventory and "
            "must be followed by correctness-first repair, not public-score-only adoption."
        ),
    )
    result["date"] = "2026-06-11"
    result["interpretation_rule"] = (
        "observed_drop close to expected_drop_if_all_alive means all 16 targets are public-scoring alive; "
        "a smaller drop means the missing drop is the score mass of public-zero targets. Because 16-way rounded "
        "subset sums collide, use subset_sum_collisions.csv and, if needed, a single split follow-up after this "
        "probe completes."
    )
    result["subset_sum_collision_audit"] = collision_audit
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
