from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


EXP_ID = "exp053_all_task_cost_250_600_inventory"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

TASK_INVENTORY = ROOT / "experiments" / "exp048_submit_safe_seed_inventory" / "task_inventory.csv"
BACKLOG = ROOT / "experiments" / "exp_b001_rule_replacement_backlog" / "rule_replacement_backlog.csv"


def points_for_cost(cost: int) -> float:
    return 25.0 - math.log(float(cost))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bucket_for_cost(cost: int) -> str:
    if cost <= 250:
        return "already_<=250"
    if cost <= 600:
        return "already_251_600"
    if cost <= 9000:
        return "601_9000"
    if cost <= 50000:
        return "9001_50000"
    if cost <= 100000:
        return "50001_100000"
    return "100000_plus"


def compression_class(cost: int) -> str:
    ratio_to_600 = cost / 600.0
    if cost <= 600:
        return "done"
    if ratio_to_600 <= 5:
        return "small"
    if ratio_to_600 <= 20:
        return "medium"
    if ratio_to_600 <= 100:
        return "large"
    return "extreme"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inv_rows = load_csv(TASK_INVENTORY)
    backlog_rows = load_csv(BACKLOG) if BACKLOG.exists() else []
    backlog_by_task = {int(r["task_id"]): r for r in backlog_rows}

    target250_points = points_for_cost(250)
    target600_points = points_for_cost(600)

    task_rows: list[dict[str, object]] = []
    for r in inv_rows:
        task_id = int(r["task_id"])
        strict_cost = int(float(r["strict_cost"]))
        strict_points = float(r["strict_points"])
        teacher_cost = int(float(r["teacher_cost"]))
        teacher_points = float(r["teacher_points"])
        route = r["route"]
        backlog = backlog_by_task.get(task_id, {})
        family = backlog.get("family") or route
        priority = backlog.get("priority") or r.get("priority") or "P2"

        gain_to_600 = max(0.0, target600_points - strict_points)
        gain_to_250 = max(0.0, target250_points - strict_points)
        teacher_gain_to_600 = max(0.0, target600_points - teacher_points)
        teacher_gain_to_250 = max(0.0, target250_points - teacher_points)

        if strict_cost <= 600:
            next_action = "keep_current; already in requested 250-600 band"
        elif teacher_cost <= 600:
            next_action = "rule-compress teacher; teacher cost is already in target band but unsafe until explainable"
        elif teacher_cost < strict_cost:
            next_action = "mine teacher delta into submit-safe DSL, then lower to <=600"
        else:
            next_action = "new DSL/DAG synthesis from task examples; teacher gives no cost help"

        task_rows.append(
            {
                "task_id": task_id,
                "family": family,
                "route": route,
                "priority": priority,
                "strict_cost": strict_cost,
                "strict_points": strict_points,
                "teacher_cost": teacher_cost,
                "teacher_points": teacher_points,
                "cost_bucket": bucket_for_cost(strict_cost),
                "compression_class_to_600": compression_class(strict_cost),
                "cost_ratio_to_600": strict_cost / 600.0,
                "cost_ratio_to_250": strict_cost / 250.0,
                "gain_to_600": gain_to_600,
                "gain_to_250": gain_to_250,
                "teacher_remaining_gain_to_600": teacher_gain_to_600,
                "teacher_remaining_gain_to_250": teacher_gain_to_250,
                "teacher_gain_vs_strict": float(r["teacher_gain_vs_strict"]),
                "rule_target": backlog.get("rule_target", ""),
                "lowering_target": backlog.get("lowering_target", ""),
                "reject_lowering": backlog.get("reject_lowering", ""),
                "next_action": next_action,
            }
        )

    task_rows.sort(key=lambda x: (float(x["gain_to_600"]), int(x["strict_cost"])), reverse=True)

    family_acc: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "family": "",
            "task_count": 0,
            "already_<=250": 0,
            "already_251_600": 0,
            "need_above_600": 0,
            "strict_cost_sum": 0,
            "teacher_cost_sum": 0,
            "strict_points_sum": 0.0,
            "gain_to_600_sum": 0.0,
            "gain_to_250_sum": 0.0,
            "teacher_gain_vs_strict_sum": 0.0,
            "extreme_or_large_count": 0,
            "top_tasks": [],
        }
    )
    for r in task_rows:
        family = str(r["family"])
        acc = family_acc[family]
        acc["family"] = family
        acc["task_count"] = int(acc["task_count"]) + 1
        if int(r["strict_cost"]) <= 250:
            acc["already_<=250"] = int(acc["already_<=250"]) + 1
        elif int(r["strict_cost"]) <= 600:
            acc["already_251_600"] = int(acc["already_251_600"]) + 1
        else:
            acc["need_above_600"] = int(acc["need_above_600"]) + 1
        acc["strict_cost_sum"] = int(acc["strict_cost_sum"]) + int(r["strict_cost"])
        acc["teacher_cost_sum"] = int(acc["teacher_cost_sum"]) + int(r["teacher_cost"])
        acc["strict_points_sum"] = float(acc["strict_points_sum"]) + float(r["strict_points"])
        acc["gain_to_600_sum"] = float(acc["gain_to_600_sum"]) + float(r["gain_to_600"])
        acc["gain_to_250_sum"] = float(acc["gain_to_250_sum"]) + float(r["gain_to_250"])
        acc["teacher_gain_vs_strict_sum"] = float(acc["teacher_gain_vs_strict_sum"]) + float(r["teacher_gain_vs_strict"])
        if str(r["compression_class_to_600"]) in {"large", "extreme"}:
            acc["extreme_or_large_count"] = int(acc["extreme_or_large_count"]) + 1
        top_tasks = acc["top_tasks"]
        assert isinstance(top_tasks, list)
        if len(top_tasks) < 8:
            top_tasks.append(int(r["task_id"]))

    family_rows = []
    for acc in family_acc.values():
        task_count = int(acc["task_count"])
        family_rows.append(
            {
                "family": acc["family"],
                "task_count": task_count,
                "already_<=250": acc["already_<=250"],
                "already_251_600": acc["already_251_600"],
                "need_above_600": acc["need_above_600"],
                "strict_cost_sum": acc["strict_cost_sum"],
                "teacher_cost_sum": acc["teacher_cost_sum"],
                "strict_points_sum": acc["strict_points_sum"],
                "gain_to_600_sum": acc["gain_to_600_sum"],
                "gain_to_250_sum": acc["gain_to_250_sum"],
                "teacher_gain_vs_strict_sum": acc["teacher_gain_vs_strict_sum"],
                "extreme_or_large_count": acc["extreme_or_large_count"],
                "top_tasks": " ".join(str(x) for x in acc["top_tasks"]),
            }
        )
    family_rows.sort(key=lambda x: float(x["gain_to_600_sum"]), reverse=True)

    bucket_acc: dict[str, dict[str, object]] = defaultdict(
        lambda: {"cost_bucket": "", "task_count": 0, "gain_to_600_sum": 0.0, "gain_to_250_sum": 0.0}
    )
    for r in task_rows:
        bucket = str(r["cost_bucket"])
        acc = bucket_acc[bucket]
        acc["cost_bucket"] = bucket
        acc["task_count"] = int(acc["task_count"]) + 1
        acc["gain_to_600_sum"] = float(acc["gain_to_600_sum"]) + float(r["gain_to_600"])
        acc["gain_to_250_sum"] = float(acc["gain_to_250_sum"]) + float(r["gain_to_250"])
    bucket_rows = sorted(bucket_acc.values(), key=lambda x: str(x["cost_bucket"]))

    task_fields = [
        "task_id",
        "family",
        "route",
        "priority",
        "strict_cost",
        "strict_points",
        "teacher_cost",
        "teacher_points",
        "cost_bucket",
        "compression_class_to_600",
        "cost_ratio_to_600",
        "cost_ratio_to_250",
        "gain_to_600",
        "gain_to_250",
        "teacher_remaining_gain_to_600",
        "teacher_remaining_gain_to_250",
        "teacher_gain_vs_strict",
        "rule_target",
        "lowering_target",
        "reject_lowering",
        "next_action",
    ]
    family_fields = [
        "family",
        "task_count",
        "already_<=250",
        "already_251_600",
        "need_above_600",
        "strict_cost_sum",
        "teacher_cost_sum",
        "strict_points_sum",
        "gain_to_600_sum",
        "gain_to_250_sum",
        "teacher_gain_vs_strict_sum",
        "extreme_or_large_count",
        "top_tasks",
    ]
    write_csv(OUT_DIR / "task_cost_targets.csv", task_rows, task_fields)
    write_csv(OUT_DIR / "family_cost_gap.csv", family_rows, family_fields)
    write_csv(OUT_DIR / "cost_bucket_summary.csv", bucket_rows, ["cost_bucket", "task_count", "gain_to_600_sum", "gain_to_250_sum"])

    strict_total = sum(float(r["strict_points"]) for r in task_rows)
    projected_600 = sum(max(float(r["strict_points"]), target600_points) for r in task_rows)
    projected_250 = sum(max(float(r["strict_points"]), target250_points) for r in task_rows)
    already_600 = sum(1 for r in task_rows if int(r["strict_cost"]) <= 600)
    already_250 = sum(1 for r in task_rows if int(r["strict_cost"]) <= 250)

    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-06",
        "status": "inventory_ready",
        "purpose": "all 400 task cost-gap map for lowering tasks into the 250-600 cost band",
        "source": {
            "strict_seed_inventory": str(TASK_INVENTORY.relative_to(ROOT)),
            "rule_backlog": str(BACKLOG.relative_to(ROOT)),
        },
        "score_formula": "points = 25 - ln(cost), inferred from existing task_inventory cost/points pairs",
        "task_count": len(task_rows),
        "strict_seed_score_from_inventory": strict_total,
        "target_600_points_per_task": target600_points,
        "target_250_points_per_task": target250_points,
        "already_cost_le_600": already_600,
        "already_cost_le_250": already_250,
        "need_cost_reduction_to_600": len(task_rows) - already_600,
        "projected_score_if_all_cost_le_600_floor": projected_600,
        "projected_gain_if_all_cost_le_600_floor": projected_600 - strict_total,
        "projected_score_if_all_cost_le_250_floor": projected_250,
        "projected_gain_if_all_cost_le_250_floor": projected_250 - strict_total,
        "margin_over_7700_at_600_floor": projected_600 - 7700.0,
        "margin_over_7700_at_250_floor": projected_250 - 7700.0,
        "top_family_gap_to_600": family_rows[:8],
        "top_task_gap_to_600": task_rows[:25],
        "decision": "Use task_cost_targets.csv as the master queue for all-task 250-600 lowering; start with largest family/task gaps, but require submit-safe explainable rules before replacing strict seed artifacts.",
        "leakage_risk": "medium: teacher costs and backlog labels are diagnostic only; replacements must be rule-based and all-arc-gen validated.",
        "overfitting_risk": "medium-high: all-task cost target can encourage artifact memorization; acceptance requires family holdout/Kaggle calibration before large bundles.",
        "outputs": {
            "task_cost_targets": "task_cost_targets.csv",
            "family_cost_gap": "family_cost_gap.csv",
            "cost_bucket_summary": "cost_bucket_summary.csv",
            "notes": "notes.md",
        },
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = f"""# {EXP_ID}

## Hypothesis

7700へ進むには、局所的なartifact surgeryではなく、全400 taskをcost 250〜600台へ落とすための差分地図が必要である。

## Result

- strict seed inventory score: {strict_total:.6f}
- cost<=600 already: {already_600}/400
- cost<=250 already: {already_250}/400
- need reduction to <=600: {len(task_rows) - already_600}/400
- projected score if every task is at least cost<=600: {projected_600:.6f}
- projected gain to <=600 floor: {projected_600 - strict_total:.6f}
- projected score if every task is at least cost<=250: {projected_250:.6f}
- projected gain to <=250 floor: {projected_250 - strict_total:.6f}
- margin over 7700 at <=600 floor: {projected_600 - 7700.0:.6f}
- margin over 7700 at <=250 floor: {projected_250 - 7700.0:.6f}

## Interpretation

cost<=600 floorだけでも7700を超える推定になる。ただし、これは「全taskを600以下にできる」という強い条件であり、既存teacherをそのまま使う意味ではない。

最大の実装課題は、signature lookup/current高cost taskを説明可能なDSL/DAGに変換し、full-grid GatherND、large ScatterND、dynamic MatMul、長いWhere chainを避けるloweringを作ること。

## Next

1. `task_cost_targets.csv` をmaster queueとして使う。
2. family gap上位から、submit-safe rule searcherを実装する。
3. 最初のcalibration bundleは小さくし、local/LB差を再測定する。

## Risk

- leakage risk: medium。teacher artifactはoracleであり、提出候補ではない。
- overfitting risk: medium-high。all arc-gen exactだけではhidden汎化の証明にならないため、family holdoutと段階submissionが必要。
"""
    (OUT_DIR / "notes.md").write_text(notes, encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
