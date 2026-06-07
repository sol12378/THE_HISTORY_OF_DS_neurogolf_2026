from __future__ import annotations

import csv
import json
import math
import pathlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP_ID = "exp048_submit_safe_seed_inventory"
EXP_DIR = ROOT / "experiments" / EXP_ID
STRICT_EXP = ROOT / "experiments" / "exp005_top_cost_rewrite_strict"
TEACHER_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
ROUTE_EXP = ROOT / "experiments" / "exp011_gpu_route_classifier"


def point(cost: float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, cost)))


@dataclass
class TaskRow:
    task_id: int
    route: str
    strict_source: str
    strict_cost: int
    strict_points: float
    teacher_source: str
    teacher_template: str
    teacher_cost: int
    teacher_points: float
    teacher_gain_vs_strict: float
    teacher_cost_ratio: float
    strict_top_rank: int
    priority: str
    action: str


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    routes = {int(row["task_id"]): row["pred_route"] for row in read_csv(ROUTE_EXP / "route_predictions.csv")}
    strict_rows = {int(row["task_id"]): row for row in read_csv(STRICT_EXP / "rewrite_manifest.csv")}
    teacher_rows = {int(row["task_id"]): row for row in read_csv(TEACHER_EXP / "selected_manifest.csv")}

    task_rows: list[TaskRow] = []
    for task_id in sorted(strict_rows):
        strict = strict_rows[task_id]
        teacher = teacher_rows[task_id]
        strict_cost = int(float(strict["new_cost"]))
        teacher_cost = int(float(teacher["cost"]))
        strict_points = float(strict["new_points"])
        teacher_points = float(teacher["local_points"])
        gain = teacher_points - strict_points
        ratio = strict_cost / max(1, teacher_cost)
        route = routes.get(task_id, teacher.get("route", "unknown"))
        if gain >= 2.0 and teacher_cost < strict_cost:
            priority = "P0"
            action = "teacher explains large gain but is unsafe; mine explicit rule first"
        elif strict_cost >= 30000:
            priority = "P1"
            action = "high strict cost; search submit-safe DSL replacement"
        elif strict_cost >= 9000:
            priority = "P2"
            action = "medium strict cost; include in family rule search"
        else:
            priority = "P3"
            action = "keep strict seed unless family compiler finds cheap rule"
        task_rows.append(
            TaskRow(
                task_id=task_id,
                route=route,
                strict_source=strict["source"],
                strict_cost=strict_cost,
                strict_points=strict_points,
                teacher_source=teacher["source"],
                teacher_template=teacher["template_name"],
                teacher_cost=teacher_cost,
                teacher_points=teacher_points,
                teacher_gain_vs_strict=gain,
                teacher_cost_ratio=ratio,
                strict_top_rank=0,
                priority=priority,
                action=action,
            )
        )

    ranked = sorted(task_rows, key=lambda row: (-row.strict_cost, row.task_id))
    ranks = {row.task_id: idx + 1 for idx, row in enumerate(ranked)}
    task_rows = [TaskRow(**{**asdict(row), "strict_top_rank": ranks[row.task_id]}) for row in task_rows]

    family: dict[str, dict[str, Any]] = defaultdict(lambda: {"task_count": 0, "strict_score": 0.0, "teacher_score": 0.0, "strict_cost_sum": 0, "teacher_cost_sum": 0, "p0_count": 0, "p1_count": 0})
    for row in task_rows:
        item = family[row.route]
        item["task_count"] += 1
        item["strict_score"] += row.strict_points
        item["teacher_score"] += row.teacher_points
        item["strict_cost_sum"] += row.strict_cost
        item["teacher_cost_sum"] += row.teacher_cost
        if row.priority == "P0":
            item["p0_count"] += 1
        if row.priority == "P1":
            item["p1_count"] += 1

    family_rows = []
    for route, item in family.items():
        family_rows.append(
            {
                "route": route,
                **item,
                "teacher_gain_vs_strict": item["teacher_score"] - item["strict_score"],
                "strict_avg_points": item["strict_score"] / item["task_count"],
                "teacher_avg_points": item["teacher_score"] / item["task_count"],
            }
        )
    family_rows = sorted(family_rows, key=lambda row: (-row["teacher_gain_vs_strict"], row["route"]))

    p0 = sorted([row for row in task_rows if row.priority == "P0"], key=lambda row: (-row.teacher_gain_vs_strict, -row.strict_cost, row.task_id))
    p1 = sorted([row for row in task_rows if row.priority == "P1"], key=lambda row: (-row.strict_cost, row.task_id))
    strict_score = sum(row.strict_points for row in task_rows)
    teacher_score = sum(row.teacher_points for row in task_rows)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "inventory_ready",
        "strict_seed_exp": str(STRICT_EXP.relative_to(ROOT)),
        "teacher_exp": str(TEACHER_EXP.relative_to(ROOT)),
        "strict_seed_score": strict_score,
        "strict_seed_full_validation": "400/400 all arc-gen pass per exp005 result.json",
        "teacher_local_score": teacher_score,
        "teacher_gain_vs_strict": teacher_score - strict_score,
        "public_lb_lesson": "exp041 teacher-derived local upper scored 3417.71 LB; teacher gains are not submit-safe until converted to explicit rules.",
        "p0_teacher_gain_task_count": len(p0),
        "p1_high_strict_cost_task_count": len(p1),
        "top_p0_tasks": [asdict(row) for row in p0[:30]],
        "top_p1_tasks": [asdict(row) for row in p1[:30]],
        "family_summary": family_rows,
        "decision": "Use exp005 as submit-safe seed and exp023/041 only as teacher. Next search targets P0 teacher-gain tasks by family, not local artifact surgery.",
        "leakage_risk": "low for strict seed; high for teacher artifacts until rule-compressed.",
        "overfitting_risk": "medium: full arc-gen pass is necessary but not sufficient; use family holdout next.",
    }
    with (EXP_DIR / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (EXP_DIR / "task_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TaskRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in sorted(task_rows, key=lambda row: (row.priority, -row.teacher_gain_vs_strict, row.task_id))])
    with (EXP_DIR / "family_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(family_rows[0].keys()))
        writer.writeheader()
        writer.writerows(family_rows)

    notes = f"""# {EXP_ID}

## 仮説

exp041のLB崩壊後は、exp005 strict full-validation bundleをsubmit-safe seedとし、exp023/041系はteacherとしてのみ使うべき。

## 結果

- strict seed score: {strict_score:.6f}
- teacher local score: {teacher_score:.6f}
- teacher gain vs strict: {teacher_score - strict_score:.6f}
- P0 teacher-gain tasks: {len(p0)}
- P1 high strict-cost tasks: {len(p1)}

## 解釈

exp005は400/400 all arc-gen pass済みの信頼seed。exp023/041はlocal scoreが高いがLB崩壊済みなので、その差分は「答え」ではなく「圧縮すべきteacher signal」として扱う。

## 次

P0 taskをfamily別にrule-compressする。最初はteacher gainが大きく、routeが `sparse_edit_or_object_completion` または `crop_or_resize` のtaskから始める。

## リスク

- leakage risk: strict seedは低、teacherは高。
- overfitting risk: family holdout未導入のため中。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
