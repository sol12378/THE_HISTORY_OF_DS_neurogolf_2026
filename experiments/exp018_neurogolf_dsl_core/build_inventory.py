from __future__ import annotations

import csv
import json
import math
import pathlib
import sys
from collections import Counter
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurogolf.synthesis.dsl import PRIMITIVE_CATALOG
from neurogolf.synthesis.inventory import task_feature_row


EXP_ID = "exp018_neurogolf_dsl_core"
EXP_DIR = ROOT / "experiments" / EXP_ID
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
BASE_EXP = ROOT / "experiments" / "exp016_top100_rewrite_campaign"
ROUTE_EXP = ROOT / "experiments" / "exp011_gpu_route_classifier"


def point(cost: int | float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def load_routes() -> dict[int, str]:
    routes: dict[int, str] = {}
    route_path = ROUTE_EXP / "route_predictions.csv"
    if not route_path.exists():
        return routes
    with route_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            routes[int(row["task_id"])] = row.get("pred_route", "")
    return routes


def load_base_rows() -> dict[int, dict[str, Any]]:
    selected = BASE_EXP / "selected_manifest.csv"
    if not selected.exists():
        raise FileNotFoundError(selected)
    rows: dict[int, dict[str, Any]] = {}
    with selected.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            task_id = int(row["task_id"])
            cost = int(float(row["cost"]))
            rows[task_id] = {
                "source": row.get("source", ""),
                "template_name": row.get("template_name", ""),
                "cost": cost,
                "points": float(row.get("local_points") or point(cost)),
            }
    return rows


def load_task(task_id: int) -> dict[str, Any]:
    return json.loads((DATA_DIR / f"task{task_id:03d}.json").read_text(encoding="utf-8"))


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_catalog(path: pathlib.Path) -> None:
    lines = [
        "# exp018 Primitive Catalog",
        "",
        "7600を見据え、タスクを小さな合成primitiveへ分解するための初期カタログ。",
        "ここでは提出ONNXの実装ではなく、探索・分類・優先順位付けの共通語彙を固定する。",
        "",
        "| Primitive | Family | ONNX lowering | Leakage risk | Private risk | Description |",
        "|---|---|---|---|---|---|",
    ]
    for spec in PRIMITIVE_CATALOG:
        lines.append(
            f"| `{spec.name}` | `{spec.family}` | {spec.onnx_lowering} | "
            f"{spec.leakage_risk} | {spec.private_risk} | {spec.description} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    routes = load_routes()
    base_rows = load_base_rows()
    rows: list[dict[str, object]] = []
    for task_id in range(1, 401):
        task = load_task(task_id)
        base = base_rows[task_id]
        rows.append(
            task_feature_row(
                task_id=task_id,
                task=task,
                route=routes.get(task_id, ""),
                source=str(base["source"]),
                template_name=str(base["template_name"]),
                cost=int(base["cost"]),
                points=float(base["points"]),
            )
        )

    rows.sort(key=lambda row: (-int(row["current_cost"]), int(row["task_id"])))
    for rank, row in enumerate(rows, start=1):
        row["cost_rank"] = rank
        row["priority_band"] = "top100" if rank <= 100 else "top200" if rank <= 200 else "tail"

    write_csv(EXP_DIR / "task_registry.csv", rows)
    write_catalog(EXP_DIR / "primitive_catalog.md")

    family_counts = Counter(str(row["synthesis_family"]) for row in rows)
    source_counts = Counter(str(row["current_source"]) for row in rows)
    top_remaining = [
        {
            "task_id": int(row["task_id"]),
            "family": row["synthesis_family"],
            "route": row["route_prediction"],
            "cost": int(row["current_cost"]),
            "source": row["current_source"],
            "template": row["current_template"],
        }
        for row in rows[:30]
    ]
    local_estimate = sum(float(row["current_points"]) for row in rows)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "inventory_complete",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "task_count": len(rows),
        "local_estimate_from_base": local_estimate,
        "gap_to_6500": max(0.0, 6500.0 - local_estimate),
        "gap_to_7000": max(0.0, 7000.0 - local_estimate),
        "gap_to_7600": max(0.0, 7600.0 - local_estimate),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts_top": dict(source_counts.most_common(12)),
        "top_remaining_cost_tasks": top_remaining,
        "primitive_count": len(PRIMITIVE_CATALOG),
        "next_required_families": [
            "boundary_flood_fill",
            "rectangular_room_fill",
            "point_to_line_pattern",
            "crop_resize_with_object_anchor",
            "static_lookup_compression",
        ],
        "leakage_risk": "high: base exp016 includes signature lookup candidates and sample-local validation.",
        "overfitting_risk": "high until task-family programs pass full arc-gen and private-like holdout.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    notes = [
        f"# {EXP_ID}",
        "",
        "## Hypothesis",
        "",
        "7600を狙うには、公開artifact blendや単発template追加ではなく、タスクをDSL familyへ分解して、",
        "候補生成、検証、ONNX lowering、cost最適化を同じ形式で回す必要がある。",
        "",
        "## Result",
        "",
        f"- base: `{result['base_exp']}`",
        f"- local estimate from base: `{local_estimate:.6f}`",
        f"- gap to 6500: `{result['gap_to_6500']:.6f}`",
        f"- gap to 7600: `{result['gap_to_7600']:.6f}`",
        f"- primitive count: `{len(PRIMITIVE_CATALOG)}`",
        "",
        "## Family Counts",
        "",
    ]
    for family, count in sorted(family_counts.items()):
        notes.append(f"- `{family}`: {count}")
    notes.extend(
        [
            "",
            "## Interpretation",
            "",
            "exp017の単純な反復近傍fillはsample20では改善しなかった。",
            "残り上位cost taskは、領域分割、線/部屋構造、点から線・パターン生成、object anchor cropの比率が高い。",
            "次PDCAは `boundary_flood_fill` と `rectangular_room_fill` をONNX loweringまで実装し、task187/198/137/203/286周辺を直接狙う。",
            "",
            "## Risks",
            "",
            "- leakage risk: high。現baseはexp016で、signature lookupを含むlocal upper bound。",
            "- overfitting risk: high。full arc-genとprivate-like holdout前のfamily分類である。",
        ]
    )
    (EXP_DIR / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
