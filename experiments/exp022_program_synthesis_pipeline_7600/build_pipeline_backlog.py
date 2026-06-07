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
EXP_ID = "exp022_program_synthesis_pipeline_7600"
EXP_DIR = ROOT / "experiments" / EXP_ID
REGISTRY_PATH = ROOT / "experiments" / "exp018_neurogolf_dsl_core" / "task_registry.csv"
BASE_RESULT_PATH = ROOT / "experiments" / "exp016_top100_rewrite_campaign" / "result.json"
PROBE_RESULT_PATH = ROOT / "experiments" / "exp021_diagonal_scatternd_lowmem" / "result.json"


TARGET_COSTS = {
    "phase1_6500_bridge": 9000,
    "phase2_7000_submit_safe": 3000,
    "phase3_7400_private_like": 1000,
    "phase4_7600_contender": 300,
}


FAMILY_REQUIREMENTS = {
    "signature_lookup_current": {
        "system": "lookup compression and rule extraction",
        "lowering": "compressed decision tree, affine signature hashing, or extracted DSL program; avoid memorized arc-gen labels",
        "blocker": "high leakage risk; many gains are local-upper-bound only",
    },
    "crop_resize": {
        "system": "object-anchor crop/resize synthesizer",
        "lowering": "Slice/Gather with constant object anchors, bbox from cheap reductions, small ScatterND for padding",
        "blocker": "bbox/dynamic shape lowering can become expensive or forbidden if NonZero/Compress is used",
    },
    "sparse_edit_or_object_completion": {
        "system": "sparse edit and object completion enumerator",
        "lowering": "GatherND/ScatterND over sparse coordinate programs; small Conv kernels for local masks",
        "blocker": "full-grid Where chains pass validation but lose on official-like cost",
    },
    "region_partition_fill": {
        "system": "region and room-fill synthesizer",
        "lowering": "static line-grid reductions, boundary masks, limited unroll only after cost model approval",
        "blocker": "naive flood-fill unroll is correct but too costly",
    },
    "line_grid_fill": {
        "system": "line, ray, and grid grammar",
        "lowering": "precomputed row/column masks, small convolution kernels, sparse ScatterND",
        "blocker": "needs grammar search across line direction, color role, and stop conditions",
    },
    "point_to_line_pattern": {
        "system": "seed-to-pattern grammar",
        "lowering": "constant kernels, affine coordinate generation, sparse ScatterND",
        "blocker": "must infer period, direction, and color binding without overfitting train examples",
    },
    "color_map": {
        "system": "color role normalizer",
        "lowering": "Equal + Where or small one-hot MatMul",
        "blocker": "usually low upside; useful as a subprogram rather than standalone",
    },
    "same_shape_global_transform": {
        "system": "global transform and artifact surgery",
        "lowering": "Gather, Transpose, Slice, color map, and initializer pruning",
        "blocker": "some current public artifacts are already compact and need graph-level surgery",
    },
}


def point(cost: float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def gain_to_cost(current_cost: int, target_cost: int) -> float:
    if current_cost <= target_cost:
        return 0.0
    return point(target_cost) - point(current_cost)


def load_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def phase_for_rank(rank: int, family: str, cost: int) -> str:
    if rank <= 100 or cost >= 40000:
        return "phase1_6500_bridge"
    if family == "signature_lookup_current" or rank <= 200:
        return "phase2_7000_submit_safe"
    if rank <= 320:
        return "phase3_7400_private_like"
    return "phase4_7600_contender"


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    registry = load_csv(REGISTRY_PATH)
    base_result = load_json(BASE_RESULT_PATH)
    probe_result = load_json(PROBE_RESULT_PATH)
    local_estimate = float(base_result.get("new_local_estimate", 0.0) or base_result.get("local_estimate", 0.0))

    rows: list[dict[str, Any]] = []
    for row in registry:
        task_id = int(row["task_id"])
        cost = int(float(row["current_cost"]))
        rank = int(row["cost_rank"])
        family = row["synthesis_family"]
        requirement = FAMILY_REQUIREMENTS.get(
            family,
            {
                "system": "generic DSL enumerator",
                "lowering": "static-shape primitive composition",
                "blocker": "family is not well classified yet",
            },
        )
        phase = phase_for_rank(rank, family, cost)
        row_out = {
            "priority_rank": rank,
            "task_id": task_id,
            "current_cost": cost,
            "current_points": row["current_points"],
            "synthesis_family": family,
            "route_prediction": row["route_prediction"],
            "current_source": row["current_source"],
            "current_template": row["current_template"],
            "target_phase": phase,
            "required_system": requirement["system"],
            "required_lowering": requirement["lowering"],
            "main_blocker": requirement["blocker"],
        }
        for phase_name, target_cost in TARGET_COSTS.items():
            row_out[f"gain_if_cost_le_{target_cost}"] = f"{gain_to_cost(cost, target_cost):.6f}"
        rows.append(row_out)

    rows.sort(key=lambda item: (str(item["target_phase"]), int(item["priority_rank"])))
    write_csv(EXP_DIR / "synthesis_backlog.csv", rows)

    cost_sorted = sorted(rows, key=lambda item: -int(item["current_cost"]))
    family_counts = Counter(str(item["synthesis_family"]) for item in rows)
    phase_counts = Counter(str(item["target_phase"]) for item in rows)
    system_counts = Counter(str(item["required_system"]) for item in rows)
    projected = {}
    for phase_name, target_cost in TARGET_COSTS.items():
        gain_key = f"gain_if_cost_le_{target_cost}"
        top_n = 100 if phase_name == "phase1_6500_bridge" else 200 if phase_name == "phase2_7000_submit_safe" else 320 if phase_name == "phase3_7400_private_like" else 400
        projected[phase_name] = {
            "target_cost": target_cost,
            "scope_top_n": top_n,
            "projected_local_if_scope_hit": local_estimate + sum(float(item[gain_key]) for item in cost_sorted[:top_n]),
            "projected_gain": sum(float(item[gain_key]) for item in cost_sorted[:top_n]),
        }

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "pipeline_backlog_ready",
        "base_exp": "experiments/exp016_top100_rewrite_campaign",
        "local_estimate": local_estimate,
        "gap_to_6500": 6500.0 - local_estimate,
        "gap_to_7000": 7000.0 - local_estimate,
        "gap_to_7400": 7400.0 - local_estimate,
        "gap_to_7600": 7600.0 - local_estimate,
        "exp021_probe": {
            "status": probe_result.get("status"),
            "delta": probe_result.get("delta"),
            "task_ids": probe_result.get("task_ids"),
            "lesson": "rule correctness is insufficient; low-cost lowering and cost-aware candidate generation are mandatory.",
        },
        "family_counts": dict(sorted(family_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "required_system_counts": dict(system_counts.most_common()),
        "projected_targets": projected,
        "top_required_items": cost_sorted[:30],
        "acceptance": {
            "phase1_6500_bridge": "local estimate >= 6500, zip sanity pass, no submit without full validation/risk audit.",
            "phase2_7000_submit_safe": "replace high-risk signature lookup where possible; full arc-gen 400/400 before submit.",
            "phase3_7400_private_like": "private-like holdout by primitive family and reduced public-artifact dependence.",
            "phase4_7600_contender": "multi-program enumerator with per-task proof logs, cost-optimal lowering, and submission cadence by threshold.",
        },
        "leakage_risk": "high while exp016 signature lookup remains the base.",
        "overfitting_risk": "high until synthesized programs are validated by full arc-gen and family holdout.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = [
        f"# {EXP_ID}",
        "",
        "## Hypothesis",
        "",
        "7600を狙うには、個別taskの手作業template追加では足りない。必要なのは、task family分類、DSL候補生成、ONNX lowering、static/rule検証、cost最小化、採用監査を同じ形で回すprogram synthesis pipelineである。",
        "",
        "## 現状",
        "",
        f"- base local estimate: `{local_estimate:.6f}`",
        f"- gap to 6500: `{6500.0 - local_estimate:.6f}`",
        f"- gap to 7600: `{7600.0 - local_estimate:.6f}`",
        "- exp021のtask398 probeでは、ruleはvalidation可能だったが `ScatterND` loweringのcostがbaselineより重く、採用されなかった。",
        "",
        "## 必要なシステム",
        "",
    ]
    for system, count in system_counts.most_common():
        notes.append(f"- `{system}`: {count} tasks")
    notes.extend(
        [
            "",
            "## 段階別Acceptance",
            "",
            "- Phase 1 / 6500: top cost taskを中心に低cost loweringを作り、local estimate >= 6500。submitは別途full validation後。",
            "- Phase 2 / 7000: high-risk lookupをDSL programへ置換し、提出安全性を上げる。",
            "- Phase 3 / 7400: primitive family別holdoutでprivate耐性を測る。",
            "- Phase 4 / 7600: enumerator、cost model、lowering optimizer、proof logを統合し、改善閾値ごとのsubmit cadenceに乗せる。",
            "",
            "## 次の実装優先度",
            "",
            "1. `static_lookup_compression`: signature lookup候補から実ルールを抽出し、memorization依存を下げる。",
            "2. `object_anchor_crop`: `NonZero` なしでbbox/cropを表す低cost loweringを作る。",
            "3. `sparse_coordinate_program`: full-grid `Where` を避け、少数座標の `GatherND/ScatterND` に落とす。",
            "4. `region_partition_fill`: naive flood unrollではなく、line-grid/room前提の閉領域判定に寄せる。",
            "5. `cost_model`: ONNX生成前にmemory+paramsを予測し、高cost loweringを生成前に棄却する。",
            "",
            "## Risks",
            "",
            "- leakage risk: high。現baseはexp016のsignature lookupを含む。",
            "- overfitting risk: high。local/sampleだけでなくfull arc-genとfamily holdoutが必要。",
        ]
    )
    (EXP_DIR / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if not REGISTRY_PATH.exists():
        sys.exit(f"missing registry: {REGISTRY_PATH}")
    main()
