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
EXP_ID = "exp045_rule_searcher_best_practices_7700"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_RESULT = ROOT / "experiments" / "exp041_task145_deeper_mul_chain" / "result.json"
QUEUE_PATH = ROOT / "experiments" / "exp026_synthesis_orchestrator_core" / "synthesis_queue.csv"
HITS_PATH = ROOT / "experiments" / "exp042_freeop_dag_search_top200" / "program_hits.csv"


PHASES = [
    ("phase1_6500_bridge", 100, 9000),
    ("phase2_7000_submit_safe", 200, 3000),
    ("phase3_7400_private_like", 320, 1000),
    ("phase4_7700_proof", 400, 250),
]


BEST_PRACTICES: dict[str, dict[str, str]] = {
    "signature_lookup_current": {
        "searcher": "lookup compression miner",
        "best_practice": "signatureを説明変数へ分解し、浅いdecision tree/affine hash/color-role ruleへ置換する。arc-gen labelの丸暗記は禁止。",
        "preferred_lowering": "small Gather, initializer-pruned decision tree, constant Slice/Gather when crop-like",
        "reject_lowering": "large ScatterND table, full arc-gen memorization, large MatMul hash",
    },
    "crop_resize": {
        "searcher": "object-anchor crop/resize synthesizer",
        "best_practice": "fixed crop, anchor crop, bbox cropを分けて探索し、ONNX前にshape ruleとindex sizeをcost gateへ通す。",
        "preferred_lowering": "constant Slice/Gather, small Concat/Pad, static row/column indices",
        "reject_lowering": "NonZero/Compress, full-grid GatherND bbox, full-grid Tile plus mask",
    },
    "sparse_edit_or_object_completion": {
        "searcher": "sparse edit/object completion enumerator",
        "best_practice": "changed-cell maskを先に説明し、少数座標だけをemitする。広いmaskはConv/Reduceで閉形式化する。",
        "preferred_lowering": "small Conv kernels, tiny GatherND/ScatterND, Equal plus single Where",
        "reject_lowering": "large coordinate initializer, repeated full-grid Where, unrolled fill without cost forecast",
    },
    "point_to_line_pattern": {
        "searcher": "seed-to-pattern grammar",
        "best_practice": "seed color, direction, period, stop conditionをroleで推定し、絶対色より構造不変量を優先する。",
        "preferred_lowering": "constant affine masks, row/column masks, small Conv line detectors",
        "reject_lowering": "per-cell ScatterND table, long Where chains over all periods",
    },
    "region_partition_fill": {
        "searcher": "region/room-fill synthesizer",
        "best_practice": "flood fillを直接unrollせず、境界線・部屋・内外判定をrow/column prefixや小kernelで閉形式化する。",
        "preferred_lowering": "static row/column ReduceSum masks, small Conv kernels, bounded single-pass room masks",
        "reject_lowering": "radius-scale dilation, many-step flood-fill unroll, dynamic full-grid color map",
    },
    "line_grid_fill": {
        "searcher": "line/ray/grid grammar",
        "best_practice": "complete/incomplete row-column、交点、stop blockerを列挙し、線分単位で正しさを証明する。",
        "preferred_lowering": "ReduceSum row/column masks, small Conv kernels, constant masks plus Where",
        "reject_lowering": "per-cell ScatterND table, full-grid Tile, long Where chain",
    },
    "color_map": {
        "searcher": "color role normalizer",
        "best_practice": "単独color mapだけでなく他familyの前処理として扱い、consistent role mapのみ採用する。",
        "preferred_lowering": "1x1 Conv, small Equal/Where chain, color-role constants",
        "reject_lowering": "large MatMul color map, dynamic full-grid color map",
    },
    "same_shape_global_transform": {
        "searcher": "global transform and compact surgery",
        "best_practice": "flip/rotate/transpose/recolorを最小ONNXと既存artifact surgeryの両方で比較する。",
        "preferred_lowering": "Transpose, Gather, initializer prune, small 1x1 Conv",
        "reject_lowering": "rewriting an already compact artifact with a larger full-grid graph",
    },
}


@dataclass
class FamilyPlan:
    family: str
    task_count: int
    phase1_count: int
    phase4_count: int
    gain_to_250: float
    current_cost_median: float
    target_cost: int
    searcher: str
    best_practice: str
    preferred_lowering: str
    reject_lowering: str
    acceptance: str


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def point(cost: float) -> float:
    if cost <= 0:
        return 25.0
    return max(1.0, 25.0 - math.log(cost))


def median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    mid = len(ys) // 2
    if len(ys) % 2:
        return ys[mid]
    return (ys[mid - 1] + ys[mid]) / 2.0


def phase_projection(queue: list[dict[str, str]], base_local: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, scope, target in PHASES:
        selected = queue[:scope]
        gain = 0.0
        for row in selected:
            current_points = float(row["current_points"])
            gain += max(0.0, point(target) - current_points)
        rows.append(
            {
                "phase": phase,
                "scope_top_n": scope,
                "target_cost": target,
                "projected_gain": gain,
                "projected_local": base_local + gain,
                "gap_to_7700": 7700.0 - (base_local + gain),
            }
        )
    return rows


def family_plans(queue: list[dict[str, str]]) -> list[FamilyPlan]:
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in queue:
        by_family[row["synthesis_family"]].append(row)

    plans: list[FamilyPlan] = []
    for family, rows in sorted(by_family.items()):
        practice = BEST_PRACTICES.get(
            family,
            {
                "searcher": "generic DSL enumerator",
                "best_practice": "family固有のinvariantを先に記述し、ONNX前にcost gateへ通す。",
                "preferred_lowering": "small static tensors and bounded masks",
                "reject_lowering": "large dynamic full-grid tensors",
            },
        )
        costs = [float(row["current_cost"]) for row in rows]
        gain_to_250 = sum(max(0.0, point(250) - float(row["current_points"])) for row in rows)
        plans.append(
            FamilyPlan(
                family=family,
                task_count=len(rows),
                phase1_count=sum(1 for row in rows if row["target_phase"] == "phase1_6500_bridge"),
                phase4_count=sum(1 for row in rows if row["target_phase"].startswith("phase4")),
                gain_to_250=gain_to_250,
                current_cost_median=median(costs),
                target_cost=250,
                searcher=practice["searcher"],
                best_practice=practice["best_practice"],
                preferred_lowering=practice["preferred_lowering"],
                reject_lowering=practice["reject_lowering"],
                acceptance=(
                    "full train/test/all-arc-gen exact, official utility pass, candidate cost lower than current, "
                    "leakage/overfitting note, and family holdout before submit"
                ),
            )
        )
    return sorted(plans, key=lambda p: (-p.gain_to_250, -p.task_count, p.family))


def notes_text(result: dict[str, Any], phase_rows: list[dict[str, Any]], plans: list[FamilyPlan]) -> str:
    lines = [
        f"# {EXP_ID}",
        "",
        "## 目的",
        "",
        "この実験はスコアを直接探すものではなく、7700到達がルール探索器の整備で説明可能かを検証するための設計証明である。",
        "既存のbest localを固定し、400 taskをfamily別に分解して、必要なtarget cost、best practice、禁止lowering、受理基準を明文化する。",
        "",
        "## 仮説",
        "",
        "全400 taskのうち、artifact/lookup依存をfamily別DSL programへ置換し、平均的にcost<=250相当まで落とせれば7700を超える。",
        "過去実験の失敗から、正しいルールを見つけるだけでは不十分であり、ONNX loweringのcost gateを探索器に組み込む必要がある。",
        "",
        "## Projection",
        "",
        "| phase | scope | target cost | projected local | gap to 7700 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in phase_rows:
        lines.append(
            f"| `{row['phase']}` | {row['scope_top_n']} | {row['target_cost']} | "
            f"{row['projected_local']:.3f} | {row['gap_to_7700']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Family Best Practices",
            "",
            "| family | tasks | gain_to_250 | searcher | best practice |",
            "|---|---:|---:|---|---|",
        ]
    )
    for plan in plans:
        lines.append(
            f"| `{plan.family}` | {plan.task_count} | {plan.gain_to_250:.3f} | "
            f"{plan.searcher} | {plan.best_practice} |"
        )
    lines.extend(
        [
            "",
            "## 証明としての意味",
            "",
            f"- base local: `{result['base_local_estimate']:.6f}`",
            f"- cost<=250 projection: `{result['projection_7700']['projected_local']:.6f}`",
            f"- 7700 margin: `{result['projection_7700']['margin_over_7700']:.6f}`",
            f"- exp042 rule-search proxy delta: `{result['prior_evidence']['exp042_total_proxy_delta']:.6f}`",
            f"- exp042 vs exp040 surgery multiple: `{result['prior_evidence']['exp042_vs_surgery_multiple']:.2f}x`",
            "",
            "これは「今すぐ7700のsubmissionがある」証明ではない。証明しているのは、7700に必要な利得が、局所surgeryではなくfamily別ルール探索器とcost-aware loweringの改善量として表現できる、という開発命題である。",
            "",
            "## Leakage / Overfitting Risk",
            "",
            "- leakage risk: 中〜高。baseにはsignature lookup由来のlocal upper boundが残るため、Phase2以降はlookup compressionで説明可能ruleへ置換する。",
            "- overfitting risk: 中。all arc-gen exactをproofに使うが、提出候補はfamily holdout、公式utility、private-like validationを通す。",
            "",
            "## 次アクション",
            "",
            "1. `signature_lookup_current` をまず圧縮対象にし、lookupを説明可能なtree/ruleへ置換する。",
            "2. `crop_resize` はbboxをfull-grid GatherNDにしないshape-specialized compilerを作る。",
            "3. `region_partition_fill` と `point_to_line_pattern` は正解ルールより先にcost gate付きlowering skeletonを用意する。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    queue = read_csv(QUEUE_PATH)
    base = load_json(BASE_RESULT)
    base_local = float(base["local_estimate"])
    phase_rows = phase_projection(queue, base_local)
    plans = family_plans(queue)
    hits = read_csv(HITS_PATH)

    projection_7700 = phase_rows[-1]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "proof_plan_ready",
        "base_exp": "experiments/exp041_task145_deeper_mul_chain",
        "base_local_estimate": base_local,
        "projection_7700": {
            "target_cost_all_400": projection_7700["target_cost"],
            "projected_gain": projection_7700["projected_gain"],
            "projected_local": projection_7700["projected_local"],
            "margin_over_7700": projection_7700["projected_local"] - 7700.0,
        },
        "phase_projection": phase_rows,
        "family_plan_count": len(plans),
        "top_family_plans": [asdict(plan) for plan in plans],
        "prior_evidence": {
            "exp042_hit_count": len(hits),
            "exp042_total_proxy_delta": sum(float(row["proxy_delta"]) for row in hits),
            "exp042_vs_surgery_multiple": 319.12100187849006,
            "exp043_lesson": "naive fixed DSL lowering can lose against golfed artifacts",
            "exp044_lesson": "dynamic bbox as full-grid GatherND is correct but too expensive",
        },
        "acceptance_criteria_for_future_searcher": [
            "program is a human-readable rule, not per-example label lookup",
            "full train/test/all-arc-gen exact match",
            "official static/banned-op/cost validation pass",
            "candidate cost lower than selected artifact",
            "family-specific leakage and overfitting note",
            "reject pattern checked before ONNX emission",
        ],
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: proof/planning experiment, not a new submission artifact",
        "leakage_risk": "medium-high: current base still includes lookup-derived artifacts; this experiment reduces risk by defining replacement rules.",
        "overfitting_risk": "medium: proof uses public arc-gen; future candidates need family holdout and official utility validation.",
    }

    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(EXP_DIR / "phase_projection.csv", phase_rows)
    write_csv(EXP_DIR / "family_best_practices.csv", [asdict(plan) for plan in plans])
    (EXP_DIR / "notes.md").write_text(notes_text(result, phase_rows, plans), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
