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
EXP_ID = "exp_b001_rule_replacement_backlog"
EXP_DIR = ROOT / "experiments" / EXP_ID

STRICT_RESULT = ROOT / "experiments" / "exp005_top_cost_rewrite_strict" / "result.json"
TEACHER_RESULT = ROOT / "experiments" / "exp041_task145_deeper_mul_chain" / "result.json"
QUEUE_PATH = ROOT / "experiments" / "exp026_synthesis_orchestrator_core" / "synthesis_queue.csv"
EXP045_RESULT = ROOT / "experiments" / "exp045_rule_searcher_best_practices_7700" / "result.json"
EXP048_RESULT = ROOT / "experiments" / "exp048_submit_safe_seed_inventory" / "result.json"


FAMILY_GUIDANCE: dict[str, dict[str, str]] = {
    "signature_lookup_current": {
        "priority": "P0",
        "rule_target": "説明可能な decision tree / affine hash / color-role rule",
        "lowering_target": "small Gather or pruned branch graph; no per-example label table",
        "reject": "arc-gen memorization, large ScatterND, large MatMul hash",
        "best_practice_task": "teacher artifactはoracleとしてprofileし、入出力差分から人間可読なinvariantだけを採用する。",
    },
    "crop_resize": {
        "priority": "P1",
        "rule_target": "fixed crop / object-anchor crop / shape rule / bbox crop",
        "lowering_target": "constant Slice/Gather, small Pad/Concat, static row-column indices",
        "reject": "NonZero, Compress, full-grid GatherND bbox, full-grid Tile",
        "best_practice_task": "bboxを見つけてもすぐONNX化せず、index table sizeとactivation memoryを先に見積もる。",
    },
    "sparse_edit_or_object_completion": {
        "priority": "P1",
        "rule_target": "changed-cell rule / component completion / object copy with sparse mask",
        "lowering_target": "tiny coordinate ScatterND, small Conv kernels, Equal plus one Where",
        "reject": "large coordinate initializer, repeated full-grid Where, unrolled local propagation",
        "best_practice_task": "変更セル数が少ないことを証明してから、全grid処理を避けるloweringだけをemitする。",
    },
    "point_to_line_pattern": {
        "priority": "P1",
        "rule_target": "seed-to-line / ray / periodic pattern with color roles",
        "lowering_target": "constant affine masks, row-column masks, small Conv line detectors",
        "reject": "per-cell ScatterND table, long Where chains over all periods",
        "best_practice_task": "seed、方向、period、stop conditionを分けて探索し、絶対色ではなくroleで記述する。",
    },
    "line_grid_fill": {
        "priority": "P1",
        "rule_target": "row-column completion / line extension / grid crossing rule",
        "lowering_target": "ReduceSum row-column masks, small Conv kernels, constant masks plus Where",
        "reject": "full-grid Tile, long Where chain, per-cell ScatterND",
        "best_practice_task": "線分単位の閉形式maskを作り、unrollしない。",
    },
    "region_partition_fill": {
        "priority": "P1",
        "rule_target": "room/inside-outside/boundary partition rule",
        "lowering_target": "static row-column prefix masks, small Conv kernels, bounded room mask",
        "reject": "many-step flood fill, dilation unroll, dynamic full-grid colormap",
        "best_practice_task": "flood fillの正しさを利用しても、実装は閉形式の部屋maskへ落とす。",
    },
    "same_shape_global_transform": {
        "priority": "P2",
        "rule_target": "flip/rotate/transpose/recolor or graph-surgery proof",
        "lowering_target": "Transpose, Gather, initializer prune, small color Conv",
        "reject": "compact artifactより大きい再実装",
        "best_practice_task": "既存artifactが小さい場合はrule ONNXよりsurgeryの方を優先する。",
    },
    "color_map": {
        "priority": "P2",
        "rule_target": "consistent color role map",
        "lowering_target": "small Equal/Where chain or 1x1 Conv",
        "reject": "large dynamic MatMul colormap",
        "best_practice_task": "単独改善より他familyの前処理として使う。",
    },
}


@dataclass
class BacklogRow:
    rank: int
    task_id: int
    family: str
    priority: str
    current_cost: int
    current_points: float
    target_cost: int
    gain_to_target: float
    rule_target: str
    lowering_target: str
    reject_lowering: str
    first_action: str
    acceptance: str
    submission_policy: str


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


def choose_target_cost(row: dict[str, str]) -> int:
    phase = row["target_phase"]
    if phase == "phase1_6500_bridge":
        return 9000
    if phase == "phase2_7000_submit_safe":
        return 3000
    if phase == "phase3_7400_private_like":
        return 1000
    return 250


def build_backlog(queue: list[dict[str, str]]) -> list[BacklogRow]:
    rows: list[BacklogRow] = []
    for i, row in enumerate(queue, start=1):
        family = row["synthesis_family"]
        guidance = FAMILY_GUIDANCE[family]
        current_cost = int(float(row["current_cost"]))
        current_points = float(row["current_points"])
        target_cost = choose_target_cost(row)
        rows.append(
            BacklogRow(
                rank=i,
                task_id=int(row["task_id"]),
                family=family,
                priority=guidance["priority"],
                current_cost=current_cost,
                current_points=current_points,
                target_cost=target_cost,
                gain_to_target=max(0.0, point(target_cost) - current_points),
                rule_target=guidance["rule_target"],
                lowering_target=guidance["lowering_target"],
                reject_lowering=guidance["reject"],
                first_action=guidance["best_practice_task"],
                acceptance=(
                    "human-readable rule; train/test/all-arc-gen exact; official utility pass; "
                    "candidate cost < current strict seed or accepted submit-safe artifact; leakage/overfit note"
                ),
                submission_policy=(
                    "submit only small calibrated bundles first; compare LB/local before adding high-risk families"
                    if i <= 40
                    else "hold until family calibration is trusted"
                ),
            )
        )
    return sorted(rows, key=lambda r: (r.priority, r.rank))


def family_summary(backlog: list[BacklogRow]) -> list[dict[str, Any]]:
    by_family: dict[str, list[BacklogRow]] = defaultdict(list)
    for row in backlog:
        by_family[row.family].append(row)
    out: list[dict[str, Any]] = []
    for family, rows in sorted(by_family.items(), key=lambda kv: -sum(r.gain_to_target for r in kv[1])):
        guidance = FAMILY_GUIDANCE[family]
        out.append(
            {
                "family": family,
                "task_count": len(rows),
                "priority": guidance["priority"],
                "projected_gain_to_phase_targets": sum(r.gain_to_target for r in rows),
                "rule_target": guidance["rule_target"],
                "lowering_target": guidance["lowering_target"],
                "reject_lowering": guidance["reject"],
                "best_practice": guidance["best_practice_task"],
            }
        )
    return out


def calibration_plan() -> list[dict[str, Any]]:
    return [
        {
            "stage": "calib_001_strict_seed_resubmit",
            "source": "exp005_top_cost_rewrite_strict",
            "purpose": "local/LBの基準線を作る。exp041崩壊後、strict seedがKaggleでどこまで通用するか確認する。",
            "submit": "yes_when_user_confirms_or_next_submission_slot_available",
            "expected_lb": "near strict public-safe score if local utility matches Kaggle; otherwise environment gap investigation",
            "risk": "low",
        },
        {
            "stage": "calib_002_single_family_delta",
            "source": "first accepted exp_b rule-lowering candidate",
            "purpose": "1 task or 1 small familyだけ差し替え、local deltaとLB deltaの対応を見る。",
            "submit": "yes_after_official_validation",
            "expected_lb": "delta should be small but directionally consistent",
            "risk": "low-medium",
        },
        {
            "stage": "calib_003_p0_lookup_replacement_bundle",
            "source": "P0 lookup-compressed rules only",
            "purpose": "signature lookupを説明可能ruleに置換したbundleがLB collapseを起こさないか確認する。",
            "submit": "yes_after_family_holdout",
            "expected_lb": "local/LB gap materially smaller than exp041",
            "risk": "medium",
        },
        {
            "stage": "calib_004_mixed_family_growth",
            "source": "trusted calibrated families",
            "purpose": "crop/sparse/line/regionを段階追加し、7700に向けてLB対応を維持する。",
            "submit": "cadenced_threshold_submissions",
            "expected_lb": "monotonic or explainable movement",
            "risk": "medium",
        },
    ]


def notes_text(result: dict[str, Any]) -> str:
    lines = [
        f"# {EXP_ID}",
        "",
        "## 目的",
        "",
        "exp_b系列の開始点として、全400 taskを説明可能rule / decision tree / color-role ruleへ置換するbacklogを作る。",
        "有用でない、または既存artifactがすでに小さいtaskは、無理な置換ではなくfamily別best practice探索へ回す。",
        "",
        "## 仮説",
        "",
        "7700への近道はlocal score huntingではなく、submit-safe seedから始めて、説明可能ruleとcost-aware ONNX loweringをfamily単位で較正すること。",
        "",
        "## 結果",
        "",
        f"- strict seed: `{result['strict_seed']['exp']}` / local `{result['strict_seed']['local_estimate']}`",
        f"- teacher/local upper: `{result['teacher']['exp']}` / local `{result['teacher']['local_estimate']}` / LB `{result['teacher']['lb']}`",
        f"- backlog tasks: `{result['backlog_task_count']}`",
        f"- family count: `{len(result['family_summary'])}`",
        "",
        "## Family Summary",
        "",
        "| family | tasks | projected gain | priority | best practice |",
        "|---|---:|---:|---|---|",
    ]
    for row in result["family_summary"]:
        lines.append(
            f"| `{row['family']}` | {row['task_count']} | {row['projected_gain_to_phase_targets']:.3f} | "
            f"{row['priority']} | {row['best_practice']} |"
        )
    lines.extend(
        [
            "",
            "## Submission Calibration",
            "",
            "local/LB差を早く縮めるため、今後は大きなbundleを一気に出さず、strict seed再提出、single-family delta、P0 lookup replacement bundleの順に提出して較正する。",
            "",
            "## リスク",
            "",
            "- leakage risk: 中。teacher artifactはoracle扱いで、rule化されるまでsubmit-safeとは見なさない。",
            "- overfitting risk: 中。all arc-gen exactだけでなく、family holdoutとKaggle calibration submissionで確認する。",
            "",
            "## 次アクション",
            "",
            "1. `calib_001_strict_seed_resubmit` を実行し、exp005 strict seedのLB基準を取る。",
            "2. P0上位taskからrule minerを開始する。まず `task020` をteacher profileから説明可能ruleへ圧縮する。",
            "3. 受理済みruleだけを安いONNX loweringへ落とし、single-task差し替えsubmissionでlocal/LB対応を見る。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    queue = read_csv(QUEUE_PATH)
    strict = load_json(STRICT_RESULT)
    teacher = load_json(TEACHER_RESULT)
    exp045 = load_json(EXP045_RESULT)
    exp048 = load_json(EXP048_RESULT) if EXP048_RESULT.exists() else {}

    backlog = build_backlog(queue)
    families = family_summary(backlog)
    submit_plan = calibration_plan()

    strict_local = float(strict.get("local_estimate", strict.get("new_local_estimate", 0.0)))
    teacher_local = float(teacher["local_estimate"])
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "backlog_ready",
        "purpose": "start exp_b series for explainable rule replacement and Kaggle calibration",
        "strict_seed": {
            "exp": "exp005_top_cost_rewrite_strict",
            "local_estimate": strict_local,
            "role": "submit-safe seed / calibration baseline",
        },
        "teacher": {
            "exp": "exp041_task145_deeper_mul_chain",
            "local_estimate": teacher_local,
            "lb": 3417.71,
            "role": "teacher oracle only; not submit-safe",
        },
        "exp045_projection": exp045["projection_7700"],
        "exp048_context": {
            "status": exp048.get("status", ""),
            "strict_seed": exp048.get("strict_seed", {}),
            "p0_task_count": exp048.get("p0_teacher_gain_count", exp048.get("p0_count", "")),
        },
        "backlog_task_count": len(backlog),
        "family_summary": families,
        "submission_calibration_plan": submit_plan,
        "outputs": {
            "rule_replacement_backlog": "rule_replacement_backlog.csv",
            "family_best_practices": "family_best_practices.csv",
            "submission_calibration_plan": "submission_calibration_plan.csv",
            "notes": "notes.md",
        },
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: backlog/protocol experiment; next action is strict seed calibration submission",
        "leakage_risk": "medium: teacher artifacts are used only as diagnostic oracles; backlog requires human-readable rules.",
        "overfitting_risk": "medium: accepted rules require all-arc-gen plus Kaggle calibration, not local-only acceptance.",
    }

    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(EXP_DIR / "rule_replacement_backlog.csv", [asdict(row) for row in backlog])
    write_csv(EXP_DIR / "family_best_practices.csv", families)
    write_csv(EXP_DIR / "submission_calibration_plan.csv", submit_plan)
    (EXP_DIR / "notes.md").write_text(notes_text(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
