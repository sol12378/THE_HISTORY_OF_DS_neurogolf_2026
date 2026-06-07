from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter
from datetime import date
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurogolf.synthesis.orchestrator import (  # noqa: E402
    PHASE_TARGETS,
    build_queue,
    dataclass_rows,
    guardrail_rows,
    load_csv,
    program_schema,
    summarize_queue,
    write_csv,
)


EXP_ID = "exp026_synthesis_orchestrator_core"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
BACKLOG_PATH = ROOT / "experiments" / "exp022_program_synthesis_pipeline_7600" / "synthesis_backlog.csv"
BASE_RESULT_PATH = BASE_EXP / "result.json"
SELECTED_PATH = BASE_EXP / "selected_manifest.csv"


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_worker_task_queue(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    worker_rows: list[dict[str, Any]] = []
    for row in rows[:80]:
        worker_rows.append(
            {
                "queue_rank": row["queue_rank"],
                "task_id": row["task_id"],
                "target_phase": row["target_phase"],
                "family": row["synthesis_family"],
                "worker_role": "low_reasoning_candidate_worker",
                "task": (
                    f"task{int(row['task_id']):03d}について、{row['required_system']}の候補を1 familyだけ生成する。"
                    "ONNX生成前にpreferred/reject loweringを確認し、proof logを残す。"
                ),
                "acceptance": row["worker_acceptance"],
                "review_points": "validation, static shape, banned ops, cost, leakage risk, overfitting risk",
            }
        )
    write_csv(path, worker_rows)


def notes_text(result: dict[str, Any]) -> str:
    top = result["top_queue_preview"][:12]
    lines = [
        f"# {EXP_ID}",
        "",
        "## Hypothesis",
        "",
        "7600を狙うには、公開artifact blendの足し算ではなく、taskごとに説明可能なDSL programを合成し、"
        "公式costに近い制約でONNXへ落とすオーケストレータが必要である。",
        "",
        "## Result",
        "",
        f"- base: `{result['base_exp']}`",
        f"- base local estimate: `{result['base_local_estimate']:.6f}`",
        f"- gap to 6500: `{result['gap_to_6500']:.6f}`",
        f"- gap to 7600: `{result['gap_to_7600']:.6f}`",
        f"- queue items: `{result['queue_summary']['task_count']}`",
        "",
        "## Phase Projection",
        "",
    ]
    for phase, info in result["queue_summary"]["projected"].items():
        lines.append(
            f"- `{phase}`: top{info['scope_top_n']}をcost<={info['target_cost']}にできれば "
            f"`+{info['projected_gain']:.3f}`"
        )
    lines.extend(
        [
            "",
            "## Top Queue",
            "",
            "| rank | task | family | current cost | target | projected gain | first action |",
            "|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in top:
        lines.append(
            f"| {row['queue_rank']} | {row['task_id']} | `{row['synthesis_family']}` | "
            f"{row['current_cost']} | {row['target_cost']} | {row['projected_gain_to_target']:.3f} | "
            f"{row['first_action'].split(';')[0]} |"
        )
    lines.extend(
        [
            "",
            "## Worker PDCA",
            "",
            "- main agentはorchestratorとして、`worker_task_queue.csv` の上位から低reasoning workerに狭い候補生成を渡す。",
            "- worker成果物は必ずmainがreviewし、基準未満なら同じtask/familyで再帰的に修正させる。",
            "- 採用条件は `validation pass`, `candidate cost < baseline`, `static/banned-op gate pass`, `proof logあり`。",
            "",
            "## Risks",
            "",
            "- leakage risk: high。現best baseはexp023で、exp016由来のsignature lookup local upper boundを含む。",
            "- overfitting risk: high。Phase 3以降はfamily holdoutとfull arc-gen validationが必要。",
            "",
            "## Next PDCA",
            "",
            "1. `exp027_cost_aware_lowering_bench` で過去失敗loweringをguardrail化する。",
            "2. `exp028_lookup_to_rule_miner` でsignature lookup 196件をrule候補へ圧縮する。",
            "3. `exp029_crop_object_synthesizer` でcrop/resize 85件をSlice/Gather中心に置換する。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base_result = load_json(BASE_RESULT_PATH)
    base_local = float(base_result.get("new_local_estimate") or base_result.get("local_estimate"))
    backlog_rows = load_csv(BACKLOG_PATH)
    selected_rows = load_csv(SELECTED_PATH)

    queue = build_queue(backlog_rows, selected_rows)
    queue_rows = dataclass_rows(queue)
    write_csv(EXP_DIR / "synthesis_queue.csv", queue_rows)
    write_worker_task_queue(EXP_DIR / "worker_task_queue.csv", queue_rows)
    write_csv(EXP_DIR / "lowering_guardrails.csv", guardrail_rows())
    write_json(EXP_DIR / "program_schema.json", program_schema())

    summary = summarize_queue(queue)
    family_counts = Counter(row["synthesis_family"] for row in queue_rows)
    high_risk_lookup = sum(1 for row in queue_rows if row["synthesis_family"] == "signature_lookup_current")
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "orchestrator_ready",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_local_estimate": base_local,
        "gap_to_6500": 6500.0 - base_local,
        "gap_to_7000": 7000.0 - base_local,
        "gap_to_7400": 7400.0 - base_local,
        "gap_to_7600": 7600.0 - base_local,
        "phase_targets": {phase: {"scope_top_n": scope, "target_cost": target} for phase, (scope, target) in PHASE_TARGETS.items()},
        "queue_summary": summary,
        "top_queue_preview": queue_rows[:25],
        "family_counts": dict(sorted(family_counts.items())),
        "high_risk_signature_lookup_count": high_risk_lookup,
        "outputs": {
            "synthesis_queue": "synthesis_queue.csv",
            "worker_task_queue": "worker_task_queue.csv",
            "lowering_guardrails": "lowering_guardrails.csv",
            "program_schema": "program_schema.json",
        },
        "worker_pdca": {
            "orchestrator": "main agent",
            "worker_reasoning": "low",
            "review_required": True,
            "recursive_fix_if_below_threshold": True,
            "threshold": "validation pass and candidate cost lower than current baseline",
        },
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: this is pipeline infrastructure and does not improve local estimate.",
        "leakage_risk": "high while exp023 signature lookup candidates remain selected.",
        "overfitting_risk": "high until synthesized programs pass full arc-gen and private-like family holdout.",
    }
    write_json(EXP_DIR / "result.json", result)
    (EXP_DIR / "notes.md").write_text(notes_text(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
