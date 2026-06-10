from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "exp192_phase_a_risk_probe_review"


PROBES = [
    ("exp176", [202, 382, 205, 383], "all_alive", 5952.64),
    ("exp179", [251, 109, 239, 358], "all_alive", 5952.01),
    ("exp181", [281, 203, 126, 159], "all_alive", 5951.48),
    ("exp182", [313, 370, 234, 303], "all_alive", 5951.46),
    ("exp183", [187, 204, 198, 364], "partial_zero_task187_suspect", 5965.28),
    ("exp184", [284, 300, 379, 340], "all_alive", 5951.06),
    ("exp188", [328, 301, 306, 387], "all_alive", 5950.27),
    ("exp190", [238, 112, 377, 177], "all_alive", 5950.05),
]

REPAIRS = [
    ("exp186", 187, "no_public_gain", 6005.90, "massimilianoghiotto full-local-valid raw did not repair public score"),
]


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    all_alive = [p for p in PROBES if p[2] == "all_alive"]
    partial = [p for p in PROBES if p[2].startswith("partial")]
    probed_tasks = sorted({tid for _, tasks, _, _ in PROBES for tid in tasks})
    excluded_tasks = sorted({tid for _, tasks, status, _ in PROBES if status == "all_alive" for tid in tasks})
    unresolved = [187]

    result = {
        "exp_id": "exp192_phase_a_risk_probe_review",
        "date": "2026-06-10",
        "status": "review_complete",
        "purpose": "Review recent Phase A risk-inventory fail-stub probes against docs/experiment_plan_2026-06-10.md review gate.",
        "probe_count": len(PROBES),
        "probed_task_count": len(probed_tasks),
        "all_alive_probe_count": len(all_alive),
        "partial_zero_probe_count": len(partial),
        "successful_repair_count": 0,
        "failed_repair_count": len(REPAIRS),
        "excluded_public_zero_suspects": excluded_tasks,
        "unresolved_public_zero_repair_queue": unresolved,
        "probes": [
            {"exp": exp, "targets": tasks, "interpretation": status, "public_lb": lb}
            for exp, tasks, status, lb in PROBES
        ],
        "repairs": [
            {"exp": exp, "task_id": task_id, "status": status, "public_lb": lb, "note": note}
            for exp, task_id, status, lb, note in REPAIRS
        ],
        "decision": (
            "Do not keep spending primary submission budget on lower-ranked Phase A risk probes. "
            "Keep task187 in a focused repair queue, but shift main PDCA to Phase C cost-band compression from the experiment plan."
        ),
        "next_actions": [
            "Start Phase C review with dtype/cost evidence from exp158-161.",
            "Pick one solved-rule task with realistic low-cost lowering surface and build a 20-minute single-task cost probe.",
            "Keep any official-valid low-risk delta submission-ready, but avoid broad public-source swaps without LB evidence.",
        ],
        "leakage_risk": "low: review only; no new model adopted.",
        "overfitting_risk": "low: review only, but conclusion depends on recent public probe evidence.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict) -> None:
    lines = [
        "# exp192_phase_a_risk_probe_review",
        "",
        "## 目的",
        "",
        "docs/experiment_plan_2026-06-10.md のreview gateに従い、直近Phase A risk-inventory probeの限界効率を評価する。",
        "",
        "## 結果",
        "",
        f"- probe_count: `{result['probe_count']}`",
        f"- probed_task_count: `{result['probed_task_count']}`",
        f"- all_alive_probe_count: `{result['all_alive_probe_count']}`",
        f"- partial_zero_probe_count: `{result['partial_zero_probe_count']}`",
        f"- successful_repair_count: `{result['successful_repair_count']}`",
        f"- failed_repair_count: `{result['failed_repair_count']}`",
        f"- unresolved_public_zero_repair_queue: `{result['unresolved_public_zero_repair_queue']}`",
        "",
        "## 判断",
        "",
        str(result["decision"]),
        "",
        "## 次アクション",
        "",
    ]
    lines.extend([f"- {item}" for item in result["next_actions"]])
    lines.extend(["", "## リスク", "", str(result["leakage_risk"]), str(result["overfitting_risk"])])
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
