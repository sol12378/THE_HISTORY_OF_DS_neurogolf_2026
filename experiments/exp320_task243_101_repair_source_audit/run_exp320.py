from __future__ import annotations

import csv
import json
from pathlib import Path


EXP_ID = "exp320_task243_101_repair_source_audit"
OUT_DIR = Path("experiments") / EXP_ID
MANIFEST = Path("experiments/exp002_public_blend_6500_fast/candidate_manifest.csv")
BASE_PUBLIC_LB = 6008.96
POINTS_IF_FIXED = {
    "101": 13.904485714156337,
    "243": 13.874532744845478,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("task_id") in POINTS_IF_FIXED:
                rows.append(row)

    accepted = [row for row in rows if row.get("status") == "accepted"]
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-11",
        "status": "completed",
        "purpose": (
            "Audit existing full-local-valid candidate sources for task243/task101 "
            "after exp319 showed both are public-zero in the exp297 lineage."
        ),
        "base_exp": "experiments/exp297_exp262_skip_task048_336_fresh_candidates",
        "base_public_lb": BASE_PUBLIC_LB,
        "exp319_observed_public_lb": 6008.96,
        "exp319_expected_lb_if_both_alive": 5981.180981540998,
        "interpretation": (
            "Fail-stubbing task243/task101 caused no public LB drop, so both tasks "
            "are treated as public-zero repair targets."
        ),
        "targets": [243, 101],
        "source_manifest": str(MANIFEST),
        "candidate_counts": {},
        "top_candidates": {},
        "recommended_repair": {},
        "expected_public_gain_if_both_fixed": sum(POINTS_IF_FIXED.values()),
        "expected_public_lb_if_both_fixed": BASE_PUBLIC_LB + sum(POINTS_IF_FIXED.values()),
        "submission_decision": (
            "no_submit_audit_only; next experiment should build a repair zip with "
            "the top full-local-valid candidates for task101/task243 and submit as "
            "a repair bundle or one task at a time."
        ),
        "leakage_risk": (
            "medium-high: strongest candidates are public-code/raw blend sources; "
            "they are full-local-valid but may encode public-test behavior and need "
            "private robustness caution."
        ),
        "overfitting_risk": (
            "medium: exp319 is a public diagnostic; repair candidates must be accepted "
            "only through full local validation and expected-LB accounting, not score chasing."
        ),
    }

    keep_keys = [
        "task_id",
        "filename",
        "source_label",
        "source_ref",
        "relative_path",
        "sha256",
        "file_bytes",
        "normalized_bytes",
        "params",
        "memory_bytes",
        "cost",
        "simple_cost",
        "local_points",
        "validation_status",
        "validation_pass",
        "validation_fail",
    ]
    for task_id in ["101", "243"]:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        task_accepted = [row for row in accepted if row["task_id"] == task_id]
        task_accepted.sort(key=lambda row: float(row.get("local_points") or 0.0), reverse=True)
        kept = [{key: row.get(key) for key in keep_keys} for row in task_accepted[:10]]
        result["candidate_counts"][task_id] = {
            "total_manifest_rows": len(task_rows),
            "accepted_rows": len(task_accepted),
            "rejected_rows": len(task_rows) - len(task_accepted),
        }
        result["top_candidates"][task_id] = kept
        result["recommended_repair"][task_id] = kept[0] if kept else None

    (OUT_DIR / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# {EXP_ID}",
        "",
        "## 目的",
        "",
        (
            "exp319 で task243/task101 を fail-stub しても Public LB が "
            "`6008.96` から落ちなかったため、この2 task を public-zero repair "
            "target として既存 accepted candidate source を監査する。"
        ),
        "",
        "## 結果",
        "",
        "- status: `completed`",
        "- base_public_lb: `6008.96`",
        "- exp319 public LB: `6008.96`",
        "- interpretation: task243/task101 は current-best lineage 上で public-zero と扱う",
        f"- expected_public_gain_if_both_fixed: `{result['expected_public_gain_if_both_fixed']}`",
        f"- expected_public_lb_if_both_fixed: `{result['expected_public_lb_if_both_fixed']}`",
        "",
        "## Top Candidates",
    ]
    for task_id in ["101", "243"]:
        lines.extend(
            [
                "",
                f"### task{task_id}",
                f"- accepted rows: `{result['candidate_counts'][task_id]['accepted_rows']}`",
            ]
        )
        for candidate in result["top_candidates"][task_id][:5]:
            lines.append(
                "- "
                f"`{candidate['source_label']}` / "
                f"`{candidate['source_ref']}` / "
                f"local_points `{candidate['local_points']}` / "
                f"cost `{candidate['cost']}` / "
                f"sha `{candidate['sha256']}`"
            )

    lines.extend(
        [
            "",
            "## 判断",
            "",
            (
                "提出はしない。次実験で recommended_repair の top candidate を exp297 に差し替え、"
                "full validation と zip sanity の後、単体または2 task bundle repair として提出する。"
            ),
            "",
            "## リスク",
            "",
            (
                "- leakage risk: medium-high。public-code/raw blend source 由来であり、"
                "private robustness は低めに見る。"
            ),
            (
                "- overfitting risk: medium。public diagnostic で特定した repair target なので、"
                "採用判断は full local validation と expected LB 一致に限定する。"
            ),
        ]
    )
    (OUT_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["recommended_repair"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
