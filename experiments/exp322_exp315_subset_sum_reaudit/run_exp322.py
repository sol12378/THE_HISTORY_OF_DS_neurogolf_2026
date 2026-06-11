from __future__ import annotations

import itertools
import json
from pathlib import Path


EXP_ID = "exp322_exp315_subset_sum_reaudit"
EXP_DIR = Path("experiments") / EXP_ID
EXP315 = Path("experiments/exp315_wide_failure_bisection_probe_a/result.json")
EXP318 = Path("experiments/exp318_split_probe_task193_275/result.json")
EXP319 = Path("experiments/exp319_split_probe_task243_101/result.json")
EXP321 = Path("experiments/exp321_task243_101_franksunp_repair/result.json")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def subset_candidates(points: dict[int, float], target: float, max_size: int = 4) -> list[dict]:
    out = []
    items = sorted(points.items())
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(items, size):
            tasks = [task_id for task_id, _ in combo]
            total = sum(value for _, value in combo)
            out.append(
                {
                    "tasks": tasks,
                    "points_sum": total,
                    "absolute_error": abs(total - target),
                    "size": size,
                }
            )
    out.sort(key=lambda row: (row["absolute_error"], row["size"], row["tasks"]))
    return out


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    exp315 = load_json(EXP315)
    exp318 = load_json(EXP318)
    exp319 = load_json(EXP319)
    exp321 = load_json(EXP321)

    points = {int(task_id): float(value) for task_id, value in exp315["target_points"].items()}
    missing = float(exp315["missing_drop_vs_all_alive"])
    confirmed_alive = set(exp318["targets"])
    nonrecoverable_pair = set(exp319["targets"])
    remaining_after_alive = {task_id: value for task_id, value in points.items() if task_id not in confirmed_alive}
    remaining_after_alive_and_pair = {
        task_id: value
        for task_id, value in points.items()
        if task_id not in confirmed_alive and task_id not in nonrecoverable_pair
    }

    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-11",
        "status": "completed",
        "purpose": (
            "Re-audit exp315 missing-drop subset-sum after exp318 proved task193/task275 alive "
            "and exp321 showed task101/task243 franksunp repair had no public gain."
        ),
        "base_public_lb": exp315["base_public_lb"],
        "exp315": {
            "targets": exp315["targets"],
            "expected_drop_if_all_alive": exp315["expected_drop_if_all_alive"],
            "public_lb": exp315["public_lb"],
            "observed_drop_from_base": exp315["observed_drop_from_base"],
            "missing_drop_vs_all_alive": missing,
        },
        "followup_evidence": {
            "exp318_task193_275": {
                "public_lb": exp318["public_lb"],
                "interpretation": "all_alive",
                "observed_drop_from_base": exp318["observed_drop_from_base"],
            },
            "exp319_task243_101": {
                "public_lb": exp319["public_lb"],
                "interpretation": "fail_stub_no_drop",
                "observed_drop_vs_base": exp319["observed_drop_vs_base"],
            },
            "exp321_task243_101_repair": {
                "public_lb": exp321["public_lb"],
                "interpretation": "repair_no_gain",
                "observed_gain_vs_base": exp321["observed_gain_vs_base"],
            },
        },
        "top_subsets_all_exp315_targets": subset_candidates(points, missing, 4)[:20],
        "top_subsets_excluding_confirmed_alive_193_275": subset_candidates(remaining_after_alive, missing, 4)[:20],
        "top_subsets_excluding_alive_and_nonrecoverable_101_243": subset_candidates(
            remaining_after_alive_and_pair, missing, 4
        )[:20],
        "decision": (
            "Do not spend another repair attempt on task101/task243 from public-code sources. "
            "The pair explains the no-drop probe arithmetically, but exp321 showed no recoverable "
            "public gain with the best full-arc source. Treat exp315 as diagnostic but not yet a "
            "reliable repair queue; resume fresh wide probing or move to GridSample queue."
        ),
        "next_recommended_experiment": (
            "exp323 fresh wide bisection probe on a new unprobed high-risk group, with pair-wise "
            "subset-sum uniqueness checked before submission; alternatively start task251 GridSample "
            "if submission budget or probe confidence is constrained."
        ),
        "submission_decision": "no_submit_audit_only",
        "leakage_risk": "low: analysis uses only prior public submission scores and local experiment metadata.",
        "overfitting_risk": (
            "medium: the audit reasons about public LB diagnostics, so it should guide probe design "
            "rather than direct final model selection."
        ),
    }

    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {EXP_ID}",
        "",
        "## 目的",
        "",
        "exp315 の missing drop を、exp318/319/321 の follow-up 結果込みで再監査する。",
        "",
        "## 結果",
        "",
        f"- status: `{result['status']}`",
        f"- exp315 missing_drop_vs_all_alive: `{missing}`",
        "- exp318: task193/task275 は all-alive",
        "- exp319: task101/task243 fail-stub は no-drop",
        "- exp321: task101/task243 repair は no-gain",
        "",
        "## Top Subsets After Exclusions",
        "",
    ]
    for row in result["top_subsets_excluding_alive_and_nonrecoverable_101_243"][:10]:
        lines.append(
            f"- tasks `{row['tasks']}` / sum `{row['points_sum']}` / error `{row['absolute_error']}`"
        )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            result["decision"],
            "",
            "## 次",
            "",
            result["next_recommended_experiment"],
            "",
            "## リスク",
            "",
            f"- leakage risk: {result['leakage_risk']}",
            f"- overfitting risk: {result['overfitting_risk']}",
        ]
    )
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
