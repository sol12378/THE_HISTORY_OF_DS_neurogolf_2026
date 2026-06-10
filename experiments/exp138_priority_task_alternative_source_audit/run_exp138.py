from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "exp138_priority_task_alternative_source_audit"
CONSENSUS = ROOT / "experiments" / "exp137_private_failure_consensus_audit" / "consensus_audit.csv"

TARGETS = [13, 2, 29, 9, 24, 18]
MANIFEST_NAMES = {"selected_manifest.csv", "rewrite_manifest.csv"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_result(path: Path) -> dict[str, Any]:
    result = path.parent / "result.json"
    if not result.exists():
        return {}
    try:
        return json.loads(result.read_text(encoding="utf-8"))
    except Exception:
        return {}


def norm_cost(row: dict[str, str]) -> float | None:
    for key in ["cost", "new_cost", "candidate_cost", "delta_cost", "official_cost"]:
        value = row.get(key, "")
        if value in {"", None}:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def norm_points(row: dict[str, str]) -> float | None:
    for key in ["local_points", "new_points", "candidate_points", "delta_points", "points"]:
        value = row.get(key, "")
        if value in {"", None}:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def row_source(row: dict[str, str], manifest_path: Path) -> str:
    source = row.get("source", "")
    if source:
        return source
    return manifest_path.parent.name


def collect_manifest_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target_set = set(TARGETS)
    for manifest in (ROOT / "experiments").rglob("*.csv"):
        if manifest.name not in MANIFEST_NAMES or not manifest.is_file():
            continue
        exp_name = manifest.parent.name
        result = read_result(manifest)
        exp_status = str(result.get("status", ""))
        validation_hint = ""
        if result.get("validation_all_ok") is True:
            validation_hint = "validation_all_ok"
        elif result.get("full_arc_gen_validation", {}).get("status") == "pass":
            validation_hint = "full_arc_gen_validation_pass"
        elif "full" in exp_status and "pass" in exp_status:
            validation_hint = exp_status
        for row in read_csv(manifest):
            try:
                tid = int(row.get("task_id", ""))
            except ValueError:
                continue
            if tid not in target_set:
                continue
            cost = norm_cost(row)
            points = norm_points(row)
            out.append(
                {
                    "task_id": tid,
                    "exp": exp_name,
                    "manifest": str(manifest.relative_to(ROOT)),
                    "source": row_source(row, manifest),
                    "template_name": row.get("template_name", ""),
                    "route": row.get("route", ""),
                    "cost": cost if cost is not None else "",
                    "points": points if points is not None else "",
                    "status": row.get("status", ""),
                    "reason": row.get("reason", ""),
                    "sha256": row.get("sha256", ""),
                    "exp_status": exp_status,
                    "validation_hint": validation_hint,
                }
            )
    return out


def collect_candidate_eval_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target_set = set(TARGETS)
    for path in ROOT.glob("experiments/**/candidate_eval.csv"):
        exp_name = path.parent.name
        for row in read_csv(path):
            try:
                tid = int(row.get("task_id", ""))
            except ValueError:
                continue
            if tid not in target_set:
                continue
            cost = norm_cost(row)
            points = norm_points(row)
            out.append(
                {
                    "task_id": tid,
                    "exp": exp_name,
                    "manifest": str(path.relative_to(ROOT)),
                    "source": exp_name,
                    "template_name": row.get("template_name", ""),
                    "route": row.get("route", ""),
                    "cost": cost if cost is not None else "",
                    "points": points if points is not None else "",
                    "status": row.get("status", ""),
                    "reason": row.get("reason", ""),
                    "sha256": row.get("sha256", ""),
                    "validation_status": row.get("validation_status", ""),
                }
            )
    return out


def rank_alternatives(manifest_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    consensus_rows = {int(r["task_id"]): r for r in read_csv(CONSENSUS)}
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        by_task[int(row["task_id"])].append(row)

    summaries = []
    priority = []
    for tid in TARGETS:
        current = consensus_rows[tid]
        current_cost = float(current["cost"])
        rows = by_task.get(tid, [])
        unique_sources = sorted({str(r["source"]) for r in rows})
        cheaper = [r for r in rows if r["cost"] != "" and float(r["cost"]) < current_cost]
        non_current = [r for r in rows if r["source"] != current["source"]]
        fullish = [r for r in rows if r.get("validation_hint")]
        cand_task = [r for r in candidate_rows if int(r["task_id"]) == tid]
        improved_cands = [r for r in cand_task if r.get("status") == "improved"]
        full_pass_cands = [r for r in cand_task if str(r.get("validation_status", "")).endswith("_pass_0_fail")]
        best_alt = None
        if non_current:
            best_alt = sorted(
                non_current,
                key=lambda r: (
                    0 if r.get("validation_hint") else 1,
                    float(r["cost"]) if r["cost"] != "" else 1e18,
                ),
            )[0]
            priority.append(
                {
                    "task_id": tid,
                    "current_source": current["source"],
                    "current_cost": current_cost,
                    "alternative_source": best_alt["source"],
                    "alternative_exp": best_alt["exp"],
                    "alternative_cost": best_alt["cost"],
                    "alternative_validation_hint": best_alt.get("validation_hint", ""),
                    "action": "validate_alternative_source" if best_alt.get("validation_hint") else "inspect_alternative_before_validation",
                }
            )
        summaries.append(
            {
                "task_id": tid,
                "current_source": current["source"],
                "current_cost": current_cost,
                "current_points": current["points"],
                "manifest_rows": len(rows),
                "unique_sources": len(unique_sources),
                "sources": " | ".join(unique_sources[:12]),
                "non_current_rows": len(non_current),
                "cheaper_rows": len(cheaper),
                "full_validation_hint_rows": len(fullish),
                "candidate_eval_rows": len(cand_task),
                "improved_candidate_rows": len(improved_cands),
                "full_pass_candidate_rows": len(full_pass_cands),
                "best_alt_source": best_alt["source"] if best_alt else "",
                "best_alt_exp": best_alt["exp"] if best_alt else "",
                "best_alt_cost": best_alt["cost"] if best_alt else "",
            }
        )
    return summaries, priority


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp138_priority_task_alternative_source_audit",
        "",
        "## Hypothesis",
        "",
        "private failure consensus上位taskには、既存実験内に代替sourceまたはfull-validation済み候補があり、bisection提出前に修復候補を絞れる。",
        "",
        "## Result",
        "",
        f"- targets: `{result['targets']}`",
        f"- manifest rows: `{result['manifest_row_count']}`",
        f"- candidate eval rows: `{result['candidate_eval_row_count']}`",
        "",
        "## Task Summary",
        "",
    ]
    for row in result["task_summaries"]:
        lines.append(
            f"- task{int(row['task_id']):03d}: sources `{row['unique_sources']}`, non-current `{row['non_current_rows']}`, "
            f"full hints `{row['full_validation_hint_rows']}`, best_alt `{row['best_alt_source']}` from `{row['best_alt_exp']}` cost `{row['best_alt_cost']}`"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            result["decision"],
            "",
            "## Leakage / Overfitting Risk",
            "",
            result["leakage_risk"],
            result["overfitting_risk"],
        ]
    )
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = collect_manifest_rows()
    candidate_rows = collect_candidate_eval_rows()
    summaries, priority = rank_alternatives(manifest_rows, candidate_rows)

    write_csv(EXP_DIR / "manifest_alternatives.csv", manifest_rows)
    write_csv(EXP_DIR / "candidate_eval_hits.csv", candidate_rows)
    write_csv(EXP_DIR / "task_alternative_summary.csv", summaries)
    write_csv(EXP_DIR / "next_validation_queue.csv", priority)

    result = {
        "exp_id": "exp138_priority_task_alternative_source_audit",
        "date": "2026-06-10",
        "status": "alternative_audit_ready",
        "targets": TARGETS,
        "source_exp": "exp137_private_failure_consensus_audit",
        "manifest_row_count": len(manifest_rows),
        "candidate_eval_row_count": len(candidate_rows),
        "task_summaries": summaries,
        "next_validation_queue": priority,
        "outputs": {
            "manifest_alternatives": str((EXP_DIR / "manifest_alternatives.csv").relative_to(ROOT)),
            "candidate_eval_hits": str((EXP_DIR / "candidate_eval_hits.csv").relative_to(ROOT)),
            "task_alternative_summary": str((EXP_DIR / "task_alternative_summary.csv").relative_to(ROOT)),
            "next_validation_queue": str((EXP_DIR / "next_validation_queue.csv").relative_to(ROOT)),
        },
        "elapsed_s": round(time.time() - t0, 3),
        "decision": "Validate non-current alternatives for the six consensus tasks first; if none are robust, build a bisection probe group from task013/002/029/009.",
        "leakage_risk": "low: scans existing manifests/candidate evals only; no new model is adopted.",
        "overfitting_risk": "medium: alternative sources may share the same hidden failure mode and require full validation plus LB calibration.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
