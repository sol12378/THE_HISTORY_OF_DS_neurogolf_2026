from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


EXP_ID = "exp056_submission_calibration_inventory"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
LB_TRACKING = ROOT / "neurogolf_kaggle_obsidian" / "09_Submissions" / "LB_Tracking.md"


KNOWN_LB = {
    "exp001_baseline": {"lb": 14.50, "ref": "53383536", "role": "baseline"},
    "exp041_task145_deeper_mul_chain": {"lb": 3417.71, "ref": "53414511", "role": "high_risk_local_upper"},
    "exp005_top_cost_rewrite_strict": {"lb": 5929.89, "ref": "53414978", "role": "strict_seed_calibration"},
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_score(payload: dict[str, Any]) -> float | None:
    keys = [
        "new_local_estimate",
        "local_estimate",
        "teacher_local_score",
        "strict_seed_score",
        "strict_seed_score_from_inventory",
        "projected_score_if_all_cost_le_250_floor",
    ]
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    strict = payload.get("strict_seed")
    if isinstance(strict, dict) and isinstance(strict.get("local_estimate"), (int, float)):
        return float(strict["local_estimate"])
    return None


def risk_label(exp_id: str, payload: dict[str, Any]) -> tuple[str, str]:
    text = json.dumps(payload, ensure_ascii=False).lower()
    if exp_id == "exp005_top_cost_rewrite_strict":
        return "low", "strict seed already Kaggle-calibrated"
    if exp_id == "exp004_public_blend_relaxed_static":
        return "low_duplicate", "strict seed predecessor; exp005 already calibrates the same environment with a tiny prune delta"
    if exp_id == "exp001_baseline":
        return "low", "minimal baseline; already submitted"
    high_risk_local_upper = {
        "exp016_top100_rewrite_campaign",
        "exp023_graph_surgery_exp016",
        "exp037_fullarc_filter_exp036",
        "exp038_fullarc_gated_noop_bypass",
        "exp039_deep_fullarc_safe_bypass",
        "exp040_deeper_fullarc_safe_bypass",
        "exp041_task145_deeper_mul_chain",
    }
    if exp_id in high_risk_local_upper:
        return "high", "same local-upper lineage as exp041; LB collapse already measured by exp041"
    if "signature" in text or "teacher" in text or "local upper" in text or exp_id in {"exp016_top100_rewrite_campaign", "exp041_task145_deeper_mul_chain"}:
        return "high", "teacher/signature/local-upper dependence; submit only for calibration, not as trusted progress"
    if "full_arc" in text or "full-arc" in text or "full validation" in text:
        return "medium", "local validation exists but LB calibration unknown"
    return "unknown", "insufficient metadata"


def submit_value(exp_id: str, has_zip: bool, known_lb: dict[str, Any] | None, risk: str) -> tuple[str, str]:
    if not has_zip:
        return "none", "no submission.zip"
    if known_lb is not None:
        return "done", f"already submitted ref {known_lb['ref']} LB {known_lb['lb']}"
    if risk == "low":
        return "high", "safe calibration candidate"
    if risk == "low_duplicate":
        return "low", "duplicate strict-seed calibration; lower value than a new single-task delta"
    if risk == "medium":
        return "medium", "could submit as small calibration if it is a strict-seed delta"
    if risk == "high":
        return "low", "only useful as high-risk collapse measurement; avoid spending slots unless needed"
    return "low", "metadata incomplete"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for exp_dir in sorted(p for p in EXPERIMENTS.iterdir() if p.is_dir()):
        result_path = exp_dir / "result.json"
        payload = read_json(result_path) if result_path.exists() else {}
        exp_id = payload.get("exp_id") or exp_dir.name
        has_zip = (exp_dir / "submission.zip").exists()
        score = find_score(payload)
        known = KNOWN_LB.get(exp_id)
        risk, risk_reason = risk_label(exp_id, payload)
        value, value_reason = submit_value(exp_id, has_zip, known, risk)
        local_lb_gap = None
        if known and score is not None:
            local_lb_gap = float(known["lb"]) - score
        rows.append(
            {
                "exp_id": exp_id,
                "has_submission_zip": has_zip,
                "local_score": score if score is not None else "",
                "known_lb": known["lb"] if known else "",
                "submission_ref": known["ref"] if known else "",
                "local_lb_gap": local_lb_gap if local_lb_gap is not None else "",
                "risk": risk,
                "submit_value": value,
                "risk_reason": risk_reason,
                "submit_value_reason": value_reason,
            }
        )

    rows.sort(
        key=lambda r: (
            {"high": 0, "medium": 1, "low": 2, "done": 3, "none": 4}.get(str(r["submit_value"]), 9),
            str(r["risk"]),
            str(r["exp_id"]),
        )
    )

    fieldnames = [
        "exp_id",
        "has_submission_zip",
        "local_score",
        "known_lb",
        "submission_ref",
        "local_lb_gap",
        "risk",
        "submit_value",
        "risk_reason",
        "submit_value_reason",
    ]
    with (OUT_DIR / "submission_calibration_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    submit_ready = [r for r in rows if r["submit_value"] in {"high", "medium"}]
    known_rows = [r for r in rows if r["known_lb"] != ""]
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-06",
        "status": "inventory_ready",
        "purpose": "calibrate local vs Kaggle LB while pursuing all-task cost 250-600 lowering",
        "candidate_count": len(rows),
        "submission_zip_count": sum(1 for r in rows if r["has_submission_zip"]),
        "known_lb_count": len(known_rows),
        "known_lb_rows": known_rows,
        "submit_ready_count": len(submit_ready),
        "submit_ready_rows": submit_ready[:20],
        "decision": "Do not spend submissions on diagnostic/no-hit experiments. Next LB submission should be the first official-valid strict-seed delta from object-role/bbox-local compiler; if none exists, keep exp005 as current calibration baseline.",
        "calibration_policy": [
            "submit low-risk strict seed or strict-seed delta bundles first",
            "record local, LB, ref, and local_lb_gap immediately",
            "avoid large high-risk teacher/signature bundles except explicit collapse-measurement submissions",
            "prefer single-task or single-family deltas to identify whether local/LB environments match",
        ],
        "outputs": {
            "inventory": "submission_calibration_inventory.csv",
            "notes": "notes.md",
        },
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = f"""# {EXP_ID}

## 目的

低cost化と並行して、local/LBの差を早く縮めるため、既存submission候補を棚卸しする。

## 結果

- candidate experiments: {len(rows)}
- submission.zipあり: {result['submission_zip_count']}
- known LBあり: {len(known_rows)}
- submit-ready候補: {len(submit_ready)}

## 既知LB

| exp | local | LB | gap | ref |
|---|---:|---:|---:|---|
"""
    for row in known_rows:
        notes += f"| {row['exp_id']} | {row['local_score']} | {row['known_lb']} | {row['local_lb_gap']} | {row['submission_ref']} |\n"
    notes += """
## Decision

現時点では、exp005 strict seedが唯一の健全な較正基準。次に提出すべきなのは、object-role/bbox-local compilerで作ったofficial-validなsingle-task deltaであり、no-hit診断実験やteacher lookup bundleではない。

ただし、ユーザー方針に従い、今後full-passかつofficial-validな小deltaが出たら、localだけで温存せず早めにKaggleへ提出してlocal/LB gapを記録する。
"""
    (OUT_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
