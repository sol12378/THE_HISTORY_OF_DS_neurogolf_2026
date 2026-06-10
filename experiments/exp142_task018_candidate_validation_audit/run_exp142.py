from __future__ import annotations

import csv
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import load_neurogolf_utils, score_model, sha256, validate_examples


EXP_DIR = ROOT / "experiments" / "exp142_task018_candidate_validation_audit"
ALT_CSV = ROOT / "experiments" / "exp138_priority_task_alternative_source_audit" / "manifest_alternatives.csv"
TASK_ID = 18


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def candidate_rows() -> list[dict[str, str]]:
    return [row for row in read_csv(ALT_CSV) if int(row["task_id"]) == TASK_ID]


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    rows = []
    seen: set[str] = set()
    for row in candidate_rows():
        manifest = ROOT / row["manifest"]
        zip_path = manifest.parent / "submission.zip"
        if not zip_path.exists():
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                raw = zf.read(f"task{TASK_ID:03d}.onnx")
        except Exception as exc:
            rows.append({**row, "raw_sha256": "", "audit_status": "read_failed", "audit_reason": str(exc)[:180]})
            continue
        raw_hash = sha256(raw)
        if raw_hash in seen:
            continue
        seen.add(raw_hash)
        ok, reason, passed, failed = validate_examples(utils, raw, TASK_ID, arc_gen_sample=-1)
        memory, params, score_reason = score_model(utils, raw, TASK_ID, f"exp142_{row['exp']}", EXP_DIR)
        cost = int(memory + params) if memory is not None and params is not None else None
        out_path = EXP_DIR / f"task018_{len(seen):02d}_{row['exp']}.onnx"
        out_path.write_bytes(raw)
        rows.append(
            {
                **row,
                "raw_sha256": raw_hash,
                "raw_path": str(out_path.relative_to(ROOT)),
                "audit_status": "full_ok" if ok else "full_fail",
                "audit_reason": reason,
                "validation_status": f"{passed}_pass_{failed}_fail",
                "official_cost": cost if cost is not None else "",
                "score_reason": score_reason,
                "submit_candidate": bool(ok and cost is not None),
            }
        )

    rows.sort(key=lambda r: (0 if r.get("audit_status") == "full_ok" else 1, float(r["official_cost"]) if r.get("official_cost") not in {"", None} else 1e18))
    out_csv = EXP_DIR / "task018_candidate_audit.csv"
    if rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_csv.write_text("", encoding="utf-8")
    accepted = [r for r in rows if r.get("submit_candidate") is True or str(r.get("submit_candidate")) == "True"]
    result: dict[str, Any] = {
        "exp_id": "exp142_task018_candidate_validation_audit",
        "date": "2026-06-10",
        "status": "candidate_audit_complete",
        "task_id": TASK_ID,
        "source_rows": len(candidate_rows()),
        "unique_raw_candidates": len([r for r in rows if r.get("raw_sha256")]),
        "full_ok_candidates": len(accepted),
        "best_candidates": accepted[:5],
        "outputs": {"candidate_audit": str(out_csv.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "decision": "Use the cheapest full_ok candidate for a single-task repair probe if any exist; otherwise task018 needs rule repair rather than existing-source replacement.",
        "leakage_risk": "medium-to-high: existing public/teacher candidates may be lookup-like.",
        "overfitting_risk": "medium-to-high: full local validation is not sufficient for public/private robustness.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp142_task018_candidate_validation_audit",
        "",
        "## Hypothesis",
        "",
        "task018の既存source候補の中に、full validationを通るrepair候補が残っている可能性がある。",
        "",
        "## Result",
        "",
        f"- source rows: `{result['source_rows']}`",
        f"- unique raw candidates: `{result['unique_raw_candidates']}`",
        f"- full_ok_candidates: `{result['full_ok_candidates']}`",
        "",
        "## Decision",
        "",
        str(result["decision"]),
        "",
        "## Leakage / Overfitting Risk",
        "",
        str(result["leakage_risk"]),
        str(result["overfitting_risk"]),
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
