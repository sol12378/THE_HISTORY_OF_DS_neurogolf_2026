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


EXP_DIR = ROOT / "experiments" / "exp151_task023_candidate_validation_audit"
TASK_ID = 23
MANIFEST_NAMES = {"selected_manifest.csv", "rewrite_manifest.csv"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def manifest_rows() -> list[dict[str, str]]:
    rows = []
    for manifest in (ROOT / "experiments").rglob("*.csv"):
        if manifest.name not in MANIFEST_NAMES:
            continue
        for row in read_csv(manifest):
            try:
                tid = int(row.get("task_id", ""))
            except ValueError:
                continue
            if tid != TASK_ID:
                continue
            rows.append({**row, "exp": manifest.parent.name, "manifest": str(manifest.relative_to(ROOT))})
    return rows


def norm_cost(row: dict[str, str]) -> str:
    for key in ["cost", "new_cost", "candidate_cost", "delta_cost"]:
        if row.get(key):
            return row[key]
    return ""


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    rows = []
    seen: set[str] = set()
    source_rows = manifest_rows()
    for row in source_rows:
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
        memory, params, score_reason = score_model(utils, raw, TASK_ID, f"exp151_{row['exp']}", EXP_DIR)
        cost = int(memory + params) if memory is not None and params is not None else None
        out_path = EXP_DIR / f"task023_{len(seen):02d}_{row['exp']}.onnx"
        out_path.write_bytes(raw)
        rows.append(
            {
                **row,
                "manifest_cost": norm_cost(row),
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
    out_csv = EXP_DIR / "task023_candidate_audit.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_csv.write_text("", encoding="utf-8")
    accepted = [r for r in rows if r.get("submit_candidate") is True or str(r.get("submit_candidate")) == "True"]
    result: dict[str, Any] = {
        "exp_id": "exp151_task023_candidate_validation_audit",
        "date": "2026-06-10",
        "status": "candidate_audit_complete",
        "task_id": TASK_ID,
        "source_rows": len(source_rows),
        "unique_raw_candidates": len([r for r in rows if r.get("raw_sha256")]),
        "full_ok_candidates": len(accepted),
        "best_candidates": accepted[:8],
        "outputs": {"candidate_audit": str(out_csv.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "decision": "Use a public-distinct full_ok candidate for a single-task repair probe if any candidate is not the current public-zero raw.",
        "leakage_risk": "medium-to-high: existing public/teacher candidates may be lookup-like.",
        "overfitting_risk": "medium-to-high: full local validation is not sufficient for public/private robustness.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp151_task023_candidate_validation_audit",
        "",
        "## Hypothesis",
        "",
        "task023の既存source候補の中に、public-zeroを修復できるfull validation候補がある可能性がある。",
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
