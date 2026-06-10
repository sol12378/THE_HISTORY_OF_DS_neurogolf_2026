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


EXP_DIR = ROOT / "experiments" / "exp189_risk_next4f_candidate_validation_audit"
TARGETS = [328, 301, 306, 387]
MANIFEST_NAMES = {"selected_manifest.csv", "rewrite_manifest.csv"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def manifest_rows(task_id: int) -> list[dict[str, str]]:
    rows = []
    for manifest in (ROOT / "experiments").rglob("*.csv"):
        if manifest.name not in MANIFEST_NAMES:
            continue
        for row in read_csv(manifest):
            try:
                tid = int(row.get("task_id", ""))
            except ValueError:
                continue
            if tid == task_id:
                rows.append({**row, "exp": manifest.parent.name, "manifest": str(manifest.relative_to(ROOT))})
    return rows


def audit_task(utils: Any, task_id: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    source_rows = manifest_rows(task_id)
    for row in source_rows:
        manifest = ROOT / row["manifest"]
        zip_path = manifest.parent / "submission.zip"
        if not zip_path.exists():
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                raw = zf.read(f"task{task_id:03d}.onnx")
        except Exception as exc:
            rows.append({"task_id": task_id, **row, "raw_sha256": "", "audit_status": "read_failed", "audit_reason": str(exc)[:180]})
            continue
        raw_hash = sha256(raw)
        if raw_hash in seen:
            continue
        seen.add(raw_hash)
        ok, reason, passed, failed = validate_examples(utils, raw, task_id, arc_gen_sample=-1)
        memory, params, score_reason = score_model(utils, raw, task_id, f"exp189_task{task_id:03d}_{row['exp']}", EXP_DIR)
        cost = int(memory + params) if memory is not None and params is not None else None
        out_path = EXP_DIR / f"task{task_id:03d}_{len(seen):02d}_{row['exp']}.onnx"
        out_path.write_bytes(raw)
        rows.append(
            {
                "task_id": task_id,
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
    accepted = [r for r in rows if r.get("submit_candidate") is True]
    return rows, {
        "task_id": task_id,
        "source_rows": len(source_rows),
        "unique_raw_candidates": len([r for r in rows if r.get("raw_sha256")]),
        "full_ok_candidates": len(accepted),
        "best_candidates": accepted[:5],
    }


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    all_rows: list[dict[str, Any]] = []
    summaries = []
    for task_id in TARGETS:
        rows, summary = audit_task(utils, task_id)
        all_rows.extend(rows)
        summaries.append(summary)

    out_csv = EXP_DIR / "risk_next4f_candidate_audit.csv"
    if all_rows:
        fieldnames = sorted({key for row in all_rows for key in row.keys()})
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        out_csv.write_text("", encoding="utf-8")

    result = {
        "exp_id": "exp189_risk_next4f_candidate_validation_audit",
        "date": "2026-06-10",
        "status": "candidate_audit_complete",
        "targets": TARGETS,
        "summaries": summaries,
        "full_ok_total": sum(s["full_ok_candidates"] for s in summaries),
        "outputs": {"candidate_audit": str(out_csv.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "decision": "Use this audit if exp188 identifies a public-zero target among task328/301/306/387.",
        "leakage_risk": "medium-to-high: existing public/teacher candidates may be lookup-like.",
        "overfitting_risk": "medium-to-high: full local validation is not sufficient for public/private robustness.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp189_risk_next4f_candidate_validation_audit",
        "",
        "## 目的",
        "",
        "exp188対象 `328/301/306/387` について、既存source候補のfull validation / costを事前監査する。",
        "",
        "## 結果",
        "",
        f"- targets: `{result['targets']}`",
        f"- full_ok_total: `{result['full_ok_total']}`",
        "",
        "```json",
        json.dumps(result["summaries"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 判断",
        "",
        str(result["decision"]),
        "",
        "## リスク",
        "",
        str(result["leakage_risk"]),
        str(result["overfitting_risk"]),
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
