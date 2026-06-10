from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import load_neurogolf_utils, score_model, sha256, validate_examples, zip_sanity


EXP_DIR = ROOT / "experiments" / "exp172_task133_158_b035_repair_probe"
BASE_EXP = ROOT / "experiments" / "exp166_task025_line_projection_publiczero_repair"
BASE_ZIP = BASE_EXP / "submission.zip"
BASE_PUBLIC_LB = 5968.18
REPAIRS = {
    133: ROOT / "experiments" / "exp170_next4m_candidate_validation_audit" / "task133_07_exp_b035_new_source_full_arc_blend.onnx",
    158: ROOT / "experiments" / "exp170_next4m_candidate_validation_audit" / "task158_06_exp_b035_new_source_full_arc_blend.onnx",
}
EXPECTED_POINTS = {
    133: 12.810626,
    158: 12.828177,
}


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    repair_raws = {tid: path.read_bytes() for tid, path in REPAIRS.items()}

    repair_validation = []
    for tid, raw in repair_raws.items():
        ok, reason, passed, failed = validate_examples(utils, raw, tid, arc_gen_sample=-1)
        memory, params, score_reason = score_model(utils, raw, tid, f"exp172_task{tid:03d}", EXP_DIR)
        cost = int(memory + params) if memory is not None and params is not None else None
        repair_validation.append(
            {
                "task_id": tid,
                "raw_sha256": sha256(raw),
                "ok": ok,
                "reason": reason,
                "passed": passed,
                "failed": failed,
                "status": f"{passed}_pass_{failed}_fail",
                "official_cost": cost if cost is not None else "",
                "score_reason": score_reason,
            }
        )

    out_zip = EXP_DIR / "submission.zip"
    with zipfile.ZipFile(BASE_ZIP) as zin, zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name in sorted(zin.namelist()):
            tid = int(Path(name).stem.replace("task", ""))
            zout.writestr(name, repair_raws.get(tid, zin.read(name)))

    expected_gain = sum(EXPECTED_POINTS.values())
    result: dict[str, Any] = {
        "exp_id": "exp172_task133_158_b035_repair_probe",
        "date": "2026-06-10",
        "status": "repair_zip_ready",
        "purpose": "Repair exp169-identified public-zero tasks 133 and 158 using distinct exp_b035 full-local-valid raws on top of exp166.",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_public_lb": BASE_PUBLIC_LB,
        "repairs": {tid: str(path.relative_to(ROOT)) for tid, path in REPAIRS.items()},
        "repair_validation": repair_validation,
        "expected_gain_if_public_pass": expected_gain,
        "expected_lb_if_public_pass": BASE_PUBLIC_LB + expected_gain,
        "zip_sanity": zip_sanity(out_zip),
        "outputs": {"submission_zip": str(out_zip.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "submission_decision": "submit_repair_probe",
        "leakage_risk": "high: exp_b035 public/source blend candidate; public-zero repair may not imply private robustness.",
        "overfitting_risk": "high: selected from public-zero bisection evidence and existing public sources.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp172_task133_158_b035_repair_probe",
        "",
        "## 目的",
        "",
        "exp169でpublic-zeroと推定された `task133` / `task158` を、exp_b035のdistinct full-local-valid rawで修復する。",
        "",
        "## 結果",
        "",
        f"- status: `{result['status']}`",
        f"- expected_gain_if_public_pass: `{result['expected_gain_if_public_pass']}`",
        f"- expected_lb_if_public_pass: `{result['expected_lb_if_public_pass']}`",
        f"- zip sha256: `{result['zip_sanity']['sha256']}`",
        "",
        "## Repair Validation",
        "",
    ]
    for row in result["repair_validation"]:
        lines.append(f"- task{int(row['task_id']):03d}: `{row['status']}`, cost `{row['official_cost']}`, sha `{row['raw_sha256']}`")
    lines.extend(["", "## 判断", "", str(result["submission_decision"]), "", "## リスク", "", str(result["leakage_risk"]), str(result["overfitting_risk"])])
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
