from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import load_neurogolf_utils, score_model, sha256, validate_examples, zip_sanity


EXP_DIR = ROOT / "experiments" / "exp186_task187_massimiliano_repair_probe"
BASE_EXP = ROOT / "experiments" / "exp178_task285_b035_repair_probe"
REPAIR_RAW = ROOT / "experiments" / "exp185_risk_next4d_candidate_validation_audit" / "task187_01_exp002_public_blend_6500_fast.onnx"
BASE_PUBLIC_LB = 6005.93
TASK_ID = 187
EXPECTED_GAIN = 13.435307852707972


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()

    repair_raw = REPAIR_RAW.read_bytes()
    ok, reason, passed, failed = validate_examples(utils, repair_raw, TASK_ID, arc_gen_sample=-1)
    memory, params, score_reason = score_model(utils, repair_raw, TASK_ID, "exp186_task187_repair", EXP_DIR)
    repair_cost = int(memory + params) if memory is not None and params is not None else None

    base_zip = BASE_EXP / "submission.zip"
    out_zip = EXP_DIR / "submission.zip"
    with zipfile.ZipFile(base_zip) as zin, zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name in sorted(zin.namelist()):
            raw = repair_raw if name == f"task{TASK_ID:03d}.onnx" else zin.read(name)
            zout.writestr(name, raw)

    result = {
        "exp_id": "exp186_task187_massimiliano_repair_probe",
        "date": "2026-06-10",
        "status": "repair_zip_ready",
        "purpose": "Replace public-zero task187 in exp178 with a distinct full-local-valid massimilianoghiotto_6254 raw from exp185 audit.",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_public_lb": BASE_PUBLIC_LB,
        "task_id": TASK_ID,
        "expected_gain_if_public_pass": EXPECTED_GAIN,
        "expected_lb_if_public_pass": BASE_PUBLIC_LB + EXPECTED_GAIN,
        "repair_source": str(REPAIR_RAW.relative_to(ROOT)),
        "repair_sha256": sha256(repair_raw),
        "validation": {"ok": ok, "reason": reason, "passed": passed, "failed": failed, "status": f"{passed}_pass_{failed}_fail"},
        "repair_cost": repair_cost,
        "score_reason": score_reason,
        "zip_sanity": zip_sanity(out_zip),
        "outputs": {"submission_zip": str(out_zip.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "submission_decision": "submit: exp183 missing drop identifies task187 as public-zero and this candidate is distinct full-local-valid.",
        "leakage_risk": "medium-to-high: repair raw is from an existing public/source blend candidate.",
        "overfitting_risk": "medium-to-high: full local validation plus public-zero evidence may not guarantee private robustness.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict) -> None:
    lines = [
        "# exp186_task187_massimiliano_repair_probe",
        "",
        "## 目的",
        "",
        "exp183でpublic-zeroと推定したtask187を、exp185監査で見つけたdistinct full-local-valid rawへ差し替える。",
        "",
        "## 結果",
        "",
        f"- status: `{result['status']}`",
        f"- base_public_lb: `{result['base_public_lb']}`",
        f"- expected_gain_if_public_pass: `{result['expected_gain_if_public_pass']}`",
        f"- expected_lb_if_public_pass: `{result['expected_lb_if_public_pass']}`",
        f"- repair_source: `{result['repair_source']}`",
        f"- repair_sha256: `{result['repair_sha256']}`",
        f"- validation: `{result['validation']['status']}`, reason `{result['validation']['reason']}`",
        f"- repair_cost: `{result['repair_cost']}`",
        f"- zip sha256: `{result['zip_sanity']['sha256']}`",
        "",
        "## 判断",
        "",
        str(result["submission_decision"]),
        "",
        "## リスク",
        "",
        str(result["leakage_risk"]),
        str(result["overfitting_risk"]),
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
