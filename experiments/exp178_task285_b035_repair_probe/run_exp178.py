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


EXP_DIR = ROOT / "experiments" / "exp178_task285_b035_repair_probe"
BASE_EXP = ROOT / "experiments" / "exp172_task133_158_b035_repair_probe"
BASE_ZIP = BASE_EXP / "submission.zip"
BASE_PUBLIC_LB = 5993.82
REPAIR_TASK = 285
REPAIR_RAW = ROOT / "experiments" / "exp175_next2o_candidate_validation_audit" / "task285_06_exp_b035_new_source_full_arc_blend.onnx"
EXPECTED_POINTS = 12.112175


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    repair_raw = REPAIR_RAW.read_bytes()

    ok, reason, passed, failed = validate_examples(utils, repair_raw, REPAIR_TASK, arc_gen_sample=-1)
    memory, params, score_reason = score_model(utils, repair_raw, REPAIR_TASK, "exp178_task285", EXP_DIR)
    cost = int(memory + params) if memory is not None and params is not None else None
    repair_validation = {
        "task_id": REPAIR_TASK,
        "raw_sha256": sha256(repair_raw),
        "ok": ok,
        "reason": reason,
        "passed": passed,
        "failed": failed,
        "status": f"{passed}_pass_{failed}_fail",
        "official_cost": cost if cost is not None else "",
        "score_reason": score_reason,
    }

    out_zip = EXP_DIR / "submission.zip"
    with zipfile.ZipFile(BASE_ZIP) as zin, zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name in sorted(zin.namelist()):
            tid = int(Path(name).stem.replace("task", ""))
            zout.writestr(name, repair_raw if tid == REPAIR_TASK else zin.read(name))

    result: dict[str, Any] = {
        "exp_id": "exp178_task285_b035_repair_probe",
        "date": "2026-06-10",
        "status": "repair_zip_ready",
        "purpose": "Repair exp174-identified public-zero task285 using distinct exp_b035 full-local-valid raw on top of exp172.",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_public_lb": BASE_PUBLIC_LB,
        "repair": {REPAIR_TASK: str(REPAIR_RAW.relative_to(ROOT))},
        "repair_validation": repair_validation,
        "expected_gain_if_public_pass": EXPECTED_POINTS,
        "expected_lb_if_public_pass": BASE_PUBLIC_LB + EXPECTED_POINTS,
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
    row = result["repair_validation"]
    lines = [
        "# exp178_task285_b035_repair_probe",
        "",
        "## 目的",
        "",
        "exp174でpublic-zeroと推定されたtask285を、exp_b035のdistinct full-local-valid rawで修復する。",
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
        f"- task{int(row['task_id']):03d}: `{row['status']}`, cost `{row['official_cost']}`, sha `{row['raw_sha256']}`",
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
