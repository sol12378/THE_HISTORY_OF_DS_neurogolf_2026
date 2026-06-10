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


EXP_DIR = ROOT / "experiments" / "exp152_task023_b035_repair_probe"
BASE_EXP = ROOT / "experiments" / "exp144_task018_b035_repair_probe"
BASE_ZIP = BASE_EXP / "submission.zip"
TASK_ID = 23
CANDIDATE_PATH = ROOT / "experiments" / "exp151_task023_candidate_validation_audit" / "task023_05_exp_b035_new_source_full_arc_blend.onnx"


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    candidate = CANDIDATE_PATH.read_bytes()
    ok, reason, passed, failed = validate_examples(utils, candidate, TASK_ID, arc_gen_sample=-1)
    memory, params, score_reason = score_model(utils, candidate, TASK_ID, "exp152_task023_b035", EXP_DIR)
    candidate_cost = int(memory + params) if memory is not None and params is not None else None

    out_zip = EXP_DIR / "submission.zip"
    with zipfile.ZipFile(BASE_ZIP) as zin, zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name in sorted(zin.namelist()):
            tid = int(Path(name).stem.replace("task", ""))
            raw = candidate if tid == TASK_ID else zin.read(name)
            zout.writestr(name, raw)

    result: dict[str, Any] = {
        "exp_id": "exp152_task023_b035_repair_probe",
        "date": "2026-06-10",
        "status": "repair_zip_ready",
        "purpose": "Repair task023 public-zero by replacing current raw with exp_b035 full-validation candidate on top of exp144 public-best bundle.",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_public_lb": 5942.48,
        "task_id": TASK_ID,
        "candidate_source": "exp_b035_new_source_full_arc_blend",
        "candidate_path": str(CANDIDATE_PATH.relative_to(ROOT)),
        "candidate_sha256": sha256(candidate),
        "candidate_validation": {"ok": ok, "reason": reason, "passed": passed, "failed": failed, "status": f"{passed}_pass_{failed}_fail"},
        "candidate_cost": candidate_cost,
        "candidate_score_reason": score_reason,
        "candidate_points_if_public_alive": 13.807115,
        "expected_public_lb_if_pass": 5942.48 + 13.807115,
        "zip_sanity": zip_sanity(out_zip),
        "outputs": {"submission_zip": str(out_zip.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "submission_decision": "submit_repair_probe_after_sanity" if ok and candidate_cost is not None else "do_not_submit_validation_or_score_failed",
        "leakage_risk": "high: candidate is from an existing public/source blend and may be public-overfit.",
        "overfitting_risk": "high: task023 was identified through public bisection; repair must be LB-probed and not assumed private-safe.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp152_task023_b035_repair_probe",
        "",
        "## Hypothesis",
        "",
        "exp150でpublic-zeroと推定したtask023を、full validation OKのexp_b035 rawへ差し替えるとpublic LBが回復する。",
        "",
        "## Result",
        "",
        f"- status: `{result['status']}`",
        f"- base_exp: `{result['base_exp']}`",
        f"- candidate_source: `{result['candidate_source']}`",
        f"- candidate_validation: `{result['candidate_validation']['status']}`",
        f"- candidate_cost: `{result['candidate_cost']}`",
        f"- expected_public_lb_if_pass: `{result['expected_public_lb_if_pass']}`",
        f"- zip sha256: `{result['zip_sanity']['sha256']}`",
        "",
        "## Decision",
        "",
        str(result["submission_decision"]),
        "",
        "## Leakage / Overfitting Risk",
        "",
        str(result["leakage_risk"]),
        str(result["overfitting_risk"]),
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
