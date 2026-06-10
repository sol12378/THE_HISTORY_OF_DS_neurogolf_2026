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


EXP_DIR = ROOT / "experiments" / "exp166_task025_line_projection_publiczero_repair"
BASE_EXP = ROOT / "experiments" / "exp152_task023_b035_repair_probe"
BASE_ZIP = BASE_EXP / "submission.zip"
TASK_ID = 25
CANDIDATE_PATH = ROOT / "experiments" / "exp165_task025_line_projection_onnx_probe" / "candidate_sanitized.onnx"
BASE_PUBLIC_LB = 5956.28


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    candidate = CANDIDATE_PATH.read_bytes()
    ok, reason, passed, failed = validate_examples(utils, candidate, TASK_ID, arc_gen_sample=-1)
    memory, params, score_reason = score_model(utils, candidate, TASK_ID, "exp166_task025_line_projection", EXP_DIR)
    candidate_cost = int(memory + params) if memory is not None and params is not None else None
    candidate_points = 11.894579343325983

    out_zip = EXP_DIR / "submission.zip"
    with zipfile.ZipFile(BASE_ZIP) as zin, zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name in sorted(zin.namelist()):
            tid = int(Path(name).stem.replace("task", ""))
            raw = candidate if tid == TASK_ID else zin.read(name)
            zout.writestr(name, raw)

    result: dict[str, Any] = {
        "exp_id": "exp166_task025_line_projection_publiczero_repair",
        "date": "2026-06-10",
        "status": "repair_zip_ready",
        "purpose": "Repair task025 public-zero using exp164 input-only guide-line projection rule lowered by exp165.",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_public_lb": BASE_PUBLIC_LB,
        "task_id": TASK_ID,
        "candidate_source": "exp165_task025_line_projection_onnx_probe",
        "candidate_path": str(CANDIDATE_PATH.relative_to(ROOT)),
        "candidate_sha256": sha256(candidate),
        "candidate_validation": {"ok": ok, "reason": reason, "passed": passed, "failed": failed, "status": f"{passed}_pass_{failed}_fail"},
        "candidate_cost": candidate_cost,
        "candidate_score_reason": score_reason,
        "candidate_points_if_public_alive": candidate_points,
        "expected_public_lb_if_pass": BASE_PUBLIC_LB + candidate_points,
        "zip_sanity": zip_sanity(out_zip),
        "outputs": {"submission_zip": str(out_zip.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "submission_decision": "submit_publiczero_repair_probe" if ok and candidate_cost is not None else "do_not_submit_validation_or_score_failed",
        "leakage_risk": "low-to-medium: candidate is an input-only geometric rule, but task025 was selected via public-zero bisection.",
        "overfitting_risk": "medium: full-line guide assumption passes all local generated examples but still needs LB probe.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp166_task025_line_projection_publiczero_repair",
        "",
        "## 目的",
        "",
        "exp152 current public bestに、exp165のtask025 guide-line projection candidateを重ねてpublic-zero repair probe zipを作る。",
        "",
        "## 結果",
        "",
        f"- candidate_validation: `{result['candidate_validation']['status']}`",
        f"- candidate_cost: `{result['candidate_cost']}`",
        f"- expected_public_lb_if_pass: `{result['expected_public_lb_if_pass']}`",
        f"- zip sha256: `{result['zip_sanity']['sha256']}`",
        f"- submission_decision: {result['submission_decision']}",
        "",
        "## リスク",
        "",
        result["leakage_risk"],
        result["overfitting_risk"],
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
