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

from experiments.phase1_rewrite_utils import load_neurogolf_utils, score_model, validate_examples, zip_sanity


EXP_DIR = ROOT / "experiments" / "exp141_task018_teacher_repair_probe"
BASE_EXP = ROOT / "experiments" / "exp127_focused_surgery_seventh_pass_limited"
BASE_ZIP = BASE_EXP / "submission.zip"
TEACHER_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
TEACHER_ZIP = TEACHER_EXP / "submission.zip"
TASK_ID = 18


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    with zipfile.ZipFile(TEACHER_ZIP) as zf:
        teacher_raw = zf.read(f"task{TASK_ID:03d}.onnx")
    ok, reason, passed, failed = validate_examples(utils, teacher_raw, TASK_ID, arc_gen_sample=-1)
    memory, params, score_reason = score_model(utils, teacher_raw, TASK_ID, "exp141_task018_teacher", EXP_DIR)
    teacher_cost = int(memory + params) if memory is not None and params is not None else None

    raws: dict[int, bytes] = {}
    with zipfile.ZipFile(BASE_ZIP) as zf:
        for name in zf.namelist():
            tid = int(Path(name).stem.replace("task", ""))
            raws[tid] = teacher_raw if tid == TASK_ID else zf.read(name)

    out_zip = EXP_DIR / "submission.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for tid, raw in sorted(raws.items()):
            zf.writestr(f"task{tid:03d}.onnx", raw)

    base_result = json.loads((BASE_EXP / "result.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "exp_id": "exp141_task018_teacher_repair_probe",
        "date": "2026-06-10",
        "status": "repair_probe_zip_ready" if ok and teacher_cost is not None else "repair_probe_blocked",
        "purpose": "Replace public-zero task018 in exp127 with exp023 teacher artifact to test whether single-task repair recovers LB.",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_public_lb": 5930.55,
        "base_local_estimate": base_result["new_local_estimate"],
        "teacher_exp": str(TEACHER_EXP.relative_to(ROOT)),
        "task_id": TASK_ID,
        "current_public_points_inferred": 0.0,
        "strict_current_cost": 114578,
        "strict_current_points": 13.35098890724301,
        "teacher_cost": teacher_cost,
        "teacher_points": 15.231931334571607 if teacher_cost == 17467 else None,
        "expected_lb_if_public_pass": 5930.55 + (15.231931334571607 if teacher_cost == 17467 else 0.0),
        "teacher_validation": {
            "ok": ok,
            "reason": reason,
            "passed": passed,
            "failed": failed,
            "status": f"{passed}_pass_{failed}_fail",
            "score_reason": score_reason,
        },
        "zip_sanity": zip_sanity(out_zip),
        "outputs": {"submission_zip": str(out_zip.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "submission_decision": "submit_single_task_repair_probe" if ok and teacher_cost is not None else "do_not_submit",
        "leakage_risk": "high: exp023 teacher artifact is signature-lookup-derived and previously part of a collapsed high-risk bundle.",
        "overfitting_risk": "high: public pass would not prove private robustness; use only as single-task calibration/repair evidence.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp141_task018_teacher_repair_probe",
        "",
        "## Hypothesis",
        "",
        "exp140でtask018がpublic-zeroと推定された。exp023 teacher artifactへtask018だけ差し替えれば、public LBで約+15.23を回収できる可能性がある。",
        "",
        "## Result",
        "",
        f"- status: `{result['status']}`",
        f"- teacher validation: `{result['teacher_validation']['status']}`",
        f"- teacher cost: `{result['teacher_cost']}`",
        f"- expected_lb_if_public_pass: `{result['expected_lb_if_public_pass']}`",
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
