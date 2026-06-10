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

from experiments.phase1_rewrite_utils import load_neurogolf_utils, point, validate_examples, zip_sanity


EXP_DIR = ROOT / "experiments" / "exp143_task018_beicicc_repair_probe"
BASE_EXP = ROOT / "experiments" / "exp127_focused_surgery_seventh_pass_limited"
BASE_ZIP = BASE_EXP / "submission.zip"
CANDIDATE_RAW = ROOT / "experiments" / "exp142_task018_candidate_validation_audit" / "task018_05_exp130_public_code_6285_floor.onnx"
TASK_ID = 18
CANDIDATE_COST_FROM_MANIFEST = 33897


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    raw = CANDIDATE_RAW.read_bytes()
    ok, reason, passed, failed = validate_examples(utils, raw, TASK_ID, arc_gen_sample=-1)
    raws: dict[int, bytes] = {}
    with zipfile.ZipFile(BASE_ZIP) as zf:
        for name in zf.namelist():
            tid = int(Path(name).stem.replace("task", ""))
            raws[tid] = raw if tid == TASK_ID else zf.read(name)

    out_zip = EXP_DIR / "submission.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for tid, item in sorted(raws.items()):
            zf.writestr(f"task{tid:03d}.onnx", item)

    base_result = json.loads((BASE_EXP / "result.json").read_text(encoding="utf-8"))
    candidate_points = point(CANDIDATE_COST_FROM_MANIFEST)
    result: dict[str, Any] = {
        "exp_id": "exp143_task018_beicicc_repair_probe",
        "date": "2026-06-10",
        "status": "repair_probe_zip_ready" if ok else "repair_probe_blocked",
        "purpose": "Replace public-zero task018 in exp127 with beicicc/public-code candidate that passes full local validation.",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_public_lb": 5930.55,
        "base_local_estimate": base_result["new_local_estimate"],
        "task_id": TASK_ID,
        "candidate_raw": str(CANDIDATE_RAW.relative_to(ROOT)),
        "candidate_cost_from_manifest": CANDIDATE_COST_FROM_MANIFEST,
        "candidate_points_from_manifest": candidate_points,
        "expected_lb_if_public_pass": 5930.55 + candidate_points,
        "candidate_validation": {
            "ok": ok,
            "reason": reason,
            "passed": passed,
            "failed": failed,
            "status": f"{passed}_pass_{failed}_fail",
        },
        "zip_sanity": zip_sanity(out_zip),
        "outputs": {"submission_zip": str(out_zip.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "submission_decision": "submit_single_task_repair_probe" if ok else "do_not_submit",
        "leakage_risk": "high: beicicc public-code source collapsed in exp130; this is a single-task calibration probe, not a final private-safe repair yet.",
        "overfitting_risk": "high: public pass would not prove private robustness.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp143_task018_beicicc_repair_probe",
        "",
        "## Hypothesis",
        "",
        "task018はexp127でpublic-zeroと推定される。full local validationを通るbeicicc候補へ単独差し替えすれば、public LBを回収できる可能性がある。",
        "",
        "## Result",
        "",
        f"- status: `{result['status']}`",
        f"- candidate validation: `{result['candidate_validation']['status']}`",
        f"- candidate cost from manifest: `{result['candidate_cost_from_manifest']}`",
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
