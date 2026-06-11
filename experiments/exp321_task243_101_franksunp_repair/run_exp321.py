from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    infer_static_ok,
    load_neurogolf_utils,
    point,
    score_model,
    sha256,
    validate_examples,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp321_task243_101_franksunp_repair"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp297_exp262_skip_task048_336_fresh_candidates"
SOURCE_DIR = ROOT / "data" / "cache" / "public_artifacts" / "franksunp_blended_best"
BASE_PUBLIC_LB = 6008.96
TARGET_POINTS = {
    101: 13.904485714156337,
    243: 13.874532744845478,
}


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    with zipfile.ZipFile(BASE_EXP / "submission.zip") as zf:
        raws = {
            int(Path(name).stem.replace("task", "")): zf.read(name)
            for name in zf.namelist()
            if name.endswith(".onnx")
        }

    candidate_results = []
    accepted_raws: dict[int, bytes] = {}
    for task_id in sorted(TARGET_POINTS):
        raw = (SOURCE_DIR / f"task{task_id:03d}.onnx").read_bytes()
        sanitized_raw = None
        candidate_cost: int | str = ""
        candidate_points: float | str = ""
        validation_status = "not_run"
        status = "rejected"
        reason = ""
        candidate_sha = sha256(raw)
        try:
            model = onnx.load_model_from_string(raw)
            ok, reason = infer_static_ok(model)
            if ok:
                sanitized = utils.sanitize_model(model)
                if sanitized is None:
                    reason = "sanitize failed"
                else:
                    sanitized_raw = sanitized.SerializeToString()
                    candidate_sha = sha256(sanitized_raw)
                    ok, reason, passed, failed = validate_examples(utils, sanitized_raw, task_id, -1)
                    validation_status = f"{passed}_pass_{failed}_fail"
                    if ok:
                        memory, params, score_reason = score_model(
                            utils,
                            sanitized_raw,
                            task_id,
                            "franksunp_blended_best_publiczero_repair",
                            EXP_DIR,
                        )
                        if memory is None or params is None:
                            reason = score_reason
                        else:
                            candidate_cost = int(memory) + int(params)
                            candidate_points = point(candidate_cost)
                            status = "accepted_for_publiczero_repair"
                            reason = score_reason
                    else:
                        sanitized_raw = None
            else:
                validation_status = "not_run"
        except Exception as exc:
            reason = f"candidate gate failed: {str(exc)[:180]}"
        row = {
            "task_id": task_id,
            "source": "franksunp_blended_best",
            "source_path": str(SOURCE_DIR / f"task{task_id:03d}.onnx"),
            "source_sha256": sha256(raw),
            "baseline_cost": "",
            "baseline_points": "",
            "target_points_if_public_fixed": TARGET_POINTS[task_id],
            "candidate_cost": candidate_cost,
            "candidate_points": candidate_points,
            "candidate_status": status,
            "validation_status": validation_status,
            "reason": reason,
            "candidate_sha256": candidate_sha,
            "accepted_for_repair": sanitized_raw is not None and validation_status.endswith("_pass_0_fail"),
        }
        if row["accepted_for_repair"]:
            accepted_raws[task_id] = sanitized_raw
        candidate_results.append(row)

    for task_id, raw in accepted_raws.items():
        raws[task_id] = raw

    output_zip = EXP_DIR / "submission.zip"
    if len(accepted_raws) == len(TARGET_POINTS):
        write_zip(output_zip, raws)
        sanity = zip_sanity(output_zip)
        submission_decision = "submit_repair_after_sanity"
    else:
        sanity = None
        submission_decision = "no_submit_validation_failed"

    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-11",
        "status": "zip_ready" if sanity else "completed_no_submit",
        "purpose": (
            "Build a public-zero repair bundle for task101/task243 using exp320 top "
            "franksunp_blended_best candidates over exp297."
        ),
        "hypothesis": (
            "If task101/task243 are public-zero in exp297, replacing them with full-arc-valid "
            "franksunp_blended_best candidates should recover about +27.779 public LB."
        ),
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_public_lb": BASE_PUBLIC_LB,
        "targets": sorted(TARGET_POINTS),
        "candidate_results": candidate_results,
        "accepted_targets": sorted(accepted_raws),
        "expected_public_gain_if_all_fixed": sum(TARGET_POINTS.values()),
        "expected_public_lb_if_all_fixed": BASE_PUBLIC_LB + sum(TARGET_POINTS.values()),
        "zip_sanity": sanity,
        "outputs": {"submission_zip": str(output_zip.relative_to(ROOT))} if sanity else {},
        "submission_decision": submission_decision,
        "leakage_risk": (
            "medium-high: repair candidates come from public-code/raw blend artifacts. "
            "They are used only after full-arc validation, but private robustness remains uncertain."
        ),
        "overfitting_risk": (
            "medium: target selection used public diagnostic probes; acceptance is gated by full-arc "
            "validation and expected-LB accounting, not by adopting arbitrary public-score noise."
        ),
        "elapsed_s": round(time.time() - started, 3),
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {EXP_ID}",
        "",
        "## 目的",
        "",
        "task101/task243 の public-zero repair として、exp320 で top だった `franksunp_blended_best` 候補を exp297 に差し替える。",
        "",
        "## 結果",
        "",
        f"- status: `{result['status']}`",
        f"- targets: `{sorted(TARGET_POINTS)}`",
        f"- expected_public_gain_if_all_fixed: `{result['expected_public_gain_if_all_fixed']}`",
        f"- expected_public_lb_if_all_fixed: `{result['expected_public_lb_if_all_fixed']}`",
        "",
        "## Candidate Gate",
    ]
    for row in candidate_results:
        lines.append(
            f"- task{row['task_id']:03d}: validation `{row['validation_status']}`, "
            f"status `{row['candidate_status']}`, cost `{row['candidate_cost']}`, "
            f"accepted `{row['accepted_for_repair']}`"
        )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            result["submission_decision"],
            "",
            "## リスク",
            "",
            f"- leakage risk: {result['leakage_risk']}",
            f"- overfitting risk: {result['overfitting_risk']}",
        ]
    )
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
