from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import load_neurogolf_utils, validate_examples, zip_sanity


EXP_DIR = ROOT / "experiments" / "exp139_top4_failure_bisection_probe"
BASE_EXP = ROOT / "experiments" / "exp127_focused_surgery_seventh_pass_limited"
BASE_ZIP = BASE_EXP / "submission.zip"
TARGETS = [13, 2, 29, 9]


def make_zero_stub() -> bytes:
    zero = numpy_helper.from_array(__import__("numpy").array([0.0], dtype="float32"), "zero")
    graph = helper.make_graph(
        [helper.make_node("Mul", ["input", "zero"], ["output"])],
        "exp139_zero_fail_stub",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [zero],
    )
    model = helper.make_model(
        graph,
        producer_name="exp139_zero_fail_stub",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    stub = make_zero_stub()
    raws: dict[int, bytes] = {}
    with zipfile.ZipFile(BASE_ZIP) as zf:
        for name in zf.namelist():
            tid = int(Path(name).stem.replace("task", ""))
            raws[tid] = stub if tid in TARGETS else zf.read(name)

    out_zip = EXP_DIR / "submission.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for tid, raw in sorted(raws.items()):
            zf.writestr(f"task{tid:03d}.onnx", raw)

    target_validation = []
    for tid in TARGETS:
        ok, reason, passed, failed = validate_examples(utils, stub, tid, arc_gen_sample=-1)
        target_validation.append(
            {
                "task_id": tid,
                "ok": ok,
                "reason": reason,
                "passed": passed,
                "failed": failed,
                "status": f"{passed}_pass_{failed}_fail",
            }
        )

    base_result = json.loads((BASE_EXP / "result.json").read_text(encoding="utf-8"))
    target_points = {
        13: 13.3310798421544,
        2: 13.514517676859928,
        29: 13.516328283804857,
        9: 13.69482467517415,
    }
    expected_drop_if_all_alive = sum(target_points.values())
    result: dict[str, Any] = {
        "exp_id": "exp139_top4_failure_bisection_probe",
        "date": "2026-06-10",
        "status": "probe_zip_ready",
        "purpose": "A-3 bisection probe: intentionally fail top4 private-failure consensus tasks to measure whether they currently contribute to public LB.",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_public_lb": 5930.55,
        "base_local_estimate": base_result["new_local_estimate"],
        "targets": TARGETS,
        "target_points": target_points,
        "expected_drop_if_all_alive": expected_drop_if_all_alive,
        "expected_lb_if_all_alive": 5930.55 - expected_drop_if_all_alive,
        "interpretation_rule": "observed_drop close to target point sum means tasks were public-scoring alive; near-zero drop means tasks are already failing public scoring; partial drop identifies live/failing mix.",
        "target_validation": target_validation,
        "zip_sanity": zip_sanity(out_zip),
        "outputs": {"submission_zip": str(out_zip.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "submission_decision": "submit_probe_after_sanity",
        "leakage_risk": "low: deliberate fail-stub probe; no private labels used.",
        "overfitting_risk": "medium: consumes a submission and only measures public scoring behavior for a group.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp139_top4_failure_bisection_probe",
        "",
        "## Hypothesis",
        "",
        "task013/002/029/009 が -352 gap のprivate/public scoring failure候補なら、これらを意図的にfailさせたprobeのLB dropから、現public scoringで既に死んでいるかを判別できる。",
        "",
        "## Result",
        "",
        f"- status: `{result['status']}`",
        f"- targets: `{result['targets']}`",
        f"- expected_drop_if_all_alive: `{result['expected_drop_if_all_alive']}`",
        f"- expected_lb_if_all_alive: `{result['expected_lb_if_all_alive']}`",
        f"- zip sha256: `{result['zip_sanity']['sha256']}`",
        "",
        "## Target Validation",
        "",
    ]
    for row in result["target_validation"]:
        lines.append(f"- task{int(row['task_id']):03d}: `{row['status']}`, reason `{row['reason']}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "sanityが通ればKaggle probeとして提出し、observed LB dropを記録する。",
            "",
            "## Leakage / Overfitting Risk",
            "",
            str(result["leakage_risk"]),
            str(result["overfitting_risk"]),
        ]
    )
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
