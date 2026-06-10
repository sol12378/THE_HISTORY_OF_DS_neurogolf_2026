from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import load_neurogolf_utils, validate_examples, zip_sanity


EXP_DIR = ROOT / "experiments" / "exp154_next4h_failure_bisection_probe"
BASE_EXP = ROOT / "experiments" / "exp127_focused_surgery_seventh_pass_limited"
BASE_ZIP = BASE_EXP / "submission.zip"
TARGETS = [33, 87, 276, 30]
TARGET_POINTS = {
    33: 14.081988539178049,
    87: 19.09191706183107,
    276: 22.697414907005953,
    30: 15.607504751786962,
}


def make_zero_stub() -> bytes:
    zero = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero")
    graph = helper.make_graph(
        [helper.make_node("Mul", ["input", "zero"], ["output"])],
        "exp154_zero_fail_stub",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [zero],
    )
    model = helper.make_model(graph, producer_name="exp154_zero_fail_stub", ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
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
        target_validation.append({"task_id": tid, "ok": ok, "reason": reason, "passed": passed, "failed": failed, "status": f"{passed}_pass_{failed}_fail"})

    expected_drop_if_all_alive = sum(TARGET_POINTS.values())
    result: dict[str, Any] = {
        "exp_id": "exp154_next4h_failure_bisection_probe",
        "date": "2026-06-10",
        "status": "probe_zip_ready",
        "purpose": "Continue public-zero bisection after exp153: intentionally fail next remaining frequent subset tasks 033/087/276/030.",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_public_lb": 5930.55,
        "current_public_best_lb": 5956.28,
        "targets": TARGETS,
        "target_points": TARGET_POINTS,
        "expected_drop_if_all_alive": expected_drop_if_all_alive,
        "expected_lb_if_all_alive_from_exp127": 5930.55 - expected_drop_if_all_alive,
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
        "# exp154_next4h_failure_bisection_probe",
        "",
        "## Hypothesis",
        "",
        "exp153後の未判定subset頻出上位 `033/087/276/030` のpublic scoring状態をfail-stub probeで測る。",
        "",
        "## Result",
        "",
        f"- status: `{result['status']}`",
        f"- targets: `{result['targets']}`",
        f"- expected_drop_if_all_alive: `{result['expected_drop_if_all_alive']}`",
        f"- expected_lb_if_all_alive_from_exp127: `{result['expected_lb_if_all_alive_from_exp127']}`",
        f"- zip sha256: `{result['zip_sanity']['sha256']}`",
        "",
        "## Target Validation",
        "",
    ]
    for row in result["target_validation"]:
        lines.append(f"- task{int(row['task_id']):03d}: `{row['status']}`, reason `{row['reason']}`")
    lines.extend(["", "## Decision", "", str(result["submission_decision"]), "", "## Leakage / Overfitting Risk", "", str(result["leakage_risk"]), str(result["overfitting_risk"])])
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
