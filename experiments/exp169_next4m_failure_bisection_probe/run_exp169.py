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


EXP_DIR = ROOT / "experiments" / "exp169_next4m_failure_bisection_probe"
BASE_EXP = ROOT / "experiments" / "exp166_task025_line_projection_publiczero_repair"
BASE_ZIP = BASE_EXP / "submission.zip"
BASE_PUBLIC_LB = 5968.18
TARGETS = [133, 396, 158, 47]
TARGET_POINTS = {
    133: 12.824793861384533,
    396: 13.126683504302262,
    158: 12.883983361645681,
    47: 16.985664262700574,
}


def make_zero_stub() -> bytes:
    zero = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero")
    graph = helper.make_graph(
        [helper.make_node("Mul", ["input", "zero"], ["output"])],
        "exp169_zero_fail_stub",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [zero],
    )
    model = helper.make_model(graph, producer_name="exp169_zero_fail_stub", ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    stub = make_zero_stub()
    raws: dict[int, bytes] = {}
    with zipfile.ZipFile(BASE_ZIP) as zin:
        for name in zin.namelist():
            tid = int(Path(name).stem.replace("task", ""))
            raws[tid] = stub if tid in TARGETS else zin.read(name)

    out_zip = EXP_DIR / "submission.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for tid, raw in sorted(raws.items()):
            zout.writestr(f"task{tid:03d}.onnx", raw)

    target_validation = []
    for tid in TARGETS:
        ok, reason, passed, failed = validate_examples(utils, stub, tid, arc_gen_sample=-1)
        target_validation.append({"task_id": tid, "ok": ok, "reason": reason, "passed": passed, "failed": failed, "status": f"{passed}_pass_{failed}_fail"})

    expected_drop_if_all_alive = sum(TARGET_POINTS.values())
    result: dict[str, Any] = {
        "exp_id": "exp169_next4m_failure_bisection_probe",
        "date": "2026-06-10",
        "status": "probe_zip_ready",
        "purpose": "Probe next remaining subset-frequency public-zero suspects 133/396/158/047 after exp167 all-alive.",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_public_lb": BASE_PUBLIC_LB,
        "targets": TARGETS,
        "target_points": TARGET_POINTS,
        "expected_drop_if_all_alive": expected_drop_if_all_alive,
        "expected_lb_if_all_alive": BASE_PUBLIC_LB - expected_drop_if_all_alive,
        "target_validation": target_validation,
        "zip_sanity": zip_sanity(out_zip),
        "outputs": {"submission_zip": str(out_zip.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "submission_decision": "submit_bisection_probe",
        "leakage_risk": "low: deliberate fail-stub probe; no private labels used.",
        "overfitting_risk": "medium: consumes a submission and only measures public scoring behavior for a group.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp169_next4m_failure_bisection_probe",
        "",
        "## 目的",
        "",
        "exp166 current bestをベースに、残りsubset頻度候補の `133/396/158/047` をfail-stub化してpublic-zero有無を測る。",
        "",
        "## 結果",
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
    lines.extend(["", "## 判断", "", str(result["submission_decision"]), "", "## リスク", "", str(result["leakage_risk"]), str(result["overfitting_risk"])])
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
