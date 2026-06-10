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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import load_neurogolf_utils, validate_examples, zip_sanity


def make_zero_stub(name: str) -> bytes:
    zero = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero")
    graph = helper.make_graph(
        [helper.make_node("Mul", ["input", "zero"], ["output"])],
        f"{name}_zero_fail_stub",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [zero],
    )
    model = helper.make_model(graph, producer_name=f"{name}_zero_fail_stub", ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def build_probe(
    *,
    exp_dir: Path,
    exp_id: str,
    purpose: str,
    base_exp: Path,
    base_public_lb: float,
    target_points: dict[int, float],
    submission_decision: str,
    overfitting_risk: str,
) -> dict[str, Any]:
    t0 = time.time()
    exp_dir.mkdir(parents=True, exist_ok=True)
    base_zip = base_exp / "submission.zip"
    utils = load_neurogolf_utils()
    targets = list(target_points)
    stub = make_zero_stub(exp_id)
    raws: dict[int, bytes] = {}
    with zipfile.ZipFile(base_zip) as zin:
        for name in zin.namelist():
            tid = int(Path(name).stem.replace("task", ""))
            raws[tid] = stub if tid in target_points else zin.read(name)

    out_zip = exp_dir / "submission.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for tid, raw in sorted(raws.items()):
            zout.writestr(f"task{tid:03d}.onnx", raw)

    target_validation = []
    for tid in targets:
        ok, reason, passed, failed = validate_examples(utils, stub, tid, arc_gen_sample=-1)
        target_validation.append({"task_id": tid, "ok": ok, "reason": reason, "passed": passed, "failed": failed, "status": f"{passed}_pass_{failed}_fail"})

    expected_drop_if_all_alive = sum(target_points.values())
    result: dict[str, Any] = {
        "exp_id": exp_id,
        "date": "2026-06-10",
        "status": "probe_zip_ready",
        "purpose": purpose,
        "base_exp": str(base_exp.relative_to(ROOT)),
        "base_public_lb": base_public_lb,
        "targets": targets,
        "target_points": target_points,
        "expected_drop_if_all_alive": expected_drop_if_all_alive,
        "expected_lb_if_all_alive": base_public_lb - expected_drop_if_all_alive,
        "target_validation": target_validation,
        "zip_sanity": zip_sanity(out_zip),
        "outputs": {"submission_zip": str(out_zip.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "submission_decision": submission_decision,
        "leakage_risk": "low: deliberate fail-stub probe; no private labels used.",
        "overfitting_risk": overfitting_risk,
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(exp_dir, result)
    return result


def write_notes(exp_dir: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['exp_id']}",
        "",
        "## 目的",
        "",
        str(result["purpose"]),
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
    (exp_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
