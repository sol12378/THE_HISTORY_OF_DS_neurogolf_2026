from __future__ import annotations

import csv
import json
import math
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

from experiments.phase1_rewrite_utils import examples_for, load_neurogolf_utils, load_task, one_hot_padded, score_model, validate_examples


EXP_DIR = ROOT / "experiments" / "exp197_one_conv_kernel_miner"
BASE_ZIP = ROOT / "experiments" / "exp178_task285_b035_repair_probe" / "submission.zip"
BASE_MANIFEST = ROOT / "experiments" / "exp002_public_blend_6500_fast" / "selected_manifest.csv"
COST_TARGETS = ROOT / "experiments" / "exp053_all_task_cost_250_600_inventory" / "task_cost_targets.csv"
MAX_TASKS = 20
KERNELS = [1]


def point(cost: int | float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def read_base_costs() -> dict[int, int]:
    costs: dict[int, int] = {}
    with BASE_MANIFEST.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                costs[int(row["task_id"])] = int(float(row["cost"]))
            except Exception:
                continue
    return costs


def target_tasks() -> list[int]:
    rows = []
    with COST_TARGETS.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append((float(row["gain_to_600"]), int(row["task_id"])))
            except Exception:
                continue
    rows.sort(reverse=True)
    return [tid for _, tid in rows[:MAX_TASKS]]


def patch_features(x: np.ndarray, r: int, c: int, k: int) -> np.ndarray:
    pad = k // 2
    padded = np.pad(x[0], ((0, 0), (pad, pad), (pad, pad)), mode="constant")
    patch = padded[:, r : r + k, c : c + k].reshape(-1)
    return np.concatenate([patch, np.asarray([1.0], dtype=np.float32)])


def fit_conv(task_id: int, k: int) -> tuple[bool, np.ndarray | None, np.ndarray | None, str]:
    examples = examples_for(load_task(task_id), arc_gen_sample=-1)
    feat_n = 10 * k * k + 1
    xtx = np.zeros((feat_n, feat_n), dtype=np.float64)
    xty = np.zeros((feat_n, 10), dtype=np.float64)
    for ex in examples:
        x = one_hot_padded(ex["input"]).astype(np.float32)
        y = one_hot_padded(ex["output"]).astype(np.float32)[0]
        for r in range(30):
            for c in range(30):
                feat = patch_features(x, r, c, k).astype(np.float64)
                xtx += np.outer(feat, feat)
                xty += feat[:, None] * y[:, r, c][None, :]
    try:
        coeff = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    except Exception as exc:
        return False, None, None, f"lstsq failed: {exc}"
    weights = coeff[:-1].T.reshape(10, 10, k, k).astype(np.float32)
    bias = coeff[-1].astype(np.float32)
    # Strict exactness precheck on all examples before paying ORT validation.
    for ex_idx, ex in enumerate(examples):
        x = one_hot_padded(ex["input"]).astype(np.float32)
        y = one_hot_padded(ex["output"]).astype(np.float32)
        pred = np.zeros_like(y)
        for r in range(30):
            for c in range(30):
                feat = patch_features(x, r, c, k)[:-1].reshape(10, k, k)
                pred[0, :, r, c] = (weights * feat[None, :, :, :]).sum(axis=(1, 2, 3)) + bias
        if not np.allclose(pred, y, atol=1e-4):
            return False, None, None, f"not exact example {ex_idx}"
    return True, weights, bias, "ok"


def build_conv_model(task_id: int, k: int, weights: np.ndarray, bias: np.ndarray) -> bytes:
    w = numpy_helper.from_array(weights.astype(np.float32), "weight")
    b = numpy_helper.from_array(bias.astype(np.float32), "bias")
    pad = k // 2
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "weight", "bias"], ["output"], pads=[pad, pad, pad, pad])],
        f"task{task_id:03d}_one_conv_k{k}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [w, b],
    )
    model = helper.make_model(graph, producer_name="exp197_one_conv_kernel_miner", ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_costs = read_base_costs()
    rows: list[dict[str, Any]] = []
    accepted_raws: dict[int, bytes] = {}
    tasks = target_tasks()

    for task_id in tasks:
        base_cost = base_costs.get(task_id)
        for k in KERNELS:
            row: dict[str, Any] = {
                "task_id": task_id,
                "kernel": k,
                "base_cost": base_cost if base_cost is not None else "",
                "fit_status": "",
                "fit_reason": "",
                "validation_status": "",
                "candidate_cost": "",
                "candidate_points": "",
                "status": "not_run",
                "reason": "",
            }
            fit_ok, weights, bias, fit_reason = fit_conv(task_id, k)
            row["fit_status"] = "ok" if fit_ok else "fail"
            row["fit_reason"] = fit_reason
            if not fit_ok or weights is None or bias is None:
                rows.append(row)
                continue
            raw = build_conv_model(task_id, k, weights, bias)
            (EXP_DIR / f"task{task_id:03d}_one_conv_k{k}.onnx").write_bytes(raw)
            valid, val_reason, passed, failed = validate_examples(utils, raw, task_id, arc_gen_sample=-1)
            row["validation_status"] = f"{passed}_pass_{failed}_fail"
            if not valid:
                row["status"] = "rejected"
                row["reason"] = val_reason
                rows.append(row)
                continue
            memory, params, score_reason = score_model(utils, raw, task_id, f"one_conv_k{k}", EXP_DIR)
            if memory is None or params is None:
                row["status"] = "rejected"
                row["reason"] = score_reason
                rows.append(row)
                continue
            cand_cost = int(memory + params)
            row["candidate_cost"] = cand_cost
            row["candidate_points"] = point(cand_cost)
            if base_cost is not None and cand_cost < base_cost:
                row["status"] = "improved"
                row["reason"] = "candidate cost lower than base manifest"
                if task_id not in accepted_raws:
                    accepted_raws[task_id] = raw
            else:
                row["status"] = "no_cost_gain"
                row["reason"] = "candidate cost is not lower than base manifest"
            rows.append(row)

    out_csv = EXP_DIR / "one_conv_kernel_candidates.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    bundle_zip = ""
    if accepted_raws:
        with zipfile.ZipFile(BASE_ZIP) as zin:
            raws = {int(Path(name).stem.replace("task", "")): zin.read(name) for name in zin.namelist()}
        raws.update(accepted_raws)
        out_zip = EXP_DIR / "submission.zip"
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for tid, raw in sorted(raws.items()):
                zout.writestr(f"task{tid:03d}.onnx", raw)
        bundle_zip = str(out_zip.relative_to(ROOT))

    improved = [r for r in rows if r["status"] == "improved"]
    result = {
        "exp_id": "exp197_one_conv_kernel_miner",
        "date": "2026-06-10",
        "status": "candidate_scan_complete",
        "purpose": "Instantiate the C-2 one_node_conv_kernel archetype on top gain-to-600 tasks. This bounded retry uses only 1x1 Conv after the broader 1x1/3x3 scan timed out.",
        "base_exp": str(BASE_ZIP.parent.relative_to(ROOT)),
        "scanned_tasks": len(tasks),
        "fit_ok_count": sum(1 for r in rows if r["fit_status"] == "ok"),
        "improved_count": len(improved),
        "improved_tasks": sorted({int(r["task_id"]) for r in improved}),
        "outputs": {"candidate_csv": str(out_csv.relative_to(ROOT)), "submission_zip": bundle_zip},
        "submission_decision": "submit_if_improved_and_low_risk" if improved else "no_submit_no_improved_candidates",
        "elapsed_s": round(time.time() - t0, 3),
        "leakage_risk": "medium: linear fit uses all available arc-gen examples; no private labels, but rule may be example-fit.",
        "overfitting_risk": "medium-to-high unless pattern is simple and LB-calibrated by small submission.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp197_one_conv_kernel_miner",
        "",
        "## 目的",
        "",
        "C-2 `one_node_conv_kernel` archetypeをgain_to_600上位taskへ適用し、1x1/3x3 Convだけでfull-validかつcost改善する候補を探す。",
        "",
        "## 結果",
        "",
        f"- scanned_tasks: `{result['scanned_tasks']}`",
        f"- fit_ok_count: `{result['fit_ok_count']}`",
        f"- improved_count: `{result['improved_count']}`",
        f"- improved_tasks: `{result['improved_tasks']}`",
        f"- submission_decision: `{result['submission_decision']}`",
        "",
        "## リスク",
        "",
        str(result["leakage_risk"]),
        str(result["overfitting_risk"]),
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
