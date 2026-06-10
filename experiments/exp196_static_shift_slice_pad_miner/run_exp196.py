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

from experiments.phase1_rewrite_utils import examples_for, grid_to_array, load_neurogolf_utils, load_task, score_model, validate_examples


EXP_DIR = ROOT / "experiments" / "exp196_static_shift_slice_pad_miner"
BASE_ZIP = ROOT / "experiments" / "exp178_task285_b035_repair_probe" / "submission.zip"
BASE_MANIFEST = ROOT / "experiments" / "exp002_public_blend_6500_fast" / "selected_manifest.csv"


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


def pad_grid(grid: list[list[int]], size: int = 30) -> np.ndarray:
    arr = np.zeros((size, size), dtype=np.int64)
    g = grid_to_array(grid)
    h = min(size, g.shape[0])
    w = min(size, g.shape[1])
    arr[:h, :w] = g[:h, :w]
    return arr


def shift_grid(x: np.ndarray, dr: int, dc: int) -> np.ndarray:
    y = np.zeros_like(x)
    src_r0 = max(0, -dr)
    src_r1 = min(30, 30 - dr)
    src_c0 = max(0, -dc)
    src_c1 = min(30, 30 - dc)
    dst_r0 = src_r0 + dr
    dst_r1 = src_r1 + dr
    dst_c0 = src_c0 + dc
    dst_c1 = src_c1 + dc
    if src_r0 < src_r1 and src_c0 < src_c1:
        y[dst_r0:dst_r1, dst_c0:dst_c1] = x[src_r0:src_r1, src_c0:src_c1]
    return y


def infer_shift(task_id: int) -> tuple[bool, int, int, str]:
    examples = examples_for(load_task(task_id), arc_gen_sample=-1)
    for dr in range(-29, 30):
        for dc in range(-29, 30):
            ok = True
            for ex_idx, ex in enumerate(examples):
                x = pad_grid(ex["input"])
                y = pad_grid(ex["output"])
                if not np.array_equal(shift_grid(x, dr, dc), y):
                    ok = False
                    break
            if ok and (dr != 0 or dc != 0):
                return True, dr, dc, "ok"
    return False, 0, 0, "no fixed shift"


def build_shift_model(task_id: int, dr: int, dc: int) -> bytes:
    src_r0 = max(0, -dr)
    src_r1 = min(30, 30 - dr)
    src_c0 = max(0, -dc)
    src_c1 = min(30, 30 - dc)
    top = max(0, dr)
    bottom = max(0, -dr)
    left = max(0, dc)
    right = max(0, -dc)
    starts = numpy_helper.from_array(np.asarray([src_r0, src_c0], dtype=np.int64), "starts")
    ends = numpy_helper.from_array(np.asarray([src_r1, src_c1], dtype=np.int64), "ends")
    axes = numpy_helper.from_array(np.asarray([2, 3], dtype=np.int64), "axes")
    pads = numpy_helper.from_array(np.asarray([0, 0, top, left, 0, 0, bottom, right], dtype=np.int64), "pads")
    zero = numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "zero")
    graph = helper.make_graph(
        [
            helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["crop"]),
            helper.make_node("Pad", ["crop", "pads", "zero"], ["output"], mode="constant"),
        ],
        f"task{task_id:03d}_static_shift_slice_pad",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [starts, ends, axes, pads, zero],
    )
    model = helper.make_model(graph, producer_name="exp196_static_shift_slice_pad", ir_version=10, opset_imports=[helper.make_opsetid("", 11)])
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_costs = read_base_costs()
    rows: list[dict[str, Any]] = []
    accepted_raws: dict[int, bytes] = {}

    for task_id in range(1, 401):
        ok_shift, dr, dc, shift_reason = infer_shift(task_id)
        row: dict[str, Any] = {
            "task_id": task_id,
            "shift_status": "ok" if ok_shift else "skip",
            "shift_reason": shift_reason,
            "dr": dr if ok_shift else "",
            "dc": dc if ok_shift else "",
            "base_cost": base_costs.get(task_id, ""),
            "candidate_cost": "",
            "candidate_points": "",
            "validation_status": "",
            "status": "not_run",
            "reason": "",
        }
        if not ok_shift:
            rows.append(row)
            continue
        raw = build_shift_model(task_id, dr, dc)
        (EXP_DIR / f"task{task_id:03d}_shift_{dr}_{dc}.onnx").write_bytes(raw)
        valid, val_reason, passed, failed = validate_examples(utils, raw, task_id, arc_gen_sample=-1)
        row["validation_status"] = f"{passed}_pass_{failed}_fail"
        if not valid:
            row["status"] = "rejected"
            row["reason"] = val_reason
            rows.append(row)
            continue
        memory, params, score_reason = score_model(utils, raw, task_id, "static_shift_slice_pad", EXP_DIR)
        if memory is None or params is None:
            row["status"] = "rejected"
            row["reason"] = score_reason
            rows.append(row)
            continue
        cand_cost = int(memory + params)
        row["candidate_cost"] = cand_cost
        row["candidate_points"] = point(cand_cost)
        base_cost = base_costs.get(task_id)
        if base_cost is not None and cand_cost < base_cost:
            row["status"] = "improved"
            row["reason"] = "candidate cost lower than base manifest"
            accepted_raws[task_id] = raw
        else:
            row["status"] = "no_cost_gain"
            row["reason"] = "candidate cost is not lower than base manifest"
        rows.append(row)

    out_csv = EXP_DIR / "static_shift_slice_pad_candidates.csv"
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
        "exp_id": "exp196_static_shift_slice_pad_miner",
        "date": "2026-06-10",
        "status": "candidate_scan_complete",
        "purpose": "Instantiate the C-2 static_slice_pad archetype as fixed translation with zero padding across all tasks.",
        "base_exp": str(BASE_ZIP.parent.relative_to(ROOT)),
        "scanned_tasks": 400,
        "shift_ok_count": sum(1 for r in rows if r["shift_status"] == "ok"),
        "improved_count": len(improved),
        "improved_tasks": [int(r["task_id"]) for r in improved],
        "outputs": {"candidate_csv": str(out_csv.relative_to(ROOT)), "submission_zip": bundle_zip},
        "submission_decision": "submit_if_improved_and_low_risk" if improved else "no_submit_no_improved_candidates",
        "elapsed_s": round(time.time() - t0, 3),
        "leakage_risk": "low: input-only fixed shift rule, no private labels.",
        "overfitting_risk": "low-to-medium: full arc-gen validation required, but private robustness still needs LB calibration if submitted.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp196_static_shift_slice_pad_miner",
        "",
        "## 目的",
        "",
        "C-2 `static_slice_pad` archetypeの最小形として、固定平行移動+zero paddingを全taskへ適用する。",
        "",
        "## 結果",
        "",
        f"- shift_ok_count: `{result['shift_ok_count']}`",
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
