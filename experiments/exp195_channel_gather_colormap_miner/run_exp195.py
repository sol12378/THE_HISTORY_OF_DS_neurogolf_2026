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

from experiments.phase1_rewrite_utils import (
    examples_for,
    grid_to_array,
    load_neurogolf_utils,
    load_task,
    score_model,
    validate_examples,
)


EXP_DIR = ROOT / "experiments" / "exp195_channel_gather_colormap_miner"
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


def infer_color_mapping(task_id: int) -> tuple[bool, list[int], str]:
    mapping: dict[int, int] = {}
    task = load_task(task_id)
    for ex_idx, ex in enumerate(examples_for(task, arc_gen_sample=-1)):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        if x.shape != y.shape:
            return False, [], f"shape mismatch example {ex_idx}"
        for src, dst in zip(x.ravel(), y.ravel()):
            s = int(src)
            d = int(dst)
            if s in mapping and mapping[s] != d:
                return False, [], f"conflict color {s}: {mapping[s]} vs {d} at example {ex_idx}"
            mapping[s] = d
    full = [mapping.get(i, i) for i in range(10)]
    if full == list(range(10)):
        return False, full, "identity mapping"
    # Gather(axis=1) needs inverse: output channel dst reads input channel src.
    inverse = list(range(10))
    seen_dst: set[int] = set()
    for src, dst in enumerate(full):
        if dst in seen_dst:
            return False, full, "non-injective mapping cannot be represented by single Gather"
        seen_dst.add(dst)
        inverse[dst] = src
    return True, inverse, "ok"


def build_gather_model(task_id: int, inverse: list[int]) -> bytes:
    indices = numpy_helper.from_array(np.asarray(inverse, dtype=np.int64), "indices")
    graph = helper.make_graph(
        [helper.make_node("Gather", ["input", "indices"], ["output"], axis=1)],
        f"task{task_id:03d}_channel_gather_colormap",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [indices],
    )
    model = helper.make_model(graph, producer_name="exp195_channel_gather_colormap", ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
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
        ok_map, inverse, map_reason = infer_color_mapping(task_id)
        row: dict[str, Any] = {
            "task_id": task_id,
            "mapping_status": "ok" if ok_map else "skip",
            "mapping_reason": map_reason,
            "inverse_indices": " ".join(map(str, inverse)) if inverse else "",
            "base_cost": base_costs.get(task_id, ""),
            "candidate_cost": "",
            "candidate_points": "",
            "validation_status": "",
            "status": "not_run",
            "reason": "",
        }
        if not ok_map:
            rows.append(row)
            continue
        raw = build_gather_model(task_id, inverse)
        cand_path = EXP_DIR / f"task{task_id:03d}_channel_gather.onnx"
        cand_path.write_bytes(raw)
        valid, val_reason, passed, failed = validate_examples(utils, raw, task_id, arc_gen_sample=-1)
        row["validation_status"] = f"{passed}_pass_{failed}_fail"
        if not valid:
            row["status"] = "rejected"
            row["reason"] = val_reason
            rows.append(row)
            continue
        memory, params, score_reason = score_model(utils, raw, task_id, "channel_gather_colormap", EXP_DIR)
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

    out_csv = EXP_DIR / "channel_gather_colormap_candidates.csv"
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
        "exp_id": "exp195_channel_gather_colormap_miner",
        "date": "2026-06-10",
        "status": "candidate_scan_complete",
        "purpose": "Instantiate the C-2 channel_gather_colormap archetype across all tasks and accept full-valid cost-improving candidates.",
        "base_exp": str(BASE_ZIP.parent.relative_to(ROOT)),
        "scanned_tasks": 400,
        "mapping_ok_count": sum(1 for r in rows if r["mapping_status"] == "ok"),
        "improved_count": len(improved),
        "improved_tasks": [int(r["task_id"]) for r in improved],
        "outputs": {"candidate_csv": str(out_csv.relative_to(ROOT)), "submission_zip": bundle_zip},
        "submission_decision": "submit_if_improved_and_low_risk" if improved else "no_submit_no_improved_candidates",
        "elapsed_s": round(time.time() - t0, 3),
        "leakage_risk": "low: input-only color permutation rule, no private labels.",
        "overfitting_risk": "low-to-medium: full arc-gen validation required, but public/private robustness still needs LB calibration if submitted.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp195_channel_gather_colormap_miner",
        "",
        "## 目的",
        "",
        "C-2 `channel_gather_colormap` archetypeを全taskへ適用し、full-validかつcost改善する1-node Gather候補を探す。",
        "",
        "## 結果",
        "",
        f"- mapping_ok_count: `{result['mapping_ok_count']}`",
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
