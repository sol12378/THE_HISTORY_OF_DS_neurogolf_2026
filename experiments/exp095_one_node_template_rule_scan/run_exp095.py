from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from datetime import date
from itertools import permutations
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp095_one_node_template_rule_scan"
EXP_DIR = ROOT / "experiments" / EXP_ID


def transforms(x: np.ndarray) -> dict[str, np.ndarray]:
    out = {
        "identity": x,
        "flip_lr": x[:, ::-1],
        "flip_ud": x[::-1, :],
        "rot180": x[::-1, ::-1],
        "transpose": x.T,
    }
    if x.shape[0] == x.shape[1]:
        out["rot90_ccw_py"] = np.rot90(x, 1)
        out["rot90_cw_py"] = np.rot90(x, -1)
    return out


def fit_color_map(pairs: list[tuple[np.ndarray, np.ndarray]], base_transform: str) -> tuple[bool, dict[int, int]]:
    mapping: dict[int, int] = {}
    for x, y in pairs:
        tx = transforms(x).get(base_transform)
        if tx is None or tx.shape != y.shape:
            return False, {}
        for a, b in zip(tx.ravel(), y.ravel()):
            ai, bi = int(a), int(b)
            if ai in mapping and mapping[ai] != bi:
                return False, {}
            mapping[ai] = bi
    for c in range(10):
        mapping.setdefault(c, c)
    return True, mapping


def apply_map(x: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    lut = np.asarray([mapping.get(c, c) for c in range(10)], dtype=np.int64)
    return lut[x]


def evaluate_variant(examples: list[dict[str, Any]], variant: str, mapping: dict[int, int] | None = None) -> tuple[int, int]:
    passed = 0
    failed = 0
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        tx = transforms(x).get(variant)
        if tx is None:
            failed += 1
            continue
        pred = apply_map(tx, mapping) if mapping is not None else tx
        if np.array_equal(pred, y):
            passed += 1
        else:
            failed += 1
    return passed, failed


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base_tasks = load_base_tasks()
    rows: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    transform_names = ["identity", "flip_lr", "flip_ud", "rot180", "transpose", "rot90_ccw_py", "rot90_cw_py"]
    for task_id in range(1, 401):
        task = load_task(task_id)
        examples = task["train"] + task["test"] + task["arc-gen"]
        pairs = [(grid_to_array(ex["input"]), grid_to_array(ex["output"])) for ex in examples]
        for name in transform_names:
            p, f = evaluate_variant(examples, name)
            row = {
                "task_id": task_id,
                "variant": name,
                "kind": "geometry",
                "passed": p,
                "failed": f,
                "baseline_cost": base_tasks[task_id].cost,
                "baseline_points": base_tasks[task_id].points,
                "mapping": "",
            }
            if f == 0:
                hits.append(row)
            rows.append(row)
            ok_map, mapping = fit_color_map(pairs, name)
            if ok_map:
                p2, f2 = evaluate_variant(examples, name, mapping)
                map_row = {
                    "task_id": task_id,
                    "variant": name + "_channel_recolor",
                    "kind": "geometry_recolor",
                    "passed": p2,
                    "failed": f2,
                    "baseline_cost": base_tasks[task_id].cost,
                    "baseline_points": base_tasks[task_id].points,
                    "mapping": json.dumps(mapping, sort_keys=True),
                }
                if f2 == 0:
                    hits.append(map_row)
                rows.append(map_row)
    rows.sort(key=lambda r: (int(r["failed"]), -int(r["baseline_cost"]), int(r["task_id"]), r["variant"]))
    hits.sort(key=lambda r: (-int(r["baseline_cost"]), int(r["task_id"]), r["variant"]))
    fields = ["task_id", "variant", "kind", "passed", "failed", "baseline_cost", "baseline_points", "mapping"]
    with (EXP_DIR / "one_node_template_scan.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (EXP_DIR / "one_node_template_hits.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(hits)
    route_counts = Counter(row["kind"] for row in hits)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "scan_ready",
        "hypothesis": "zip-derived official one-node templates can find full-pass tasks that can be replaced by near-zero-cost ONNX.",
        "task_count": 400,
        "candidate_rows": len(rows),
        "hit_count": len(hits),
        "hit_kind_counts": dict(route_counts),
        "top_hits": hits[:30],
        "decision": "For full hits with baseline cost > one-node official cost, emit official one-node ONNX replacements and validate as a submit-safe delta.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: scan only",
        "leakage_risk": "low: uses exact deterministic transforms; no hidden labels.",
        "overfitting_risk": "low for pure geometry; medium for color map if train/test/arc-gen all pass but color mapping is task-specific.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`exp094` で公式one-hotの1ノード幾何/色channel templateが極小costと分かったため、全400 taskへ `identity/flip/rot180/transpose/channel recolor` 系をPython ruleとしてscanする。

## 結果

- candidate rows: `{len(rows)}`
- full hits: `{len(hits)}`

## Decision

full hitがあり、baseline costより十分低ければ公式one-hot ONNX replacementを生成してfull validationする。

## Risk

- leakage risk: low。
- overfitting risk: pure geometryはlow。color mapはtask-specificなので、全arc-gen passでもLB較正対象。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
