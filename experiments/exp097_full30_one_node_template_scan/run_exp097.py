from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp097_full30_one_node_template_scan"
EXP_DIR = ROOT / "experiments" / EXP_ID


def transforms(x: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "identity": x,
        "flip_lr": x[:, ::-1],
        "flip_ud": x[::-1, :],
        "rot180": x[::-1, ::-1],
        "transpose": x.T,
        "rot90_ccw": np.rot90(x, 1),
        "rot90_cw": np.rot90(x, -1),
    }


def fit_map(pairs: list[tuple[np.ndarray, np.ndarray]], base_name: str) -> tuple[bool, dict[int, int]]:
    mapping: dict[int, int] = {}
    for x, y in pairs:
        tx = transforms(x)[base_name]
        if tx.shape != y.shape:
            return False, {}
        for a, b in zip(tx.ravel(), y.ravel()):
            ai, bi = int(a), int(b)
            if ai in mapping and mapping[ai] != bi:
                return False, {}
            mapping[ai] = bi
    for c in range(10):
        mapping.setdefault(c, c)
    return True, mapping


def apply_map(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    lut = np.asarray([mapping.get(c, c) for c in range(10)], dtype=np.int64)
    return lut[arr]


def all_30(examples: list[dict[str, Any]]) -> bool:
    for ex in examples:
        for mode in ["input", "output"]:
            g = ex[mode]
            if len(g) != 30 or len(g[0]) != 30:
                return False
    return True


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base_tasks = load_base_tasks()
    transform_names = ["identity", "flip_lr", "flip_ud", "rot180", "transpose", "rot90_ccw", "rot90_cw"]
    rows: list[dict[str, Any]] = []
    full30_tasks: list[int] = []
    hits: list[dict[str, Any]] = []
    for task_id in range(1, 401):
        task = load_task(task_id)
        examples = task["train"] + task["test"] + task["arc-gen"]
        if not all_30(examples):
            continue
        full30_tasks.append(task_id)
        pairs = [(grid_to_array(ex["input"]), grid_to_array(ex["output"])) for ex in examples]
        for name in transform_names:
            passed = 0
            failed = 0
            for x, y in pairs:
                pred = transforms(x)[name]
                if np.array_equal(pred, y):
                    passed += 1
                else:
                    failed += 1
            row = {
                "task_id": task_id,
                "variant": name,
                "kind": "geometry",
                "passed": passed,
                "failed": failed,
                "baseline_cost": base_tasks[task_id].cost,
                "baseline_points": base_tasks[task_id].points,
                "mapping": "",
            }
            rows.append(row)
            if failed == 0:
                hits.append(row)
            ok, mapping = fit_map(pairs, name)
            if ok:
                p2 = 0
                f2 = 0
                for x, y in pairs:
                    pred = apply_map(transforms(x)[name], mapping)
                    if np.array_equal(pred, y):
                        p2 += 1
                    else:
                        f2 += 1
                row2 = {
                    "task_id": task_id,
                    "variant": f"{name}_channel_recolor",
                    "kind": "geometry_recolor",
                    "passed": p2,
                    "failed": f2,
                    "baseline_cost": base_tasks[task_id].cost,
                    "baseline_points": base_tasks[task_id].points,
                    "mapping": json.dumps(mapping, sort_keys=True),
                }
                rows.append(row2)
                if f2 == 0:
                    hits.append(row2)
    rows.sort(key=lambda r: (int(r["failed"]), -int(r["baseline_cost"]), int(r["task_id"]), r["variant"]))
    hits.sort(key=lambda r: (-int(r["baseline_cost"]), int(r["task_id"]), r["variant"]))
    fields = ["task_id", "variant", "kind", "passed", "failed", "baseline_cost", "baseline_points", "mapping"]
    with (EXP_DIR / "full30_one_node_scan.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (EXP_DIR / "full30_one_node_hits.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(hits)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "scan_ready",
        "full30_task_count": len(full30_tasks),
        "full30_tasks": full30_tasks,
        "candidate_rows": len(rows),
        "hit_count": len(hits),
        "hit_kind_counts": dict(Counter(r["kind"] for r in hits)),
        "top_hits": hits[:30],
        "decision": "Only full30 hits can safely use full-grid one-node Slice/Transpose without padding mismatch. Emit replacements for any hit with baseline cost above official one-node cost.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: scan only",
        "leakage_risk": "low.",
        "overfitting_risk": "low for geometry; medium for color maps.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`exp096` で可変サイズgridのpadding mismatchが判明したため、全例が30x30のtaskだけに限定して1ノード公式templateをscanする。

## 結果

- full30 task count: `{len(full30_tasks)}`
- full hits: `{len(hits)}`

## Decision

full30 hitのみ、full-grid one-node Slice/Transpose/Gather replacementへ進める。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
