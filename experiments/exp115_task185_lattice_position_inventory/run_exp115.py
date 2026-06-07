from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp115_task185_lattice_position_inventory"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185
CURRENT_LOCAL_ESTIMATE = 6282.93615685804
CAMPAIGN_INDEX = 17


def grid_color(arr: np.ndarray) -> int:
    vals = [int(v) for v in arr.ravel() if int(v) != 0]
    return Counter(vals).most_common(1)[0][0]


def grid_lines(arr: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    h, w = arr.shape
    rows = [r for r in range(h) if int((arr[r, :] == bg).sum()) > w * 0.8]
    cols = [c for c in range(w) if int((arr[:, c] == bg).sum()) > h * 0.8]
    return rows, cols


def lattice_signature(arr: np.ndarray) -> dict[str, Any]:
    bg = grid_color(arr)
    rows, cols = grid_lines(arr, bg)
    special = [(r, c, int(arr[r, c])) for r in rows for c in cols if int(arr[r, c]) not in (0, bg)]
    rr = sorted({r for r, _, _ in special})
    cc = sorted({c for _, c, _ in special})
    return {
        "shape": f"{arr.shape[0]}x{arr.shape[1]}",
        "bg": bg,
        "line_rows": ",".join(map(str, rows)),
        "line_cols": ",".join(map(str, cols)),
        "rr": ",".join(map(str, rr)),
        "cc": ",".join(map(str, cc)),
        "rr_count": len(rr),
        "cc_count": len(cc),
        "special_count": len(special),
        "rr_delta": ",".join(map(str, np.diff(rr).tolist())) if len(rr) > 1 else "",
        "cc_delta": ",".join(map(str, np.diff(cc).tolist())) if len(cc) > 1 else "",
    }


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]

    rows: list[dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        split = "train" if idx < len(task["train"]) else ("test" if idx < len(task["train"]) + len(task["test"]) else "arc-gen")
        sig = lattice_signature(grid_to_array(ex["input"]))
        sig.update({"idx": idx, "split": split})
        rows.append(sig)

    with (EXP_DIR / "lattice_positions.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "idx",
            "split",
            "shape",
            "bg",
            "line_rows",
            "line_cols",
            "rr",
            "cc",
            "rr_count",
            "cc_count",
            "special_count",
            "rr_delta",
            "cc_delta",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    pattern_counter = Counter((row["shape"], row["rr"], row["cc"]) for row in rows)
    shape_counter = Counter(row["shape"] for row in rows)
    delta_counter = Counter((row["rr_delta"], row["cc_delta"]) for row in rows)
    by_shape: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        by_shape[str(row["shape"])].add((str(row["rr"]), str(row["cc"])))

    base = load_base_tasks()[TASK_ID]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "inventory_ready",
        "campaign_index": CAMPAIGN_INDEX,
        "task_id": TASK_ID,
        "hypothesis": "Task185 fixed lattice lowering failed because lattice positions vary; inventory position patterns before dynamic lowering.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "example_count": len(rows),
        "shape_counts": [{"shape": k, "count": v} for k, v in sorted(shape_counter.items())],
        "unique_position_pattern_count": len(pattern_counter),
        "top_position_patterns": [
            {"shape": shape, "rr": rr, "cc": cc, "count": count}
            for (shape, rr, cc), count in pattern_counter.most_common(20)
        ],
        "unique_delta_pattern_count": len(delta_counter),
        "top_delta_patterns": [
            {"rr_delta": rr_delta, "cc_delta": cc_delta, "count": count}
            for (rr_delta, cc_delta), count in delta_counter.most_common(20)
        ],
        "position_patterns_by_shape": {shape: len(patterns) for shape, patterns in sorted(by_shape.items())},
        "base_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "local_estimate_delta": 0.0,
        "new_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "decision": "If position patterns are numerous, task185 needs dynamic grid-line detection; if small by shape, shape-branch static Slice lowering may be possible.",
        "submission_decision": "no_submit: compiler design inventory",
        "leakage_risk": "low: input geometry inventory only.",
        "overfitting_risk": "medium: position branching can overfit if encoded as raw table.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## Hypothesis

task185の固定lattice loweringが失敗した理由は、4x4 lattice位置が例ごとに変わるためである。dynamic lowering前に、位置pattern数とshape依存性を測る。

## Result

- examples: `{len(rows)}`
- unique position patterns: `{len(pattern_counter)}`
- unique delta patterns: `{len(delta_counter)}`
- shape counts: `{dict(sorted(shape_counter.items()))}`
- position patterns by shape: `{ {shape: len(patterns) for shape, patterns in sorted(by_shape.items())} }`

## Interpretation

position patternが少なければshape-branch static Sliceで進める。多ければ、grid-line detectionをONNXで計算する必要がある。

## Risk

- leakage risk: low。
- overfitting risk: medium。raw position tableをそのまま提出候補にしない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
