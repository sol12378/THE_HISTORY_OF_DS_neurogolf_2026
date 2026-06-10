from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, deque
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp235_task174_component_output_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 174


def components(x: np.ndarray) -> list[dict[str, Any]]:
    seen = np.zeros(x.shape, dtype=bool)
    comps: list[dict[str, Any]] = []
    for r, c in np.argwhere(x != 0):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        color = int(x[r, c])
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            cells.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < x.shape[0] and 0 <= nc < x.shape[1] and not seen[nr, nc] and int(x[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        arr = np.asarray(cells)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0) + 1
        crop = x[int(r0) : int(r1), int(c0) : int(c1)]
        comps.append({"color": color, "size": len(cells), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "crop": crop})
    return comps


def max_color_rule(x: np.ndarray) -> tuple[int, int, int, int, int]:
    counts = Counter(int(v) for v in x.ravel() if int(v) != 0)
    color = min(counts, key=lambda k: (-counts[k], k))
    cells = np.argwhere(x == color)
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0) + 1
    return int(color), int(r0), int(c0), int(r1), int(c1)


def padded_equal(pred: np.ndarray, y: np.ndarray) -> bool:
    padded = np.zeros((30, 30), dtype=np.int64)
    padded[: pred.shape[0], : pred.shape[1]] = pred
    target = np.zeros((30, 30), dtype=np.int64)
    target[: y.shape[0], : y.shape[1]] = y
    return bool(np.array_equal(padded, target))


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()[TASK_ID]
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    rows: list[dict[str, Any]] = []
    largest_crop_pass = 0
    max_color_crop_pass = 0
    max_color_matches_largest = 0
    selector_hits = Counter()
    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        comps = components(x)
        largest = sorted(comps, key=lambda c: (-c["size"], c["color"], c["r0"], c["c0"]))[0]
        largest_ok = padded_equal(largest["crop"], y)
        largest_crop_pass += int(largest_ok)
        color, r0, c0, r1, c1 = max_color_rule(x)
        max_crop = np.where(x[r0:r1, c0:c1] == color, color, 0)
        max_ok = padded_equal(max_crop, y)
        max_color_crop_pass += int(max_ok)
        same = color == largest["color"] and r0 == largest["r0"] and c0 == largest["c0"] and r1 == largest["r1"] and c1 == largest["c1"]
        max_color_matches_largest += int(same)
        for key, reverse in [("size", True), ("r0", False), ("c0", False), ("color", False)]:
            ordered = sorted(range(len(comps)), key=lambda i: ((-comps[i][key] if reverse else comps[i][key]), comps[i]["r0"], comps[i]["c0"], comps[i]["color"]))
            if comps[ordered[0]] is largest:
                selector_hits[f"rank_{key}_{'desc' if reverse else 'asc'}"] += 1
        rows.append(
            {
                "idx": idx,
                "output_shape": f"{y.shape[0]}x{y.shape[1]}",
                "component_count": len(comps),
                "largest_ok": largest_ok,
                "max_color_ok": max_ok,
                "max_color_matches_largest": same,
                "largest_color": largest["color"],
                "largest_size": largest["size"],
                "largest_bbox": f"{largest['r0']},{largest['c0']},{largest['r1']},{largest['c1']}",
                "max_color": color,
                "max_color_bbox": f"{r0},{c0},{r1},{c1}",
            }
        )

    with (EXP_DIR / "task174_component_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "example_count": len(examples),
        "largest_crop_pass": largest_crop_pass,
        "max_color_crop_pass": max_color_crop_pass,
        "max_color_matches_largest": max_color_matches_largest,
        "selector_rank0_hits": dict(selector_hits),
        "fail_examples": [r for r in rows if not r["max_color_ok"]][:20],
        "decision": "If max_color_crop is full, adapt task300 lowering with variable max shape. Otherwise inspect fail pattern or pivot.",
        "submission_decision": "no_submit: audit only",
        "leakage_risk": "low",
        "overfitting_risk": "medium-low",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task174がtask300同様、最大componentまたは最大nonzero色count cropで解けるか監査する。

## 結果

- baseline_cost: `{base.cost}`
- largest_crop_pass: `{largest_crop_pass}/{len(examples)}`
- max_color_crop_pass: `{max_color_crop_pass}/{len(examples)}`
- max_color_matches_largest: `{max_color_matches_largest}/{len(examples)}`

## 判断

max_color_cropがfullならtask300 loweringを拡張する。そうでなければ失敗patternを監査する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
