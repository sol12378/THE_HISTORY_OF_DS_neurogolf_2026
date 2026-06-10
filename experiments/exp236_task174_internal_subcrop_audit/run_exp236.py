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


EXP_ID = "exp236_task174_internal_subcrop_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 174


def max_color_bbox_mask(x: np.ndarray) -> tuple[int, np.ndarray]:
    vals, counts = np.unique(x[x != 0], return_counts=True)
    color = int(vals[np.lexsort((vals, -counts))[0]])
    cells = np.argwhere(x == color)
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0) + 1
    crop = np.where(x[int(r0) : int(r1), int(c0) : int(c1)] == color, color, 0)
    return color, crop


def trim_zero(mask: np.ndarray) -> np.ndarray:
    nz = np.argwhere(mask != 0)
    if nz.size == 0:
        return mask[:0, :0]
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0) + 1
    return mask[int(r0) : int(r1), int(c0) : int(c1)]


def crop_at(arr: np.ndarray, h: int, w: int, anchor: str) -> np.ndarray:
    ah, aw = arr.shape
    if h > ah or w > aw:
        return np.empty((0, 0), dtype=arr.dtype)
    if "top" in anchor:
        r = 0
    elif "bottom" in anchor:
        r = ah - h
    else:
        r = (ah - h) // 2
    if "left" in anchor:
        c = 0
    elif "right" in anchor:
        c = aw - w
    else:
        c = (aw - w) // 2
    return arr[r : r + h, c : c + w]


def candidates(mask: np.ndarray, out_shape: tuple[int, int]) -> list[tuple[str, np.ndarray]]:
    h, w = out_shape
    bases = [("bbox", mask), ("trim", trim_zero(mask))]
    anchors = [
        "top_left",
        "top_center",
        "top_right",
        "center_left",
        "center",
        "center_right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
    ]
    out: list[tuple[str, np.ndarray]] = []
    for base_name, base in bases:
        for anchor in anchors:
            crop = crop_at(base, h, w, anchor)
            if crop.shape == (h, w):
                out.append((f"{base_name}_{anchor}", crop))
                out.append((f"{base_name}_{anchor}_flipud", np.flipud(crop)))
                out.append((f"{base_name}_{anchor}_fliplr", np.fliplr(crop)))
                if h == w:
                    out.append((f"{base_name}_{anchor}_rot90", np.rot90(crop, 1)))
                    out.append((f"{base_name}_{anchor}_rot180", np.rot90(crop, 2)))
                    out.append((f"{base_name}_{anchor}_rot270", np.rot90(crop, 3)))
    return out


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()[TASK_ID]
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    pass_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    per_example_best: list[dict[str, Any]] = []

    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        color, mask = max_color_bbox_mask(x)
        hits = []
        for name, pred in candidates(mask, tuple(y.shape)):
            ok = bool(np.array_equal(pred, y))
            pass_counts[name] += int(ok)
            if ok:
                hits.append(name)
        per_example_best.append(
            {
                "idx": idx,
                "output_shape": f"{y.shape[0]}x{y.shape[1]}",
                "max_color": color,
                "mask_shape": f"{mask.shape[0]}x{mask.shape[1]}",
                "hit_count": len(hits),
                "hits": hits[:20],
            }
        )

    for name, count in pass_counts.items():
        rows.append({"rule": name, "pass_count": int(count), "fail_count": len(examples) - int(count)})
    rows.sort(key=lambda r: (-r["pass_count"], r["rule"]))
    with (EXP_DIR / "subcrop_rule_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rule", "pass_count", "fail_count"])
        writer.writeheader()
        writer.writerows(rows)

    full_hits = [r for r in rows if r["fail_count"] == 0]
    near_hits = [r for r in rows if 0 < r["fail_count"] <= 10]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "example_count": len(examples),
        "best_rules": rows[:20],
        "full_hits": full_hits,
        "near_hits": near_hits,
        "examples_without_hit": [r for r in per_example_best if r["hit_count"] == 0][:20],
        "example_hit_hist": dict(Counter(r["hit_count"] for r in per_example_best)),
        "decision": "Full hit -> cost probe. Near hit -> failure audit. No hit -> task174 needs richer internal selection.",
        "submission_decision": "no_submit: audit only",
        "leakage_risk": "low",
        "overfitting_risk": "medium-low",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task174について、最大色component bbox/maskの内部sub-cropで出力を説明できるか監査する。

## 結果

- best_rules: `{rows[:10]}`
- full_hits: `{full_hits}`
- near_hits: `{near_hits[:10]}`
- examples_without_hit_count: `{sum(1 for r in per_example_best if r['hit_count'] == 0)}`

## 判断

full hitがあればcost probeへ進む。なければtask174はより複雑な内部選択が必要。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
