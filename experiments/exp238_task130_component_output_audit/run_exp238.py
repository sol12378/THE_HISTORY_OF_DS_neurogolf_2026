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


EXP_ID = "exp238_task130_component_output_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 130


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
        mask = (crop == color).astype(np.int64)
        comps.append(
            {
                "color": color,
                "size": len(cells),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "h": int(r1 - r0),
                "w": int(c1 - c0),
                "area": int((r1 - r0) * (c1 - c0)),
                "crop": crop,
                "mask": mask,
            }
        )
    return comps


def normalize_binary(y: np.ndarray) -> np.ndarray:
    vals = sorted({int(v) for v in y.ravel()})
    if len(vals) <= 1:
        return np.zeros(y.shape, dtype=np.int64)
    bg = max(vals, key=lambda v: int(np.sum(y == v)))
    return (y != bg).astype(np.int64)


def color_counts(x: np.ndarray) -> Counter[int]:
    return Counter(int(v) for v in x.ravel() if int(v) != 0)


def output_color_count(y: np.ndarray) -> int:
    return len({int(v) for v in y.ravel() if int(v) != 0})


def pad_equal(pred: np.ndarray, y: np.ndarray) -> bool:
    canvas_pred = np.zeros((30, 30), dtype=np.int64)
    canvas_target = np.zeros((30, 30), dtype=np.int64)
    canvas_pred[: pred.shape[0], : pred.shape[1]] = pred
    canvas_target[: y.shape[0], : y.shape[1]] = y
    return bool(np.array_equal(canvas_pred, canvas_target))


def crop_at(x: np.ndarray, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.int64)
    crop = x[r0 : r0 + h, c0 : c0 + w]
    out[: crop.shape[0], : crop.shape[1]] = crop
    return out


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()[TASK_ID]
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]

    rows: list[dict[str, Any]] = []
    counters = Counter()
    output_shapes = Counter()
    output_color_counts = Counter()
    binary_masks = Counter()
    exact_crop_offsets = Counter()

    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        comps = components(x)
        output_shapes[f"{y.shape[0]}x{y.shape[1]}"] += 1
        output_color_counts[output_color_count(y)] += 1
        binary_masks[tuple(int(v != 0) for v in y.ravel())] += 1

        counts = color_counts(x)
        max_color = min(counts, key=lambda k: (-counts[k], k)) if counts else 0
        largest = sorted(comps, key=lambda c: (-c["size"], c["color"], c["r0"], c["c0"]))[0] if comps else None

        shape_matches = []
        exactish_matches = []
        for ci, comp in enumerate(comps):
            same_shape = tuple(y.shape) == (comp["h"], comp["w"])
            if not same_shape:
                continue
            y_bin = normalize_binary(y)
            nonzero_vals = sorted({int(v) for v in y.ravel() if int(v) != 0})
            colorized = comp["mask"] * nonzero_vals[0] if len(nonzero_vals) == 1 else np.full(y.shape, -999, dtype=np.int64)
            rec = {
                "component_index": ci,
                "color": comp["color"],
                "size": comp["size"],
                "r0": comp["r0"],
                "c0": comp["c0"],
                "h": comp["h"],
                "w": comp["w"],
                "crop_exact": bool(np.array_equal(y, comp["crop"])),
                "mask_binary": bool(np.array_equal(y_bin, comp["mask"])),
                "mask_colorized": bool(np.array_equal(y, colorized)),
            }
            shape_matches.append(rec)
            if rec["crop_exact"] or rec["mask_binary"] or rec["mask_colorized"]:
                exactish_matches.append(rec)

        max_color_cells = np.argwhere(x == max_color)
        if len(max_color_cells):
            r0, c0 = max_color_cells.min(axis=0)
            r1, c1 = max_color_cells.max(axis=0) + 1
            max_color_crop = crop_at(x, int(r0), int(c0), y.shape[0], y.shape[1])
            max_color_mask = np.where(max_color_crop == max_color, max_color, 0)
            counters["max_color_crop_top_left"] += int(pad_equal(max_color_crop, y))
            counters["max_color_mask_top_left"] += int(pad_equal(max_color_mask, y))

        if largest:
            largest_crop = crop_at(x, largest["r0"], largest["c0"], y.shape[0], y.shape[1])
            largest_mask = np.where(largest_crop == largest["color"], largest["color"], 0)
            counters["largest_crop_top_left"] += int(pad_equal(largest_crop, y))
            counters["largest_mask_top_left"] += int(pad_equal(largest_mask, y))

        # Exhaustive same-shape crop check. If this is high/full, a dynamic crop rule may exist.
        exact_crop_count = 0
        for rr in range(max(1, x.shape[0] - y.shape[0] + 1)):
            for cc in range(max(1, x.shape[1] - y.shape[1] + 1)):
                if np.array_equal(crop_at(x, rr, cc, y.shape[0], y.shape[1]), y):
                    exact_crop_count += 1
                    exact_crop_offsets[f"{rr},{cc}"] += 1
        counters["any_same_shape_input_crop_exact"] += int(exact_crop_count > 0)

        counters["component_shape_match"] += int(len(shape_matches) > 0)
        counters["component_exactish_unique"] += int(len(exactish_matches) == 1)
        counters["component_exactish_any"] += int(len(exactish_matches) > 0)
        for key in ("crop_exact", "mask_binary", "mask_colorized"):
            counters[f"component_{key}"] += int(any(m[key] for m in exactish_matches))

        rows.append(
            {
                "idx": idx,
                "output_shape": f"{y.shape[0]}x{y.shape[1]}",
                "output_color_count": output_color_count(y),
                "component_count": len(comps),
                "shape_match_count": len(shape_matches),
                "exactish_count": len(exactish_matches),
                "max_color": max_color,
                "largest_color": largest["color"] if largest else "",
                "largest_size": largest["size"] if largest else "",
                "same_shape_input_crop_exact_count": exact_crop_count,
                "first_exactish": exactish_matches[0] if exactish_matches else {},
            }
        )

    with (EXP_DIR / "task130_component_audit.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "idx",
            "output_shape",
            "output_color_count",
            "component_count",
            "shape_match_count",
            "exactish_count",
            "max_color",
            "largest_color",
            "largest_size",
            "same_shape_input_crop_exact_count",
            "first_exactish",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "example_count": len(examples),
        "output_shapes": dict(output_shapes),
        "output_color_count_hist": dict(sorted(output_color_counts.items())),
        "binary_signature_count": len(binary_masks),
        "counters": dict(counters),
        "top_exact_crop_offsets": exact_crop_offsets.most_common(20),
        "sample_rows": rows[:20],
        "decision": "If component exactish or max-color/largest proxy is full, proceed to cost probe. If only shape matches but content does not, task130 needs separate content rule mining.",
        "submission_decision": "no_submit: diagnostic only",
        "leakage_risk": "low: input/output structural audit only.",
        "overfitting_risk": "medium-low: hypotheses must be revalidated before lowering.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp223でshape-full候補になったtask130について、task300と同様にcomponent crop/maskまたはmax-color proxyで出力内容まで説明できるか監査する。

## 結果

- baseline_cost: `{base.cost}`
- output_shapes: `{dict(output_shapes)}`
- output_color_count_hist: `{dict(sorted(output_color_counts.items()))}`
- binary_signature_count: `{len(binary_masks)}`
- counters: `{dict(counters)}`
- top_exact_crop_offsets: `{exact_crop_offsets.most_common(10)}`

## 判断

component exactishまたはmax-color/largest proxyがfullならcost probeへ進む。shapeのみ一致で内容が合わない場合は、task130は別途content rule miningが必要。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
