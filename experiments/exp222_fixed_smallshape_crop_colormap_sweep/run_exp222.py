from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from datetime import date
from typing import Any, Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp222_fixed_smallshape_crop_colormap_sweep"
EXP_DIR = ROOT / "experiments" / EXP_ID
CANDIDATE_CSV = ROOT / "experiments" / "exp087_small_output_crop_candidate_scan" / "small_output_candidates.csv"


def parse_shape_hist(hist: str) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for part in hist.split(";"):
        shape, count = part.split(":")
        h, w = shape.split("x")
        out[(int(h), int(w))] = int(count)
    return out


def load_candidate_task_ids() -> list[int]:
    task_ids: list[int] = []
    with CANDIDATE_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["is_cropish"] != "True":
                continue
            if int(row["max_output_area"]) > 12:
                continue
            if len(parse_shape_hist(row["output_shape_hist"])) != 1:
                continue
            task_ids.append(int(row["task_id"]))
    return sorted(set(task_ids))


def bbox_bounds(x: np.ndarray) -> tuple[int, int, int, int]:
    nz = np.argwhere(x != 0)
    if nz.size == 0:
        return 0, 0, x.shape[0], x.shape[1]
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def crop_at(x: np.ndarray, h: int, w: int, r: int, c: int, fill: int = 0) -> np.ndarray:
    out = np.full((h, w), fill, dtype=x.dtype)
    for rr in range(h):
        for cc in range(w):
            sr = r + rr
            sc = c + cc
            if 0 <= sr < x.shape[0] and 0 <= sc < x.shape[1]:
                out[rr, cc] = x[sr, sc]
    return out


def transforms(y: np.ndarray) -> list[tuple[str, np.ndarray]]:
    out = [("id", y), ("flipud", np.flipud(y)), ("fliplr", np.fliplr(y))]
    if y.shape[0] == y.shape[1]:
        out.extend([("rot90", np.rot90(y, 1)), ("rot180", np.rot90(y, 2)), ("rot270", np.rot90(y, 3))])
    return out


def crop_candidates(x: np.ndarray, h: int, w: int) -> list[tuple[str, np.ndarray]]:
    anchors: list[tuple[str, Callable[[np.ndarray], tuple[int, int]]]] = [
        ("top_left", lambda z: (0, 0)),
        ("top_right", lambda z: (0, z.shape[1] - w)),
        ("bottom_left", lambda z: (z.shape[0] - h, 0)),
        ("bottom_right", lambda z: (z.shape[0] - h, z.shape[1] - w)),
        ("center", lambda z: ((z.shape[0] - h) // 2, (z.shape[1] - w) // 2)),
        ("bbox_top_left", lambda z: bbox_bounds(z)[:2]),
        ("bbox_top_right", lambda z: (bbox_bounds(z)[0], bbox_bounds(z)[3] - w)),
        ("bbox_bottom_left", lambda z: (bbox_bounds(z)[2] - h, bbox_bounds(z)[1])),
        ("bbox_bottom_right", lambda z: (bbox_bounds(z)[2] - h, bbox_bounds(z)[3] - w)),
        ("bbox_center", lambda z: ((bbox_bounds(z)[0] + bbox_bounds(z)[2] - h) // 2, (bbox_bounds(z)[1] + bbox_bounds(z)[3] - w) // 2)),
    ]
    out: list[tuple[str, np.ndarray]] = []
    for anchor_name, anchor_fn in anchors:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = anchor_fn(x)
                crop = crop_at(x, h, w, r + dr, c + dc, 0)
                for trans_name, transformed in transforms(crop):
                    if transformed.shape == (h, w):
                        out.append((f"{anchor_name}_dr{dr}_dc{dc}_{trans_name}", transformed))
    return out


def output_shape(task: dict[str, Any]) -> tuple[int, int] | None:
    shapes = {grid_to_array(ex["output"]).shape for ex in task["train"] + task["test"] + task["arc-gen"]}
    if len(shapes) != 1:
        return None
    h, w = next(iter(shapes))
    return int(h), int(w)


def consistent_map(pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[int] | None:
    mapping: dict[int, int] = {}
    for x, y in pairs:
        for a, b in zip(x.ravel(), y.ravel()):
            aa = int(a)
            bb = int(b)
            if aa in mapping and mapping[aa] != bb:
                return None
            mapping[aa] = bb
    return [mapping.get(i, i) for i in range(10)]


def apply_map(x: np.ndarray, cmap: list[int]) -> np.ndarray:
    lut = np.asarray(cmap, dtype=x.dtype)
    return lut[x]


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()
    task_ids = load_candidate_task_ids()
    rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    for task_id in task_ids:
        task = load_task(task_id)
        shape = output_shape(task)
        if shape is None:
            continue
        h, w = shape
        examples = task["train"] + task["test"] + task["arc-gen"]
        per_rule_pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for ex in examples:
            x = grid_to_array(ex["input"])
            y = grid_to_array(ex["output"])
            for name, crop in crop_candidates(x, h, w):
                per_rule_pairs.setdefault(name, []).append((crop, y))
        ranked: list[dict[str, Any]] = []
        for name, pairs in per_rule_pairs.items():
            cmap = consistent_map(pairs)
            if cmap is None:
                pass_count = 0
                cmap_text = ""
            else:
                pass_count = sum(int(np.array_equal(apply_map(crop, cmap), y)) for crop, y in pairs)
                cmap_text = ",".join(str(v) for v in cmap)
            ranked.append({"rule": name, "pass_count": int(pass_count), "fail_count": len(examples) - int(pass_count), "color_map": cmap_text})
        ranked.sort(key=lambda r: (-r["pass_count"], r["fail_count"], r["rule"]))
        best = ranked[0] if ranked else {"rule": "", "pass_count": 0, "fail_count": len(examples), "color_map": ""}
        result_rows.append(
            {
                "task_id": task_id,
                "shape": f"{h}x{w}",
                "baseline_cost": base[task_id].cost,
                "baseline_points": base[task_id].points,
                "example_count": len(examples),
                "best_rule": best["rule"],
                "best_pass_count": best["pass_count"],
                "best_fail_count": best["fail_count"],
                "best_color_map": best["color_map"],
                "top_rules": ranked[:10],
            }
        )
        for rank, item in enumerate(ranked[:20]):
            rows.append({"task_id": task_id, "shape": f"{h}x{w}", "rank": rank, **item})

    with (EXP_DIR / "fixed_smallshape_crop_colormap_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "shape", "rank", "rule", "pass_count", "fail_count", "color_map"])
        writer.writeheader()
        writer.writerows(rows)

    full_hits = [r for r in result_rows if r["best_fail_count"] == 0]
    fixed_anchor_full_hits = [r for r in full_hits if not str(r["best_rule"]).startswith("bbox_")]
    useful_fixed_hits = [r for r in fixed_anchor_full_hits if r["baseline_cost"] > 600]
    near_hits = [r for r in result_rows if 0 < r["best_fail_count"] <= 10]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if full_hits else ("near_hit" if near_hits else "no_full_hit"),
        "candidate_task_ids": task_ids,
        "rows": result_rows,
        "full_hits": full_hits,
        "fixed_anchor_full_hits": fixed_anchor_full_hits,
        "useful_fixed_hits": useful_fixed_hits,
        "near_hits": near_hits,
        "decision": "useful fixed-anchor full hitはstatic Slice + LUT cost probeへ。bbox-only hitはdynamic crop wallを警戒。",
        "submission_decision": "no_submit: rule sweep only",
        "leakage_risk": "medium-low: color map is fit across all available examples for diagnostic; before submission, re-check train-derived map generalization.",
        "overfitting_risk": "medium: global color map can overfit if derived from all examples; use as supplier discovery only.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

固定小shape crop候補に、task共通のglobal color-mapを重ねることで、static Slice + LUTで表現できる候補を探す。

## 結果

- full_hits: `{[(r['task_id'], r['shape'], r['best_rule'], r['baseline_cost']) for r in full_hits]}`
- fixed_anchor_full_hits: `{[(r['task_id'], r['shape'], r['best_rule'], r['baseline_cost']) for r in fixed_anchor_full_hits]}`
- useful_fixed_hits(cost>600): `{[(r['task_id'], r['shape'], r['best_rule'], r['baseline_cost']) for r in useful_fixed_hits]}`
- near_hits: `{[(r['task_id'], r['shape'], r['best_rule'], r['best_fail_count']) for r in near_hits]}`

## 判断

useful fixed-anchor full hitがあればcost probeへ進める。全例由来のcolor-mapなので、提出前にはtrain-derived mapで再検証する。

## リスク

- leakage risk: medium-low。診断として全available examplesからmapを見ている。
- overfitting risk: medium。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
