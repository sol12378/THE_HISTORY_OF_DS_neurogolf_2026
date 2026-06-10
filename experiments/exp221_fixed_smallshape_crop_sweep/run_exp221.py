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


EXP_ID = "exp221_fixed_smallshape_crop_sweep"
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
            hist = parse_shape_hist(row["output_shape_hist"])
            if len(hist) != 1:
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
    out = [
        ("id", y),
        ("flipud", np.flipud(y)),
        ("fliplr", np.fliplr(y)),
        ("transpose", y.T),
        ("anti_transpose", np.fliplr(np.flipud(y)).T),
    ]
    if y.shape[0] == y.shape[1]:
        out.extend(
            [
                ("rot90", np.rot90(y, 1)),
                ("rot180", np.rot90(y, 2)),
                ("rot270", np.rot90(y, 3)),
            ]
        )
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


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()
    task_ids = load_candidate_task_ids()
    rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for task_id in task_ids:
        task = load_task(task_id)
        shape = output_shape(task)
        if shape is None:
            skipped.append({"task_id": task_id, "reason": "variable_output_shape"})
            continue
        h, w = shape
        examples = task["train"] + task["test"] + task["arc-gen"]
        pass_counts = Counter()
        rule_names: list[str] | None = None
        for ex in examples:
            x = grid_to_array(ex["input"])
            y = grid_to_array(ex["output"])
            candidates = crop_candidates(x, h, w)
            if rule_names is None:
                rule_names = [name for name, _ in candidates]
            for name, pred in candidates:
                pass_counts[name] += int(np.array_equal(pred, y))
        ranked = sorted(
            [{"rule": name, "pass_count": int(pass_counts[name]), "fail_count": len(examples) - int(pass_counts[name])} for name in (rule_names or [])],
            key=lambda r: (-r["pass_count"], r["fail_count"], r["rule"]),
        )
        best = ranked[0] if ranked else {"rule": "", "pass_count": 0, "fail_count": len(examples)}
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
                "top_rules": ranked[:10],
            }
        )
        for rank, item in enumerate(ranked[:20]):
            rows.append({"task_id": task_id, "shape": f"{h}x{w}", "rank": rank, **item})

    with (EXP_DIR / "fixed_smallshape_crop_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "shape", "rank", "rule", "pass_count", "fail_count"])
        writer.writeheader()
        writer.writerows(rows)

    full_hits = [r for r in result_rows if r["best_fail_count"] == 0]
    near_hits = [r for r in result_rows if 0 < r["best_fail_count"] <= 10]
    fixed_anchor_full_hits = [r for r in full_hits if not str(r["best_rule"]).startswith("bbox_")]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if full_hits else ("near_hit" if near_hits else "no_full_hit"),
        "candidate_task_ids": task_ids,
        "evaluated_task_ids": [r["task_id"] for r in result_rows],
        "skipped": skipped,
        "rows": result_rows,
        "full_hits": full_hits,
        "fixed_anchor_full_hits": fixed_anchor_full_hits,
        "near_hits": near_hits,
        "decision": "fixed-anchor full hitはstatic Slice cost probeへ。bbox-only hitはdynamic crop cost wallに注意。hitなしならmask/color familyへpivot。",
        "submission_decision": "no_submit: rule sweep only",
        "leakage_risk": "low: input-only crop/transform rules.",
        "overfitting_risk": "medium-low: simple anchors and +/-1 offsets only.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

固定小shape cropish候補に、static Slice化しやすい固定anchor cropと、診断用のbbox cropを横展開する。

## 結果

- evaluated: `{[r['task_id'] for r in result_rows]}`
- full_hits: `{[(r['task_id'], r['shape'], r['best_rule']) for r in full_hits]}`
- fixed_anchor_full_hits: `{[(r['task_id'], r['shape'], r['best_rule']) for r in fixed_anchor_full_hits]}`
- near_hits: `{[(r['task_id'], r['shape'], r['best_rule'], r['best_fail_count']) for r in near_hits]}`

## 判断

fixed-anchor full hitがあればstatic Slice cost probeへ進める。bbox-only full hitはexp219/220のcost wallを踏まえて慎重に扱う。

## リスク

- leakage risk: low。
- overfitting risk: medium-low。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
