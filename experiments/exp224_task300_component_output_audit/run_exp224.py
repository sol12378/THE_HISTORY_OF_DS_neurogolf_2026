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


EXP_ID = "exp224_task300_component_output_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 300


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


def matches_component(y: np.ndarray, comp: dict[str, Any]) -> dict[str, bool]:
    same_shape = tuple(y.shape) == (comp["h"], comp["w"])
    if not same_shape:
        return {"same_shape": False, "crop_exact": False, "mask_binary": False, "mask_colorized": False}
    crop = comp["crop"]
    mask = comp["mask"]
    y_bin = normalize_binary(y)
    nonzero_vals = sorted({int(v) for v in y.ravel() if int(v) != 0})
    if len(nonzero_vals) == 1:
        colorized = mask * nonzero_vals[0]
    else:
        colorized = np.full(y.shape, -999, dtype=np.int64)
    return {
        "same_shape": True,
        "crop_exact": bool(np.array_equal(y, crop)),
        "mask_binary": bool(np.array_equal(y_bin, mask)),
        "mask_colorized": bool(np.array_equal(y, colorized)),
    }


def rank_features(comps: list[dict[str, Any]], idx: int) -> dict[str, int]:
    comp = comps[idx]
    features: dict[str, int] = {}
    for key, reverse in [("size", True), ("area", True), ("r0", False), ("c0", False), ("color", False)]:
        ordered = sorted(range(len(comps)), key=lambda i: ((-comps[i][key] if reverse else comps[i][key]), comps[i]["r0"], comps[i]["c0"], comps[i]["color"]))
        features[f"rank_{key}_{'desc' if reverse else 'asc'}"] = ordered.index(idx)
    return features


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()[TASK_ID]
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    rows: list[dict[str, Any]] = []
    selector_hits = Counter()
    match_type_hits = Counter()
    ambiguous = 0

    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        comps = components(x)
        matches: list[dict[str, Any]] = []
        for ci, comp in enumerate(comps):
            flags = matches_component(y, comp)
            if flags["same_shape"]:
                matches.append({"component_index": ci, **{k: v for k, v in comp.items() if k not in ["crop", "mask"]}, **flags, **rank_features(comps, ci)})
        exactish = [m for m in matches if m["crop_exact"] or m["mask_binary"] or m["mask_colorized"]]
        if len(exactish) != 1:
            ambiguous += 1
        chosen = exactish[0] if exactish else (matches[0] if matches else None)
        if chosen:
            for key in ["crop_exact", "mask_binary", "mask_colorized"]:
                if chosen.get(key):
                    match_type_hits[key] += 1
            for feat, val in chosen.items():
                if feat.startswith("rank_") and val == 0:
                    selector_hits[feat] += 1
        rows.append(
            {
                "idx": idx,
                "output_shape": f"{y.shape[0]}x{y.shape[1]}",
                "component_count": len(comps),
                "shape_match_count": len(matches),
                "exactish_count": len(exactish),
                "chosen": chosen,
            }
        )

    flat_rows = []
    for row in rows:
        chosen = row["chosen"] or {}
        flat_rows.append(
            {
                "idx": row["idx"],
                "output_shape": row["output_shape"],
                "component_count": row["component_count"],
                "shape_match_count": row["shape_match_count"],
                "exactish_count": row["exactish_count"],
                "chosen_color": chosen.get("color", ""),
                "chosen_size": chosen.get("size", ""),
                "chosen_area": chosen.get("area", ""),
                "chosen_r0": chosen.get("r0", ""),
                "chosen_c0": chosen.get("c0", ""),
                "crop_exact": chosen.get("crop_exact", ""),
                "mask_binary": chosen.get("mask_binary", ""),
                "mask_colorized": chosen.get("mask_colorized", ""),
                "rank_size_desc": chosen.get("rank_size_desc", ""),
                "rank_area_desc": chosen.get("rank_area_desc", ""),
                "rank_r0_asc": chosen.get("rank_r0_asc", ""),
                "rank_c0_asc": chosen.get("rank_c0_asc", ""),
                "rank_color_asc": chosen.get("rank_color_asc", ""),
            }
        )

    with (EXP_DIR / "task300_component_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "example_count": len(examples),
        "ambiguous_or_no_exactish": ambiguous,
        "match_type_hits": dict(match_type_hits),
        "selector_rank0_hits": dict(selector_hits),
        "sample_rows": rows[:20],
        "decision": "If a selector rank feature has 267 hits and output is mask/crop exact, attempt a focused Python rule and cost proxy. Otherwise task300 needs richer component selection.",
        "submission_decision": "no_submit: diagnostic only",
        "leakage_risk": "low: input/output structural audit.",
        "overfitting_risk": "medium-low: selector hypotheses must be validated before lowering.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp223で最上位候補になったtask300について、出力がどの入力componentのcrop/maskに対応するか、単純rank selectorで説明できるかを監査する。

## 結果

- baseline_cost: `{base.cost}`
- ambiguous_or_no_exactish: `{ambiguous}`
- match_type_hits: `{dict(match_type_hits)}`
- selector_rank0_hits: `{dict(selector_hits)}`

## 判断

rank0 selectorが全例を説明し、出力がcomponent crop/maskに一致するなら次はPython rule化とcost proxy。そうでなければcomponent選択が複雑で、別候補へ移る。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
