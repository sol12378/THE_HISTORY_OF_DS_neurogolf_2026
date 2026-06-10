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


EXP_ID = "exp218_crop3x3_wide_rule_sweep"
EXP_DIR = ROOT / "experiments" / EXP_ID
CANDIDATE_CSV = ROOT / "experiments" / "exp087_small_output_crop_candidate_scan" / "small_output_candidates.csv"


def load_candidate_task_ids() -> list[int]:
    task_ids: list[int] = []
    with CANDIDATE_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["is_cropish"] != "True":
                continue
            if int(row["max_output_area"]) != 9 or int(row["min_output_area"]) != 9:
                continue
            if row["output_shape_hist"] != "3x3:" + row["example_count"]:
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


def crop_at(x: np.ndarray, r: int, c: int, fill: int = 0) -> np.ndarray:
    out = np.full((3, 3), fill, dtype=x.dtype)
    for rr in range(3):
        for cc in range(3):
            sr = r + rr
            sc = c + cc
            if 0 <= sr < x.shape[0] and 0 <= sc < x.shape[1]:
                out[rr, cc] = x[sr, sc]
    return out


def transforms(y: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("id", y),
        ("rot90", np.rot90(y, 1)),
        ("rot180", np.rot90(y, 2)),
        ("rot270", np.rot90(y, 3)),
        ("flipud", np.flipud(y)),
        ("fliplr", np.fliplr(y)),
        ("transpose", y.T),
        ("anti_transpose", np.fliplr(np.flipud(y)).T),
    ]


def crop_candidates(x: np.ndarray) -> list[tuple[str, np.ndarray]]:
    anchors: list[tuple[str, Callable[[np.ndarray], tuple[int, int]]]] = [
        ("top_left", lambda x: (0, 0)),
        ("top_right", lambda x: (0, x.shape[1] - 3)),
        ("bottom_left", lambda x: (x.shape[0] - 3, 0)),
        ("bottom_right", lambda x: (x.shape[0] - 3, x.shape[1] - 3)),
        ("center", lambda x: ((x.shape[0] - 3) // 2, (x.shape[1] - 3) // 2)),
        ("bbox_top_left", lambda x: bbox_bounds(x)[:2]),
        ("bbox_top_right", lambda x: (bbox_bounds(x)[0], bbox_bounds(x)[3] - 3)),
        ("bbox_bottom_left", lambda x: (bbox_bounds(x)[2] - 3, bbox_bounds(x)[1])),
        ("bbox_bottom_right", lambda x: (bbox_bounds(x)[2] - 3, bbox_bounds(x)[3] - 3)),
        ("bbox_center", lambda x: ((bbox_bounds(x)[0] + bbox_bounds(x)[2] - 3) // 2, (bbox_bounds(x)[1] + bbox_bounds(x)[3] - 3) // 2)),
    ]
    out: list[tuple[str, np.ndarray]] = []
    for anchor_name, anchor_fn in anchors:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                base_name = f"{anchor_name}_dr{dr}_dc{dc}"
                r, c = anchor_fn(x)
                crop = crop_at(x, r + dr, c + dc, 0)
                out.extend((f"{base_name}_{trans_name}", y) for trans_name, y in transforms(crop))
    return out


def is_all_3x3(task: dict[str, Any]) -> bool:
    examples = task["train"] + task["test"] + task["arc-gen"]
    return all(grid_to_array(ex["output"]).shape == (3, 3) for ex in examples)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()
    task_ids = load_candidate_task_ids()
    rule_names: list[str] | None = None
    rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for task_id in task_ids:
        task = load_task(task_id)
        if not is_all_3x3(task):
            skipped.append({"task_id": task_id, "reason": "not_all_outputs_3x3"})
            continue
        examples = task["train"] + task["test"] + task["arc-gen"]
        pass_counts = Counter()
        for ex in examples:
            x = grid_to_array(ex["input"])
            y = grid_to_array(ex["output"])
            candidates = crop_candidates(x)
            if rule_names is None:
                rule_names = [name for name, _ in candidates]
            for name, pred in candidates:
                ok = pred.shape == y.shape and np.array_equal(pred, y)
                pass_counts[name] += int(ok)
        ranked = sorted(
            [{"rule": name, "pass_count": int(pass_counts[name]), "fail_count": len(examples) - int(pass_counts[name])} for name in (rule_names or [])],
            key=lambda r: (-r["pass_count"], r["fail_count"], r["rule"]),
        )
        best = ranked[0]
        result_rows.append(
            {
                "task_id": task_id,
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
            rows.append({"task_id": task_id, "rank": rank, **item})

    with (EXP_DIR / "crop3x3_rule_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "rank", "rule", "pass_count", "fail_count"])
        writer.writeheader()
        writer.writerows(rows)

    full_hits = [r for r in result_rows if r["best_fail_count"] == 0]
    near_hits = [r for r in result_rows if 0 < r["best_fail_count"] <= 10]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if full_hits else ("near_hit" if near_hits else "no_full_hit"),
        "candidate_task_ids": task_ids,
        "evaluated_task_ids": [r["task_id"] for r in result_rows],
        "skipped": skipped,
        "rule_count": len(rule_names or []),
        "rows": result_rows,
        "full_hits": full_hits,
        "near_hits": near_hits,
        "decision": "Full hitはSlice/transform loweringへ。near hitは失敗監査。なければ単純3x3 crop laneは優先度を下げる。",
        "submission_decision": "no_submit: rule sweep only",
        "leakage_risk": "low: input-only fixed/bbox crop rules.",
        "overfitting_risk": "medium-low: hand-written crop anchors; offset sweepは広いのでhidden edgeに注意。",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp087の3x3固定cropish候補に、安いSlice/transformでloweringできるcrop ruleを横展開する。

## 結果

- evaluated: `{[r['task_id'] for r in result_rows]}`
- full_hits: `{[(r['task_id'], r['best_rule']) for r in full_hits]}`
- near_hits(<=10 fail): `{[(r['task_id'], r['best_rule'], r['best_fail_count']) for r in near_hits]}`
- best: `{[(r['task_id'], r['best_rule'], r['best_pass_count'], r['example_count']) for r in result_rows]}`

## 判断

full hitはSlice/transform cost probeへ進める。なければ単純3x3 crop laneは縮小し、色変換やmask抽出を含むfamilyへ移る。

## リスク

- leakage risk: low。入力だけのcrop/transform。
- overfitting risk: medium-low。offset sweepが広いため、full hit以外は補正追加に注意。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
