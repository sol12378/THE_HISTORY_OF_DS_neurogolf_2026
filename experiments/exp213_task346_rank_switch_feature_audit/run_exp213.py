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


EXP_ID = "exp213_task346_rank_switch_feature_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 346


def comps(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        cells = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        out.append(cells)
    return out


def color_features(x: np.ndarray, color: int) -> dict[str, Any]:
    pos = np.argwhere(x == color)
    if pos.size == 0:
        return {}
    rs = pos[:, 0]
    cs = pos[:, 1]
    h, w = x.shape
    r0, r1 = int(rs.min()), int(rs.max()) + 1
    c0, c1 = int(cs.min()), int(cs.max()) + 1
    comp_sizes = sorted([len(c) for c in comps(x == color)], reverse=True)
    return {
        "count": int(len(pos)),
        "bbox": (r0, c0, r1, c1),
        "bbox_h": r1 - r0,
        "bbox_w": c1 - c0,
        "bbox_area": (r1 - r0) * (c1 - c0),
        "density_num": int(len(pos)),
        "density_den": (r1 - r0) * (c1 - c0),
        "touch_top": int(r0 == 0),
        "touch_bottom": int(r1 == h),
        "touch_left": int(c0 == 0),
        "touch_right": int(c1 == w),
        "edge_touches": int(r0 == 0) + int(r1 == h) + int(c0 == 0) + int(c1 == w),
        "corner_count": int(x[0, 0] == color) + int(x[0, -1] == color) + int(x[-1, 0] == color) + int(x[-1, -1] == color),
        "component_count": len(comp_sizes),
        "largest_component": comp_sizes[0],
    }


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    base = load_base_tasks()[TASK_ID]
    rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = int(grid_to_array(ex["output"])[0, 0])
        counts = Counter(int(v) for v in x.ravel() if int(v) != 0)
        ranked = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
        rank0, rank1 = ranked[0][0], ranked[1][0] if len(ranked) > 1 else ranked[0][0]
        y_rank = next((i for i, (c, _) in enumerate(ranked) if c == y), -1)
        f0 = color_features(x, rank0)
        f1 = color_features(x, rank1)
        row: dict[str, Any] = {
            "idx": idx,
            "shape": f"{x.shape[0]}x{x.shape[1]}",
            "expected": y,
            "rank0": rank0,
            "rank1": rank1,
            "rank0_count": counts[rank0],
            "rank1_count": counts[rank1],
            "expected_rank": y_rank,
        }
        for k in sorted(f0):
            if k != "bbox":
                row[f"r0_{k}"] = f0[k]
                row[f"r1_{k}"] = f1[k]
                if isinstance(f0[k], int) and isinstance(f1[k], int):
                    row[f"d_{k}"] = f1[k] - f0[k]
        rows.append(row)
        if y_rank == 1:
            switch_rows.append(row)

    # Find simple feature predicates that cover all four switch rows with low false positives.
    predicates: list[dict[str, Any]] = []
    numeric_keys = [k for k in rows[0] if k.startswith("d_") or k.startswith("r0_") or k.startswith("r1_")]
    for key in numeric_keys:
        vals = sorted({int(row[key]) for row in rows if isinstance(row.get(key), int)})
        for thr in vals:
            for op in ("<=", ">=", "=="):
                def pred(row: dict[str, Any], key: str = key, thr: int = thr, op: str = op) -> bool:
                    v = int(row[key])
                    if op == "<=":
                        return v <= thr
                    if op == ">=":
                        return v >= thr
                    return v == thr

                tp = sum(1 for row in rows if row["expected_rank"] == 1 and pred(row))
                fp = sum(1 for row in rows if row["expected_rank"] == 0 and pred(row))
                fn = sum(1 for row in rows if row["expected_rank"] == 1 and not pred(row))
                if fn == 0:
                    predicates.append({"key": key, "op": op, "thr": thr, "tp": tp, "fp": fp, "fn": fn})
    predicates.sort(key=lambda p: (p["fp"], p["key"], p["op"], p["thr"]))

    with (EXP_DIR / "feature_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "audit_complete",
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "example_count": len(examples),
        "switch_count": len(switch_rows),
        "switch_rows": switch_rows,
        "best_predicates_covering_switches": predicates[:20],
        "decision": "If a simple predicate covers four rank1 cases with few false positives, validate rank-switch rule; otherwise pivot.",
        "submission_decision": "no_submit: feature audit only",
        "leakage_risk": "low: input feature audit.",
        "overfitting_risk": "medium: only four switch cases; predicate must be simple and structural.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task346で`least_nz`が失敗する4例をrank0/rank1切替として説明できる簡単な特徴を探す。

## 結果

- switch_count: `{len(switch_rows)}`
- best_predicates: `{predicates[:5]}`

## 判断

低FPの単純predicateがあればrank-switch ruleを検証する。なければpivot。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
