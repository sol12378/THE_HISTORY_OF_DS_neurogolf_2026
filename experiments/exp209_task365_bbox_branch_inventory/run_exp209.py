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


EXP_ID = "exp209_task365_bbox_branch_inventory"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 365


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs) + 1, max(cs) + 1


def object_patches(x: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for comp in components(x != 0):
        r0, c0, r1, c1 = bbox(comp)
        patch = x[r0:r1, c0:c1]
        vals = Counter(int(v) for v in patch.ravel() if int(v) != 0)
        out.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (r1 - r0, c1 - c0),
                "patch": patch,
                "count2": vals.get(2, 0),
                "area": int(patch.size),
            }
        )
    return out


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    base = load_base_tasks()[TASK_ID]
    selected_bbox_hist: Counter[str] = Counter()
    selected_shape_hist: Counter[str] = Counter()
    selected_rank_hist: Counter[int] = Counter()
    count2_margin_hist: Counter[int] = Counter()
    object_signature_hist: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    fail_count = 0
    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        objs = object_patches(x)
        objs_sorted = sorted(enumerate(objs), key=lambda io: (-io[1]["count2"], io[1]["bbox"][0], io[1]["bbox"][1]))
        rank, selected = objs_sorted[0]
        ok = selected["patch"].shape == y.shape and np.array_equal(selected["patch"], y)
        fail_count += int(not ok)
        selected_bbox_hist[str(selected["bbox"])] += 1
        selected_shape_hist(str(selected["shape"])) if False else None
        selected_shape_hist[str(selected["shape"])] += 1
        selected_rank_hist[rank] += 1
        counts = sorted([int(o["count2"]) for o in objs], reverse=True)
        margin = counts[0] - (counts[1] if len(counts) > 1 else 0)
        count2_margin_hist[margin] += 1
        sig = ";".join(f"{o['bbox']}:{o['count2']}" for o in objs)
        object_signature_hist[sig] += 1
        rows.append(
            {
                "idx": idx,
                "ok": ok,
                "shape": f"{x.shape[0]}x{x.shape[1]}",
                "object_count": len(objs),
                "selected_rank": rank,
                "selected_bbox": selected["bbox"],
                "selected_shape": selected["shape"],
                "count2_values": ",".join(map(str, counts)),
                "count2_margin": margin,
                "output_shape": tuple(y.shape),
            }
        )
    with (EXP_DIR / "bbox_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "inventory_ready",
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "example_count": len(examples),
        "fail_count": fail_count,
        "unique_selected_bbox_count": len(selected_bbox_hist),
        "top_selected_bboxes": selected_bbox_hist.most_common(20),
        "selected_shape_hist": dict(selected_shape_hist.most_common()),
        "selected_rank_hist": dict(sorted(selected_rank_hist.items())),
        "count2_margin_hist": dict(sorted(count2_margin_hist.items())),
        "unique_object_signature_count": len(object_signature_hist),
        "decision": "If selected bbox/shape space is too broad, do not spend ONNX effort on task365 dynamic object selection; pivot to smaller cropish tasks.",
        "submission_decision": "no_submit: inventory only",
        "leakage_risk": "low: structural inventory only.",
        "overfitting_risk": "medium: bbox branch tables can overfit if emitted directly.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task365のselected bbox/shape分布を確認し、少数branch loweringで押せる余地があるか判断する。

## 結果

- fail_count: `{fail_count}`
- unique_selected_bbox_count: `{len(selected_bbox_hist)}`
- selected_shape_hist: `{dict(selected_shape_hist.most_common())}`
- selected_rank_hist: `{dict(sorted(selected_rank_hist.items()))}`
- count2_margin_hist: `{dict(sorted(count2_margin_hist.items()))}`

## 判断

bbox/shape空間が広い場合、task365のdynamic object selectionには深入りせず、より小さいcropish taskへpivotする。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
