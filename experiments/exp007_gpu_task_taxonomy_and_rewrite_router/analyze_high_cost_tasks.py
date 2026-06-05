from __future__ import annotations

import csv
import json
import pathlib
from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "exp007_gpu_task_taxonomy_and_rewrite_router"
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
SELECTED = ROOT / "experiments" / "exp005_top_cost_rewrite_strict" / "rewrite_manifest.csv"


def load_task(task_id: int) -> dict[str, Any]:
    return json.loads((DATA_DIR / f"task{task_id:03d}.json").read_text(encoding="utf-8"))


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def features(task_id: int, cost: int, points: float) -> dict[str, Any]:
    task = load_task(task_id)
    examples = task["train"] + task["test"]
    rows: list[dict[str, Any]] = []
    for ex in examples:
        x = arr(ex["input"])
        y = arr(ex["output"])
        same_shape = x.shape == y.shape
        bg = Counter(x.ravel()).most_common(1)[0][0]
        non_bg = x != bg
        bb = bbox(non_bg)
        rows.append(
            {
                "in_h": x.shape[0],
                "in_w": x.shape[1],
                "out_h": y.shape[0],
                "out_w": y.shape[1],
                "same_shape": same_shape,
                "input_colors": len(set(x.ravel())),
                "output_colors": len(set(y.ravel())),
                "diff_pixels": int(np.sum(x != y)) if same_shape else -1,
                "bg": int(bg),
                "non_bg_pixels": int(np.sum(non_bg)),
                "bbox_h": 0 if bb is None else bb[2] - bb[0],
                "bbox_w": 0 if bb is None else bb[3] - bb[1],
            }
        )
    first = rows[0]
    shape_pairs = {(r["in_h"], r["in_w"], r["out_h"], r["out_w"]) for r in rows}
    same_shape_count = sum(1 for r in rows if r["same_shape"])
    if len(shape_pairs) == 1 and first["out_h"] * 2 == first["in_h"] and first["out_w"] == first["in_w"]:
        route = "height_halving_or_row_filter"
    elif same_shape_count == len(rows) and max(r["diff_pixels"] for r in rows) <= 80:
        route = "sparse_edit_or_object_completion"
    elif same_shape_count == len(rows):
        route = "same_shape_global_transform"
    else:
        route = "crop_or_resize"
    return {
        "task_id": task_id,
        "cost": cost,
        "points": points,
        "priority_gain_if_cost_900": max(1.0, 25.0 - np.log(900)) - points,
        "shape_pairs": ";".join(map(str, sorted(shape_pairs))),
        "same_shape_examples": same_shape_count,
        "avg_diff_pixels": round(float(np.mean([r["diff_pixels"] for r in rows if r["diff_pixels"] >= 0])) if any(r["diff_pixels"] >= 0 for r in rows) else -1, 3),
        "avg_non_bg_pixels": round(float(np.mean([r["non_bg_pixels"] for r in rows])), 3),
        "avg_bbox_h": round(float(np.mean([r["bbox_h"] for r in rows])), 3),
        "avg_bbox_w": round(float(np.mean([r["bbox_w"] for r in rows])), 3),
        "route": route,
    }


def draw_contact_sheet(task_ids: list[int]) -> None:
    cmap = plt.matplotlib.colors.ListedColormap(
        [
            "#000000",
            "#1e93ff",
            "#fa3d31",
            "#4ecc30",
            "#ffdd00",
            "#999999",
            "#e53ba3",
            "#ff851c",
            "#88d8f1",
            "#931131",
            "#f0f0f0",
            "#927556",
        ]
    )
    fig, axes = plt.subplots(len(task_ids), 4, figsize=(10, 2.2 * len(task_ids)))
    for row, task_id in enumerate(task_ids):
        task = load_task(task_id)
        examples = task["train"][:2]
        for col, ex in enumerate(examples):
            axes[row, col * 2].imshow(arr(ex["input"]), cmap=cmap, vmin=0, vmax=11, interpolation="nearest")
            axes[row, col * 2].set_title(f"task{task_id:03d} in{col}")
            axes[row, col * 2 + 1].imshow(arr(ex["output"]), cmap=cmap, vmin=0, vmax=11, interpolation="nearest")
            axes[row, col * 2 + 1].set_title(f"task{task_id:03d} out{col}")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(EXP_DIR / "high_cost_contact_sheet.png", dpi=160)
    plt.close(fig)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with SELECTED.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(float(r["new_cost"])), reverse=True)
    top = rows[:40]
    out_rows = [features(int(r["task_id"]), int(float(r["new_cost"])), float(r["new_points"])) for r in top]
    with (EXP_DIR / "taxonomy.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    draw_contact_sheet([int(r["task_id"]) for r in top[:10]])
    summary = {
        "exp_id": "exp007_gpu_task_taxonomy_and_rewrite_router",
        "date": "2026-06-05",
        "gpu_status": "nvidia-smi ok, PyTorch CUDA not installed in .venv, VRAM mostly occupied by other processes",
        "top_cost_tasks": [int(r["task_id"]) for r in top[:10]],
        "route_counts": dict(Counter(r["route"] for r in out_rows)),
        "best_next_tasks": [
            {"task_id": int(r["task_id"]), "route": r["route"], "gain_if_cost_900": round(float(r["priority_gain_if_cost_900"]), 3)}
            for r in out_rows[:10]
        ],
    }
    (EXP_DIR / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
