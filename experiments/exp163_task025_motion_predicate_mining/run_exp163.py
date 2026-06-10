from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_task


EXP_DIR = ROOT / "experiments" / "exp163_task025_motion_predicate_mining"
TASK_ID = 25


def bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not points:
        return None
    rs = [p[0] for p in points]
    cs = [p[1] for p in points]
    return min(rs), min(cs), max(rs), max(cs)


def coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def color_counts(arr: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def nearest_pair_stats(removed: list[tuple[int, int]], added: list[tuple[int, int]]) -> dict[str, Any]:
    if not removed or not added:
        return {"pair_count": 0, "perfect_size_match": len(removed) == len(added)}
    n = min(len(removed), len(added))
    if len(removed) <= 8 and len(added) <= 8:
        best_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        best_cost = 10**9
        for perm in permutations(range(len(added)), n):
            used_removed = removed[:n]
            pairs = [(used_removed[i], added[perm[i]]) for i in range(n)]
            cost = sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in pairs)
            if cost < best_cost:
                best_cost = cost
                best_pairs = pairs
        pairs = best_pairs
    else:
        remaining = added.copy()
        pairs = []
        for src in removed:
            if not remaining:
                break
            dst = min(remaining, key=lambda p: (abs(src[0] - p[0]) + abs(src[1] - p[1]), p[0], p[1]))
            remaining.remove(dst)
            pairs.append((src, dst))

    deltas = [(b[0] - a[0], b[1] - a[1]) for a, b in pairs]
    delta_counts = Counter(deltas)
    line_counts = Counter(classify_delta(dr, dc) for dr, dc in deltas)
    return {
        "pair_count": len(pairs),
        "perfect_size_match": len(removed) == len(added),
        "delta_counts": {f"{dr},{dc}": n for (dr, dc), n in delta_counts.items()},
        "line_counts": dict(line_counts),
        "top_delta": fmt_counter(delta_counts, limit=5),
        "top_line_type": fmt_counter(line_counts, limit=5),
        "dominant_delta_share": round(delta_counts.most_common(1)[0][1] / max(1, len(pairs)), 4),
        "dominant_line_share": round(line_counts.most_common(1)[0][1] / max(1, len(pairs)), 4),
        "mean_manhattan": round(sum(abs(dr) + abs(dc) for dr, dc in deltas) / max(1, len(deltas)), 4),
    }


def classify_delta(dr: int, dc: int) -> str:
    if dr == 0 and dc == 0:
        return "same"
    if dr == 0:
        return "horizontal"
    if dc == 0:
        return "vertical"
    if abs(dr) == abs(dc):
        return "diagonal"
    return "other"


def fmt_counter(counter: Counter[Any], limit: int = 8) -> str:
    return " ".join(f"{k}:{v}" for k, v in counter.most_common(limit))


def example_features(idx: int, split: str, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {"idx": idx, "split": split, "shape": str(tuple(x.shape))}
    row["color_counts_preserved"] = color_counts(x) == color_counts(y)
    row["nonzero_count_preserved"] = int((x != 0).sum()) == int((y != 0).sum())
    row["input_nonzero_bbox"] = str(bbox(coords(x != 0)))
    row["output_nonzero_bbox"] = str(bbox(coords(y != 0)))
    row["removed_total"] = int(((x != 0) & (y == 0)).sum())
    row["added_total"] = int(((x == 0) & (y != 0)).sum())
    row["recolor_total"] = int(((x != 0) & (y != 0) & (x != y)).sum())
    color_rows = []
    all_line_types = Counter()
    all_deltas = Counter()
    preserved_colors = 0
    balanced_colors = 0
    for color in sorted(set(x.flatten().tolist()) | set(y.flatten().tolist())):
        color = int(color)
        if color == 0:
            continue
        removed = coords((x == color) & (y != color))
        added = coords((x != color) & (y == color))
        if int((x == color).sum()) == int((y == color).sum()):
            preserved_colors += 1
        if len(removed) == len(added):
            balanced_colors += 1
        stats = nearest_pair_stats(removed, added)
        for key, value in stats.get("line_counts", {}).items():
            all_line_types[key] += int(value)
        for key, value in stats.get("delta_counts", {}).items():
            all_deltas[key] += int(value)
        color_rows.append(
            {
                "color": color,
                "input_count": int((x == color).sum()),
                "output_count": int((y == color).sum()),
                "removed": len(removed),
                "added": len(added),
                "removed_bbox": str(bbox(removed)),
                "added_bbox": str(bbox(added)),
                **stats,
            }
        )
    row["color_count_preserved_nonzero_colors"] = preserved_colors
    row["balanced_removed_added_colors"] = balanced_colors
    row["color_motion"] = json.dumps(color_rows, ensure_ascii=False, sort_keys=True)
    row["all_top_line_types"] = fmt_counter(all_line_types)
    row["all_top_deltas"] = fmt_counter(all_deltas)
    row["dominant_line_type"] = all_line_types.most_common(1)[0][0] if all_line_types else ""
    row["dominant_delta"] = all_deltas.most_common(1)[0][0] if all_deltas else ""
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "example_count": len(rows),
        "color_counts_preserved": sum(1 for r in rows if r["color_counts_preserved"]),
        "nonzero_count_preserved": sum(1 for r in rows if r["nonzero_count_preserved"]),
        "removed_added_recolor_counts": dict(Counter((r["removed_total"], r["added_total"], r["recolor_total"]) for r in rows).most_common(20)),
        "dominant_line_type_counts": dict(Counter(r["dominant_line_type"] for r in rows).most_common()),
        "dominant_delta_counts": dict(Counter(r["dominant_delta"] for r in rows).most_common(20)),
        "balanced_all_changed_colors": sum(1 for r in rows if r["removed_total"] == r["added_total"]),
    }


def decide(summary: dict[str, Any]) -> str:
    if summary["color_counts_preserved"] == summary["example_count"]:
        return "counts_preserved_motion_task; mine deterministic destination rule per color/object"
    return "mixed_recolor_motion_task; split count-preserving and recolor cases before rule synthesis"


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp163_task025_motion_predicate_mining",
        "",
        "## 目的",
        "",
        "task025の差分を色別の削除/追加ペアとして見て、cell motionやrecolorの規則性を採掘する。",
        "",
        "## 結果",
        "",
        "```json",
        json.dumps(result["summary"], ensure_ascii=False, indent=2),
        "```",
        "",
        f"- decision: {result['decision']}",
        "",
        "## リスク",
        "",
        result["leakage_risk"],
        result["overfitting_risk"],
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def jsonable_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out = dict(summary)
    out["removed_added_recolor_counts"] = {str(k): v for k, v in summary["removed_added_recolor_counts"].items()}
    return out


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    splits = [("train", ex) for ex in task["train"]] + [("test", ex) for ex in task["test"]] + [("arc-gen", ex) for ex in task["arc-gen"]]
    rows = []
    for idx, (split, ex) in enumerate(splits):
        rows.append(example_features(idx, split, grid_to_array(ex["input"]), grid_to_array(ex["output"])))

    out_csv = EXP_DIR / "task025_motion_features.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = jsonable_summary(aggregate(rows))
    result = {
        "exp_id": "exp163_task025_motion_predicate_mining",
        "date": "2026-06-10",
        "status": "diagnostic_complete",
        "task_id": TASK_ID,
        "summary": summary,
        "outputs": {"motion_features": str(out_csv.relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "decision": decide(summary),
        "leakage_risk": "low: diagnostic over provided train/test/arc-gen examples only.",
        "overfitting_risk": "medium: mined motion signatures may overfit local generated examples unless converted to input-only rule.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
