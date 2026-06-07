from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import load_task  # noqa: E402


EXP_ID = "exp_b017_l2_changed_cell_feature_profile"
EXP_DIR = ROOT / "experiments" / EXP_ID
TAXONOMY_PATH = ROOT / "experiments" / "exp054_signature_lookup_family_taxonomy" / "task_taxonomy.csv"
FAST_TOP_K = 5


@dataclass(frozen=True)
class FeatureRow:
    task_id: int
    feature: str
    positive_top_values: str
    negative_top_values: str
    positive_unique: int
    negative_unique: int
    exact_positive_only_values: str
    exact_negative_only_values: str
    max_single_value_precision: float
    max_single_value_recall: float
    best_value: str


@dataclass(frozen=True)
class TaskProfile:
    task_id: int
    strict_cost: int
    gain_to_250: float
    train_examples: int
    positive_cells: int
    negative_cells: int
    output_color_counts: str
    changed_count_by_example: str
    best_feature: str
    best_value: str
    best_precision: float
    best_recall: float
    interpretation: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def load_targets() -> dict[int, dict[str, str]]:
    with TAXONOMY_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lane_rows = [r for r in rows if r["compiler_lane"] == "L2_local_predicate_sparse_fill"]
    lane_rows = sorted(lane_rows, key=lambda r: (-float(r["gain_to_250"]), int(r["task_id"])))
    return {int(r["task_id"]): r for r in lane_rows[:FAST_TOP_K]}


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def count_neighbors(x: np.ndarray, r: int, c: int, color: int, offsets: list[tuple[int, int]]) -> int:
    n = 0
    for dr, dc in offsets:
        rr, cc = r + dr, c + dc
        if 0 <= rr < x.shape[0] and 0 <= cc < x.shape[1] and int(x[rr, cc]) == color:
            n += 1
    return n


def sees(x: np.ndarray, r: int, c: int, color: int, dr: int, dc: int) -> int:
    rr, cc = r + dr, c + dc
    dist = 1
    while 0 <= rr < x.shape[0] and 0 <= cc < x.shape[1]:
        if x[rr, cc] != 0:
            return dist if int(x[rr, cc]) == color else -dist
        rr += dr
        cc += dc
        dist += 1
    return 0


ADJ4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG4 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL8 = ADJ4 + DIAG4
R2 = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if not (dr == 0 and dc == 0)]


def cell_features(x: np.ndarray, r: int, c: int, out_color: int) -> dict[str, str]:
    b_all = bbox(x != 0)
    b_color = bbox(x == out_color)
    feats: dict[str, str] = {
        "out_color": str(out_color),
        "r": str(r),
        "c": str(c),
        "shape": f"{x.shape[0]}x{x.shape[1]}",
        "adj4": str(count_neighbors(x, r, c, out_color, ADJ4)),
        "diag4": str(count_neighbors(x, r, c, out_color, DIAG4)),
        "all8": str(count_neighbors(x, r, c, out_color, ALL8)),
        "r2": str(count_neighbors(x, r, c, out_color, R2)),
        "ray_lr": f"{sees(x,r,c,out_color,0,-1)}:{sees(x,r,c,out_color,0,1)}",
        "ray_ud": f"{sees(x,r,c,out_color,-1,0)}:{sees(x,r,c,out_color,1,0)}",
        "ray_d1": f"{sees(x,r,c,out_color,-1,-1)}:{sees(x,r,c,out_color,1,1)}",
        "ray_d2": f"{sees(x,r,c,out_color,-1,1)}:{sees(x,r,c,out_color,1,-1)}",
    }
    if b_all is not None:
        r0, c0, r1, c1 = b_all
        feats.update(
            {
                "rel_all_r": str(r - r0),
                "rel_all_c": str(c - c0),
                "on_all_bbox_border": str(int(r in (r0, r1 - 1) or c in (c0, c1 - 1))),
                "inside_all_bbox": str(int(r0 <= r < r1 and c0 <= c < c1)),
            }
        )
    if b_color is not None:
        r0, c0, r1, c1 = b_color
        feats.update(
            {
                "rel_color_r": str(r - r0),
                "rel_color_c": str(c - c0),
                "on_color_bbox_border": str(int(r in (r0, r1 - 1) or c in (c0, c1 - 1))),
                "inside_color_bbox": str(int(r0 <= r < r1 and c0 <= c < c1)),
            }
        )
    return feats


def collect_task_rows(task_id: int) -> tuple[list[dict[str, str]], Counter, Counter]:
    task = load_task(task_id)
    rows: list[dict[str, str]] = []
    color_counts = Counter()
    changed_counts = Counter()
    for ex_idx, ex in enumerate(task["train"]):
        x = arr(ex["input"])
        y = arr(ex["output"])
        if x.shape != y.shape:
            continue
        changed = (x != y) & (x == 0)
        out_colors = [int(v) for v in np.unique(y[changed])] if np.any(changed) else []
        changed_counts[int(np.count_nonzero(changed))] += 1
        for color in out_colors:
            color_counts[color] += int(np.count_nonzero(changed & (y == color)))
        candidate_colors = out_colors or [int(v) for v in np.unique(y) if int(v) != 0]
        for r, c in np.argwhere(x == 0):
            for out_color in candidate_colors:
                label = int(changed[int(r), int(c)] and int(y[int(r), int(c)]) == out_color)
                feats = cell_features(x, int(r), int(c), out_color)
                feats.update({"task_id": str(task_id), "ex_idx": str(ex_idx), "label": str(label)})
                rows.append(feats)
    return rows, color_counts, changed_counts


def feature_stats(task_id: int, rows: list[dict[str, str]]) -> list[FeatureRow]:
    features = sorted(k for k in rows[0] if k not in {"task_id", "ex_idx", "label"}) if rows else []
    out: list[FeatureRow] = []
    pos_n = sum(1 for r in rows if r["label"] == "1")
    for feat in features:
        pos = Counter(r.get(feat, "") for r in rows if r["label"] == "1")
        neg = Counter(r.get(feat, "") for r in rows if r["label"] == "0")
        best_value = ""
        best_precision = 0.0
        best_recall = 0.0
        for val, pc in pos.items():
            nc = neg.get(val, 0)
            precision = pc / max(1, pc + nc)
            recall = pc / max(1, pos_n)
            if (precision, recall, pc) > (best_precision, best_recall, pos.get(best_value, 0)):
                best_value = val
                best_precision = precision
                best_recall = recall
        out.append(
            FeatureRow(
                task_id=task_id,
                feature=feat,
                positive_top_values=json.dumps(pos.most_common(8), ensure_ascii=False),
                negative_top_values=json.dumps(neg.most_common(8), ensure_ascii=False),
                positive_unique=len(pos),
                negative_unique=len(neg),
                exact_positive_only_values=json.dumps([v for v in pos if v not in neg][:20], ensure_ascii=False),
                exact_negative_only_values=json.dumps([v for v in neg if v not in pos][:20], ensure_ascii=False),
                max_single_value_precision=best_precision,
                max_single_value_recall=best_recall,
                best_value=best_value,
            )
        )
    return sorted(out, key=lambda r: (-r.max_single_value_precision, -r.max_single_value_recall, r.feature))


def interpretation(best: FeatureRow | None) -> str:
    if best is None:
        return "no feature rows"
    if best.max_single_value_precision >= 0.95 and best.max_single_value_recall >= 0.8:
        return "single feature may be enough"
    if best.max_single_value_precision >= 0.8:
        return "feature is useful but needs branch conjunction"
    return "needs multi-feature tree or different representation"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    all_feature_rows: list[FeatureRow] = []
    task_profiles: list[TaskProfile] = []
    for task_id, meta in targets.items():
        rows, color_counts, changed_counts = collect_task_rows(task_id)
        stats = feature_stats(task_id, rows)
        all_feature_rows.extend(stats)
        best = stats[0] if stats else None
        pos_cells = sum(1 for r in rows if r["label"] == "1")
        neg_cells = sum(1 for r in rows if r["label"] == "0")
        task_profiles.append(
            TaskProfile(
                task_id=task_id,
                strict_cost=int(float(meta["strict_cost"])),
                gain_to_250=float(meta["gain_to_250"]),
                train_examples=len(load_task(task_id)["train"]),
                positive_cells=pos_cells,
                negative_cells=neg_cells,
                output_color_counts=json.dumps(dict(color_counts), ensure_ascii=False),
                changed_count_by_example=json.dumps(dict(changed_counts), ensure_ascii=False),
                best_feature=best.feature if best else "",
                best_value=best.best_value if best else "",
                best_precision=best.max_single_value_precision if best else 0.0,
                best_recall=best.max_single_value_recall if best else 0.0,
                interpretation=interpretation(best),
            )
        )
    all_feature_rows = sorted(all_feature_rows, key=lambda r: (r.task_id, -r.max_single_value_precision, -r.max_single_value_recall, r.feature))
    task_profiles = sorted(task_profiles, key=lambda r: (-r.best_precision, -r.best_recall, -r.gain_to_250, r.task_id))
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "profile_ready",
        "target_task_count": len(targets),
        "fast_top_k": FAST_TOP_K,
        "target_task_ids": sorted(targets),
        "task_profiles": [asdict(r) for r in task_profiles],
        "top_feature_rows": [asdict(r) for r in all_feature_rows[:40]],
        "decision": "use feature profiles to build branch/tree miner; prioritize tasks with high precision single features",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: diagnostic only",
        "leakage_risk": "low: train feature diagnostics only.",
        "overfitting_risk": "medium: derived trees must pass full arc-gen before adoption.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "feature_stats.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(FeatureRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in all_feature_rows])
    with (EXP_DIR / "task_profiles.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TaskProfile.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in task_profiles])
    notes = f"""# {EXP_ID}

## 目的

L2 top taskのchanged cellをpositive/negative cell datasetとしてprofileし、branch/tree合成で使うべきfeatureを特定する。

## 結果

- target tasks: {len(targets)}
- task profiles: {len(task_profiles)}
- best profiles: {[asdict(r) for r in task_profiles[:3]]}

## 判断

高precision featureがあるtaskを優先してbranch/tree minerへ進む。単一featureで足りないtaskは、color-roleやcomponent featureを追加する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
