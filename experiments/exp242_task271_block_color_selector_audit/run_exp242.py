from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp242_task271_block_color_selector_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 271
H = 3
W = 3


def crop_at(x: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return x[r0 : r0 + H, c0 : c0 + W]


def full_block_offsets(x: np.ndarray) -> list[tuple[int, int]]:
    hits = []
    for rr in range(x.shape[0] - H + 1):
        for cc in range(x.shape[1] - W + 1):
            crop = crop_at(x, rr, cc)
            if np.all(crop != 0):
                hits.append((rr, cc))
    return hits


def exact_offset(x: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    hits = []
    for off in full_block_offsets(x):
        if np.array_equal(crop_at(x, *off), y):
            hits.append(off)
    if len(hits) != 1:
        raise ValueError(f"expected unique full-block hit, got {hits}")
    return hits[0]


def signature_by_frequency(crop: np.ndarray) -> tuple[int, ...]:
    counts = Counter(int(v) for v in crop.ravel())
    ordered = sorted(counts, key=lambda k: (-counts[k], k))
    mapping = {color: idx + 1 for idx, color in enumerate(ordered)}
    return tuple(mapping[int(v)] for v in crop.ravel())


def block_features(x: np.ndarray, off: tuple[int, int]) -> dict[str, Any]:
    crop = crop_at(x, *off)
    counts = Counter(int(v) for v in crop.ravel())
    vals = list(counts.values())
    mode_color = min(counts, key=lambda k: (-counts[k], k))
    least_color = min(counts, key=lambda k: (counts[k], k))
    return {
        "r": off[0],
        "c": off[1],
        "distinct": len(counts),
        "mode_count": counts[mode_color],
        "least_count": counts[least_color],
        "color8_count": counts.get(8, 0),
        "sum_colors": int(crop.sum()),
        "max_color": max(counts),
        "min_color": min(counts),
        "center": int(crop[1, 1]),
        "corner_sum": int(crop[0, 0] + crop[0, 2] + crop[2, 0] + crop[2, 2]),
        "edge_sum": int(crop[0, 1] + crop[1, 0] + crop[1, 2] + crop[2, 1]),
        "signature_freq": signature_by_frequency(crop),
        "count_pattern": tuple(sorted(vals, reverse=True)),
    }


def rank_selectors(cands: list[dict[str, Any]], target: tuple[int, int]) -> dict[str, bool]:
    selectors: dict[str, list[dict[str, Any]]] = {}
    for key in ("distinct", "mode_count", "least_count", "color8_count", "sum_colors", "max_color", "min_color", "center", "corner_sum", "edge_sum"):
        selectors[f"{key}_min"] = sorted(cands, key=lambda r: (r[key], r["r"], r["c"]))
        selectors[f"{key}_max"] = sorted(cands, key=lambda r: (-r[key], r["r"], r["c"]))
    selectors["row_major_first"] = sorted(cands, key=lambda r: (r["r"], r["c"]))
    selectors["row_major_last"] = sorted(cands, key=lambda r: (-r["r"], -r["c"]))
    selectors["col_major_first"] = sorted(cands, key=lambda r: (r["c"], r["r"]))
    selectors["col_major_last"] = sorted(cands, key=lambda r: (-r["c"], -r["r"]))
    return {name: (ordered[0]["r"], ordered[0]["c"]) == target for name, ordered in selectors.items()}


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()[TASK_ID]
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]

    rows: list[dict[str, Any]] = []
    selector_hits = Counter()
    target_feature_hists: dict[str, Counter[Any]] = {
        "signature_freq": Counter(),
        "count_pattern": Counter(),
        "position": Counter(),
        "distinct": Counter(),
        "mode_count": Counter(),
        "color8_count": Counter(),
    }
    candidate_feature_hists: dict[str, Counter[Any]] = {
        "signature_freq": Counter(),
        "count_pattern": Counter(),
    }

    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        target = exact_offset(x, y)
        offsets = full_block_offsets(x)
        feats = [block_features(x, off) for off in offsets]
        rank_hits = rank_selectors(feats, target)
        for name, ok in rank_hits.items():
            selector_hits[name] += int(ok)
        target_feat = next(f for f in feats if (f["r"], f["c"]) == target)
        target_feature_hists["position"][f"{target[0]},{target[1]}"] += 1
        for key in ("signature_freq", "count_pattern", "distinct", "mode_count", "color8_count"):
            target_feature_hists[key][target_feat[key]] += 1
        for f in feats:
            candidate_feature_hists["signature_freq"][f["signature_freq"]] += 1
            candidate_feature_hists["count_pattern"][f["count_pattern"]] += 1
        rows.append(
            {
                "idx": idx,
                "target": f"{target[0]},{target[1]}",
                "candidate_count": len(offsets),
                "target_features": json.dumps(target_feat, ensure_ascii=False),
                "rank_hits": json.dumps({k: v for k, v in rank_hits.items() if v}, ensure_ascii=False),
                "candidate_features": json.dumps(feats, ensure_ascii=False),
            }
        )

    ranked = sorted(
        [{"selector": k, "hit": int(v), "miss": len(examples) - int(v)} for k, v in selector_hits.items()],
        key=lambda r: (-r["hit"], r["selector"]),
    )
    with (EXP_DIR / "block_color_selector_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (EXP_DIR / "selector_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["selector", "hit", "miss"])
        writer.writeheader()
        writer.writerows(ranked)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "example_count": len(examples),
        "best_selectors": ranked[:30],
        "target_feature_hists": {k: [[str(a), int(b)] for a, b in c.most_common(20)] for k, c in target_feature_hists.items()},
        "candidate_signature_top": [[str(a), int(b)] for a, b in candidate_feature_hists["signature_freq"].most_common(20)],
        "candidate_count_pattern_top": [[str(a), int(b)] for a, b in candidate_feature_hists["count_pattern"].most_common(20)],
        "decision": "If a simple rank/color feature is full or near-full, validate a Python selector. Otherwise task271 needs richer relational selector.",
        "submission_decision": "no_submit: diagnostic only",
        "leakage_risk": "low.",
        "overfitting_risk": "medium: block selector may overfit if built from target frequency tables.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task271は各入力に3x3 full-nonzero blockが4個あり、そのうち1個が出力と一致する。4候補から正解blockを色構成rankで選べるか監査する。

## 結果

- baseline_cost: `{base.cost}`
- best_selectors: `{ranked[:10]}`
- target_feature_hists: `{ {k: c.most_common(5) for k, c in target_feature_hists.items()} }`

## 判断

単純rank/color特徴がfullまたはnear-fullならPython selector化する。弱い場合はtask271を一旦保留し、別候補へpivotする。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
