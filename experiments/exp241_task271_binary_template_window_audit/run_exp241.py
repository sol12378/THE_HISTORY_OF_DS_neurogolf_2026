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


EXP_ID = "exp241_task271_binary_template_window_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 271
H = 3
W = 3


def crop_at(x: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return x[r0 : r0 + H, c0 : c0 + W]


def binary_mask(a: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v != 0) for v in a.ravel())


def color_signature(a: np.ndarray) -> tuple[int, ...]:
    vals = [int(v) for v in a.ravel()]
    nonzero = sorted({v for v in vals if v != 0})
    mapping = {0: 0}
    for idx, color in enumerate(nonzero, start=1):
        mapping[color] = idx
    return tuple(mapping[v] for v in vals)


def exact_offset(x: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    hits = []
    for rr in range(x.shape[0] - H + 1):
        for cc in range(x.shape[1] - W + 1):
            if np.array_equal(crop_at(x, rr, cc), y):
                hits.append((rr, cc))
    if len(hits) != 1:
        raise ValueError(f"expected unique hit, got {hits}")
    return hits[0]


def candidate_windows(x: np.ndarray, template: tuple[int, ...], mode: str) -> list[tuple[int, int]]:
    hits = []
    for rr in range(x.shape[0] - H + 1):
        for cc in range(x.shape[1] - W + 1):
            crop = crop_at(x, rr, cc)
            if mode == "binary" and binary_mask(crop) == template:
                hits.append((rr, cc))
            if mode == "signature" and color_signature(crop) == template:
                hits.append((rr, cc))
    return hits


def candidate_rank_features(cands: list[tuple[int, int]], target: tuple[int, int]) -> dict[str, bool]:
    if target not in cands:
        return {}
    ordered = {
        "first_row_major": sorted(cands),
        "last_row_major": sorted(cands, reverse=True),
        "first_col_major": sorted(cands, key=lambda p: (p[1], p[0])),
        "last_col_major": sorted(cands, key=lambda p: (p[1], p[0]), reverse=True),
        "closest_tl": sorted(cands, key=lambda p: (p[0] + p[1], p[0], p[1])),
        "closest_br": sorted(cands, key=lambda p: (-(p[0] + p[1]), -p[0], -p[1])),
    }
    return {name: vals[0] == target for name, vals in ordered.items()}


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()[TASK_ID]
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    y0 = grid_to_array(examples[0]["output"])
    binary_template = binary_mask(y0)
    signature_template = color_signature(y0)

    rows: list[dict[str, Any]] = []
    counters = Counter()
    count_hists = Counter()
    rank_hits = Counter()
    rank_totals = Counter()

    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        target = exact_offset(x, y)
        binary_cands = candidate_windows(x, binary_template, "binary")
        signature_cands = candidate_windows(x, signature_template, "signature")
        counters["target_in_binary_cands"] += int(target in binary_cands)
        counters["target_in_signature_cands"] += int(target in signature_cands)
        counters["binary_unique_and_target"] += int(len(binary_cands) == 1 and target in binary_cands)
        counters["signature_unique_and_target"] += int(len(signature_cands) == 1 and target in signature_cands)
        count_hists[f"binary_{len(binary_cands)}"] += 1
        count_hists[f"signature_{len(signature_cands)}"] += 1
        for prefix, cands in (("binary", binary_cands), ("signature", signature_cands)):
            feats = candidate_rank_features(cands, target)
            for name, ok in feats.items():
                rank_totals[f"{prefix}_{name}"] += 1
                rank_hits[f"{prefix}_{name}"] += int(ok)
        rows.append(
            {
                "idx": idx,
                "target": f"{target[0]},{target[1]}",
                "binary_candidate_count": len(binary_cands),
                "signature_candidate_count": len(signature_cands),
                "target_in_binary": target in binary_cands,
                "target_in_signature": target in signature_cands,
                "binary_candidates": ";".join(f"{r},{c}" for r, c in binary_cands[:30]),
                "signature_candidates": ";".join(f"{r},{c}" for r, c in signature_cands[:30]),
            }
        )

    ranked = sorted(
        [
            {
                "selector": name,
                "hit": int(rank_hits[name]),
                "eligible": int(rank_totals[name]),
                "miss": len(examples) - int(rank_hits[name]),
            }
            for name in rank_totals
        ],
        key=lambda r: (-r["hit"], r["selector"]),
    )
    with (EXP_DIR / "binary_template_window_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (EXP_DIR / "template_selector_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["selector", "hit", "eligible", "miss"])
        writer.writeheader()
        writer.writerows(ranked)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "example_count": len(examples),
        "binary_template": binary_template,
        "signature_template": signature_template,
        "counters": dict(counters),
        "candidate_count_hists": dict(sorted(count_hists.items())),
        "best_rank_selectors": ranked[:20],
        "decision": "If binary/signature template is unique or rank-selectable, build a template-window rule and cost probe. Otherwise inspect residual selector.",
        "submission_decision": "no_submit: diagnostic only",
        "leakage_risk": "low.",
        "overfitting_risk": "medium: rank rules must be structural, not table-like.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task271の出力は固定binary signatureを持つ3x3 exact cropだったため、固定binary/color-signature templateに一致する入力windowを選べるか監査する。

## 結果

- baseline_cost: `{base.cost}`
- binary_template: `{binary_template}`
- signature_template: `{signature_template}`
- counters: `{dict(counters)}`
- candidate_count_hists: `{dict(sorted(count_hists.items()))}`
- best_rank_selectors: `{ranked[:10]}`

## 判断

template一致windowが一意、または単純rankで選べるならrule/cost probeへ進む。複数候補が多くrankも弱い場合はselectorを追加で掘る。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
