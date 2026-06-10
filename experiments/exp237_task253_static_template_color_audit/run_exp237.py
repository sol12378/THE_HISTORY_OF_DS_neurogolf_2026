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


EXP_ID = "exp237_task253_static_template_color_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 253


def output_color_and_mask(y: np.ndarray) -> tuple[int, tuple[int, ...]]:
    vals = [int(v) for v in y.ravel() if int(v) != 0]
    color = Counter(vals).most_common(1)[0][0] if vals else 0
    mask = tuple(int(v != 0) for v in y.ravel())
    return color, mask


def input_features(x: np.ndarray) -> dict[str, int]:
    nz_counts = Counter(int(v) for v in x.ravel() if int(v) != 0)
    all_counts = Counter(int(v) for v in x.ravel())
    corners = [int(x[0, 0]), int(x[0, -1]), int(x[-1, 0]), int(x[-1, -1])]
    feats: dict[str, int] = {
        "mode_nz": min(nz_counts, key=lambda k: (-nz_counts[k], k)) if nz_counts else 0,
        "least_nz": min(nz_counts, key=lambda k: (nz_counts[k], k)) if nz_counts else 0,
        "mode_all": min(all_counts, key=lambda k: (-all_counts[k], k)),
        "center": int(x[x.shape[0] // 2, x.shape[1] // 2]),
        "tl": corners[0],
        "tr": corners[1],
        "bl": corners[2],
        "br": corners[3],
    }
    nz = np.argwhere(x != 0)
    if nz.size:
        r0, c0 = nz.min(axis=0)
        r1, c1 = nz.max(axis=0)
        feats.update(
            {
                "bbox_tl": int(x[int(r0), int(c0)]),
                "bbox_tr": int(x[int(r0), int(c1)]),
                "bbox_bl": int(x[int(r1), int(c0)]),
                "bbox_br": int(x[int(r1), int(c1)]),
                "bbox_center": int(x[(int(r0) + int(r1)) // 2, (int(c0) + int(c1)) // 2]),
            }
        )
    return feats


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()[TASK_ID]
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    rows: list[dict[str, Any]] = []
    mask_hist = Counter()
    color_hist = Counter()
    pass_counts = Counter()
    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        color, mask = output_color_and_mask(y)
        mask_hist[mask] += 1
        color_hist[color] += 1
        feats = input_features(x)
        for name, pred in feats.items():
            pass_counts[name] += int(pred == color)
        rows.append({"idx": idx, "output_color": color, "features": feats})

    ranked = sorted(
        [{"rule": name, "pass_count": int(count), "fail_count": len(examples) - int(count)} for name, count in pass_counts.items()],
        key=lambda r: (-r["pass_count"], r["rule"]),
    )
    with (EXP_DIR / "color_rule_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rule", "pass_count", "fail_count"])
        writer.writeheader()
        writer.writerows(ranked)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "example_count": len(examples),
        "output_shape": list(grid_to_array(examples[0]["output"]).shape),
        "binary_signature_count": len(mask_hist),
        "most_common_mask_count": mask_hist.most_common(1)[0][1],
        "output_color_hist": dict(sorted(color_hist.items())),
        "best_color_rules": ranked[:20],
        "sample_rows": rows[:20],
        "decision": "If a simple color rule is full, build static 4x4 template ONNX. Otherwise inspect color selector.",
        "submission_decision": "no_submit: audit only",
        "leakage_risk": "low",
        "overfitting_risk": "medium-low",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task253は4x4固定binary templateに見えるため、出力色が入力の簡単な特徴で決まるか監査する。

## 結果

- baseline_cost: `{base.cost}`
- binary_signature_count: `{len(mask_hist)}`
- best_color_rules: `{ranked[:10]}`

## 判断

simple color ruleがfullならstatic template ONNXへ進む。なければcolor selectorを別途監査する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
