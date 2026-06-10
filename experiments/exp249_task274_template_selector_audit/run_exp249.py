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


EXP_ID = "exp249_task274_template_selector_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 274


def binary_template(y: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v != 0) for v in y.ravel())


def output_color(y: np.ndarray) -> int:
    vals = [int(v) for v in y.ravel() if int(v) != 0]
    return Counter(vals).most_common(1)[0][0] if vals else 0


def input_basic_features(x: np.ndarray) -> dict[str, int]:
    nz_counts = Counter(int(v) for v in x.ravel() if int(v) != 0)
    all_counts = Counter(int(v) for v in x.ravel())
    nz = np.argwhere(x != 0)
    feats: dict[str, int] = {
        "mode_nz": min(nz_counts, key=lambda k: (-nz_counts[k], k)) if nz_counts else 0,
        "least_nz": min(nz_counts, key=lambda k: (nz_counts[k], k)) if nz_counts else 0,
        "mode_all": min(all_counts, key=lambda k: (-all_counts[k], k)),
        "color_count": len(nz_counts),
        "center": int(x[x.shape[0] // 2, x.shape[1] // 2]),
        "tl": int(x[0, 0]),
        "tr": int(x[0, -1]),
        "bl": int(x[-1, 0]),
        "br": int(x[-1, -1]),
    }
    if len(nz):
        r0, c0 = nz.min(axis=0)
        r1, c1 = nz.max(axis=0)
        feats.update(
            {
                "bbox_h": int(r1 - r0 + 1),
                "bbox_w": int(c1 - c0 + 1),
                "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "bbox_tl": int(x[int(r0), int(c0)]),
                "bbox_tr": int(x[int(r0), int(c1)]),
                "bbox_bl": int(x[int(r1), int(c0)]),
                "bbox_br": int(x[int(r1), int(c1)]),
            }
        )
    return feats


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()[TASK_ID]
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    rows: list[dict[str, Any]] = []
    template_hist = Counter()
    color_hist = Counter()
    color_rule_hits = Counter()
    template_rule_hits = Counter()

    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        tpl = binary_template(y)
        color = output_color(y)
        template_hist[tpl] += 1
        color_hist[color] += 1
        feats = input_basic_features(x)
        for name, value in feats.items():
            color_rule_hits[name] += int(value == color)
        rows.append({"idx": idx, "template": tpl, "color": color, "features": feats})

    templates = [tpl for tpl, _ in template_hist.most_common()]
    template_id = {tpl: i for i, tpl in enumerate(templates)}
    for row in rows:
        tid = template_id[row["template"]]
        for name, value in row["features"].items():
            template_rule_hits[name] += int(value == tid)

    color_ranked = sorted(
        [{"rule": k, "pass_count": int(v), "fail_count": len(examples) - int(v)} for k, v in color_rule_hits.items()],
        key=lambda r: (-r["pass_count"], r["rule"]),
    )
    template_ranked = sorted(
        [{"rule": k, "pass_count": int(v), "fail_count": len(examples) - int(v)} for k, v in template_rule_hits.items()],
        key=lambda r: (-r["pass_count"], r["rule"]),
    )
    with (EXP_DIR / "template_selector_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "template", "color", "features"])
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "example_count": len(examples),
        "template_hist": [[str(k), int(v)] for k, v in template_hist.most_common()],
        "output_color_hist": dict(sorted(color_hist.items())),
        "best_color_rules": color_ranked[:20],
        "best_template_rules": template_ranked[:20],
        "decision": "If template and color selectors are simple/full, build static 3x3 probe. Otherwise hold task274.",
        "submission_decision": "no_submit: audit only",
        "leakage_risk": "low.",
        "overfitting_risk": "medium.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task274は3x3固定・binary signature 4種類なので、templateと色selectorが単純特徴で決まるか監査する。

## 結果

- baseline_cost: `{base.cost}`
- template_hist: `{template_hist.most_common()}`
- output_color_hist: `{dict(sorted(color_hist.items()))}`
- best_color_rules: `{color_ranked[:10]}`
- best_template_rules: `{template_ranked[:10]}`

## 判断

template/color selectorがfullならstatic 3x3 probeへ進む。弱ければ保留。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
