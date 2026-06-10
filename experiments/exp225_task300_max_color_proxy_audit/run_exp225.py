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


EXP_ID = "exp225_task300_max_color_proxy_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 300


def components(x: np.ndarray) -> list[dict[str, Any]]:
    seen = np.zeros(x.shape, dtype=bool)
    comps: list[dict[str, Any]] = []
    for r, c in np.argwhere(x != 0):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        color = int(x[r, c])
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            cells.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < x.shape[0] and 0 <= nc < x.shape[1] and not seen[nr, nc] and int(x[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        arr = np.asarray(cells)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0) + 1
        comps.append({"color": color, "size": len(cells), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1)})
    return comps


def max_color_count_rule(x: np.ndarray) -> tuple[int, int, int, int, int]:
    counts = Counter(int(v) for v in x.ravel() if int(v) != 0)
    color = min(counts, key=lambda k: (-counts[k], k))
    cells = np.argwhere(x == color)
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0) + 1
    return int(color), int(r0), int(c0), int(r1), int(c1)


def max_color_mask4x3(x: np.ndarray) -> np.ndarray:
    color, r0, c0, _, _ = max_color_count_rule(x)
    out = np.zeros((4, 3), dtype=np.int64)
    crop = x[r0 : r0 + 4, c0 : c0 + 3]
    h, w = crop.shape
    out[:h, :w] = np.where(crop == color, color, 0)
    return out


def padded_equal(pred4x3: np.ndarray, y: np.ndarray) -> bool:
    padded = np.zeros((30, 30), dtype=np.int64)
    padded[: pred4x3.shape[0], : pred4x3.shape[1]] = pred4x3
    target = np.zeros((30, 30), dtype=np.int64)
    target[: y.shape[0], : y.shape[1]] = y
    return bool(np.array_equal(padded, target))


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()[TASK_ID]
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    rows: list[dict[str, Any]] = []
    pass_count = 0
    max_color_matches_component = 0
    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        comps = components(x)
        largest = sorted(comps, key=lambda c: (-c["size"], c["color"], c["r0"], c["c0"]))[0]
        color, r0, c0, r1, c1 = max_color_count_rule(x)
        pred = max_color_mask4x3(x)
        ok = padded_equal(pred, y)
        pass_count += int(ok)
        same_component = color == largest["color"] and r0 == largest["r0"] and c0 == largest["c0"] and r1 == largest["r1"] and c1 == largest["c1"]
        max_color_matches_component += int(same_component)
        rows.append(
            {
                "idx": idx,
                "ok": ok,
                "same_component": same_component,
                "output_shape": f"{y.shape[0]}x{y.shape[1]}",
                "max_color": color,
                "max_color_bbox": f"{r0},{c0},{r1},{c1}",
                "largest_color": largest["color"],
                "largest_size": largest["size"],
                "largest_bbox": f"{largest['r0']},{largest['c0']},{largest['r1']},{largest['c1']}",
            }
        )

    with (EXP_DIR / "max_color_proxy_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "example_count": len(examples),
        "max_color_matches_component": max_color_matches_component,
        "max_color_mask4x3_pass_count": pass_count,
        "max_color_mask4x3_fail_count": len(examples) - pass_count,
        "fail_examples": [r for r in rows if not r["ok"]][:20],
        "decision": "If 267/267, implement ONNX max-color ArgMax + bbox mask4x3 cost probe. Otherwise component selection is genuinely needed.",
        "submission_decision": "no_submit: audit only",
        "leakage_risk": "low: input-only proxy audit.",
        "overfitting_risk": "low: no fitted thresholds.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task300の最大component crop ruleを、より安い最大nonzero色count + 最大4x3 mask出力で代替できるか確認する。

## 結果

- baseline_cost: `{base.cost}`
- max_color_matches_component: `{max_color_matches_component}/{len(examples)}`
- max_color_mask4x3: `{pass_count}/{len(examples)}`

## 判断

267/267ならcomponent growthを避けてONNX cost probeへ進む。失敗するなら最大component選択が本当に必要。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
