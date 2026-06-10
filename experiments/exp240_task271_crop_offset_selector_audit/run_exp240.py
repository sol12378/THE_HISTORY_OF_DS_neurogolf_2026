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


EXP_ID = "exp240_task271_crop_offset_selector_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 271
H = 3
W = 3


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
        comps.append(
            {
                "color": color,
                "size": len(cells),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "h": int(r1 - r0),
                "w": int(c1 - c0),
                "area": int((r1 - r0) * (c1 - c0)),
            }
        )
    return comps


def crop_at(x: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return x[r0 : r0 + H, c0 : c0 + W]


def exact_offsets(x: np.ndarray, y: np.ndarray) -> list[tuple[int, int]]:
    offsets = []
    for rr in range(x.shape[0] - H + 1):
        for cc in range(x.shape[1] - W + 1):
            if np.array_equal(crop_at(x, rr, cc), y):
                offsets.append((rr, cc))
    return offsets


def bbox_offset(x: np.ndarray, mask: np.ndarray) -> tuple[int, int] | None:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return None
    r0, c0 = cells.min(axis=0)
    return int(r0), int(c0)


def clamp_offset(r: int, c: int, x: np.ndarray) -> tuple[int, int]:
    return max(0, min(int(r), x.shape[0] - H)), max(0, min(int(c), x.shape[1] - W))


def candidate_offsets(x: np.ndarray) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    nz = x != 0
    if (off := bbox_offset(x, nz)) is not None:
        out["nz_bbox_tl"] = off
    counts = Counter(int(v) for v in x.ravel() if int(v) != 0)
    if counts:
        mode = min(counts, key=lambda k: (-counts[k], k))
        least = min(counts, key=lambda k: (counts[k], k))
        for name, color in (("mode_color", mode), ("least_color", least), ("color8", 8)):
            if np.any(x == color):
                out[f"{name}_bbox_tl"] = bbox_offset(x, x == color)  # type: ignore[assignment]
                cells = np.argwhere(x == color)
                rc_mean = np.rint(cells.mean(axis=0)).astype(int)
                out[f"{name}_centered"] = clamp_offset(int(rc_mean[0]) - 1, int(rc_mean[1]) - 1, x)
    comps = components(x)
    rank_specs = [
        ("size_desc", lambda c: (-c["size"], c["color"], c["r0"], c["c0"])),
        ("area_desc", lambda c: (-c["area"], c["color"], c["r0"], c["c0"])),
        ("r0_asc", lambda c: (c["r0"], c["c0"], c["color"])),
        ("c0_asc", lambda c: (c["c0"], c["r0"], c["color"])),
        ("color_asc", lambda c: (c["color"], c["r0"], c["c0"])),
    ]
    for name, key_fn in rank_specs:
        if comps:
            comp = sorted(comps, key=key_fn)[0]
            out[f"comp_{name}_tl"] = (comp["r0"], comp["c0"])
            out[f"comp_{name}_clamped"] = clamp_offset(comp["r0"], comp["c0"], x)
    return {k: v for k, v in out.items() if v is not None}


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()[TASK_ID]
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]

    rows: list[dict[str, Any]] = []
    offset_hist = Counter()
    exact_count_hist = Counter()
    selector_hits = Counter()
    selector_close = Counter()

    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        offsets = exact_offsets(x, y)
        exact_count_hist[len(offsets)] += 1
        for off in offsets:
            offset_hist[f"{off[0]},{off[1]}"] += 1
        cands = candidate_offsets(x)
        for name, off in cands.items():
            selector_hits[name] += int(off in offsets)
            if offsets:
                dist = min(abs(off[0] - rr) + abs(off[1] - cc) for rr, cc in offsets)
                selector_close[name] += int(dist <= 1)
        rows.append(
            {
                "idx": idx,
                "exact_offsets": ";".join(f"{r},{c}" for r, c in offsets),
                "exact_offset_count": len(offsets),
                "candidate_offsets": json.dumps(cands, ensure_ascii=False),
                "input_shape": f"{x.shape[0]}x{x.shape[1]}",
            }
        )

    ranked = sorted(
        [{"selector": k, "hit": int(v), "miss": len(examples) - int(v), "within1": int(selector_close[k])} for k, v in selector_hits.items()],
        key=lambda r: (-r["hit"], -r["within1"], r["selector"]),
    )
    with (EXP_DIR / "crop_offset_selector_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (EXP_DIR / "selector_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["selector", "hit", "miss", "within1"])
        writer.writeheader()
        writer.writerows(ranked)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "example_count": len(examples),
        "exact_offset_count_hist": dict(sorted(exact_count_hist.items())),
        "top_exact_offsets": offset_hist.most_common(30),
        "best_selectors": ranked[:30],
        "decision": "If a selector is full or close with simple residual rule, build Python rule and cost probe. Otherwise task271 crop selector remains unresolved.",
        "submission_decision": "no_submit: diagnostic only",
        "leakage_risk": "low: input/output structural audit.",
        "overfitting_risk": "medium: offset selector may overfit if expressed as table.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task271は出力が全例で入力内の3x3 exact cropだったため、そのoffsetを単純な入力特徴で選べるか監査する。

## 結果

- baseline_cost: `{base.cost}`
- exact_offset_count_hist: `{dict(sorted(exact_count_hist.items()))}`
- top_exact_offsets: `{offset_hist.most_common(10)}`
- best_selectors: `{ranked[:10]}`

## 判断

fullまたはnear-fullのselectorがあればPython rule化とcost probeへ進む。なければtask271はcrop offset selector未解決として保留する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
