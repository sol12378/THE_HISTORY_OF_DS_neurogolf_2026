from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
from collections import Counter
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp200_task185_axis_separable_selector_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185
CURRENT_PUBLIC_BEST_LB = 6005.93


def grid_color(arr: np.ndarray) -> int:
    vals = [int(v) for v in arr.ravel() if int(v) != 0]
    return Counter(vals).most_common(1)[0][0]


def grid_lines(arr: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    h, w = arr.shape
    rows = [r for r in range(h) if int((arr[r, :] == bg).sum()) > w * 0.8]
    cols = [c for c in range(w) if int((arr[:, c] == bg).sum()) > h * 0.8]
    return rows, cols


def windows(lines: list[int]) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for i in range(len(lines) - 3):
        win = tuple(lines[i : i + 4])
        delta = [win[j + 1] - win[j] for j in range(3)]
        if len(set(delta)) == 1 and delta[0] in (3, 4, 5):
            out.append(win)
    return out


def pairwise_select(
    arr: np.ndarray, bg: int, row_windows: list[tuple[int, ...]], col_windows: list[tuple[int, ...]]
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]] | None:
    best: tuple[int, int, tuple[int, ...], tuple[int, ...]] | None = None
    for rr in row_windows:
        for cc in col_windows:
            vals = [int(arr[r, c]) for r in rr for c in cc]
            score = sum(1 for v in vals if v not in (0, bg))
            filled_rows = sum(any(int(arr[r, c]) not in (0, bg) for c in cc) for r in rr)
            filled_cols = sum(any(int(arr[r, c]) not in (0, bg) for r in rr) for c in cc)
            tie = filled_rows + filled_cols
            candidate = (score, tie, rr, cc)
            if best is None or (score, tie, -rr[0], -cc[0]) > (best[0], best[1], -best[2][0], -best[3][0]):
                best = candidate
    if best is None:
        return None
    score, tie, rr, cc = best
    return rr, cc, {"pair_score": score, "pair_tie": tie}


def axis_scores(
    arr: np.ndarray,
    bg: int,
    row_windows: list[tuple[int, ...]],
    col_windows: list[tuple[int, ...]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    row_stats: list[dict[str, Any]] = []
    col_stats: list[dict[str, Any]] = []
    for rr in row_windows:
        mask = arr[np.array(rr), :]
        special = (mask != 0) & (mask != bg)
        row_stats.append(
            {
                "win": rr,
                "score": int(special.sum()),
                "active_lines": int(np.any(special, axis=1).sum()),
                "start": rr[0],
            }
        )
    for cc in col_windows:
        mask = arr[:, np.array(cc)]
        special = (mask != 0) & (mask != bg)
        col_stats.append(
            {
                "win": cc,
                "score": int(special.sum()),
                "active_lines": int(np.any(special, axis=0).sum()),
                "start": cc[0],
            }
        )
    return row_stats, col_stats


def choose_axis(stats: list[dict[str, Any]], mode: str) -> tuple[int, ...] | None:
    if not stats:
        return None
    if mode == "score":
        key = lambda s: (s["score"], -s["start"])
    elif mode == "score_active":
        key = lambda s: (s["score"], s["active_lines"], -s["start"])
    elif mode == "active_score":
        key = lambda s: (s["active_lines"], s["score"], -s["start"])
    elif mode == "rightmost_score":
        key = lambda s: (s["score"], s["start"])
    else:
        raise ValueError(mode)
    return tuple(max(stats, key=key)["win"])


def render_output(arr: np.ndarray, bg: int, rr: tuple[int, ...], cc: tuple[int, ...]) -> np.ndarray:
    mat = np.zeros((4, 4), dtype=np.int64)
    for i, r in enumerate(rr):
        for j, c in enumerate(cc):
            v = int(arr[r, c])
            mat[i, j] = 0 if v in (0, bg) else v
    out = np.zeros((3, 3), dtype=np.int64)
    for r in range(3):
        for c in range(3):
            block = mat[r : r + 2, c : c + 2]
            vals = set(int(v) for v in block.ravel())
            if len(vals) == 1 and 0 not in vals:
                out[r, c] = int(block[0, 0])
    return out


def main() -> None:
    t0 = time.perf_counter()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    base = load_base_tasks()[TASK_ID]
    modes = ["score", "score_active", "active_score", "rightmost_score"]
    mode_pass = {mode: 0 for mode in modes}
    mode_match_pair = {mode: 0 for mode in modes}
    pair_pass = 0
    rows: list[dict[str, Any]] = []
    fails: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    row_target_rank_hist: Counter[int] = Counter()
    col_target_rank_hist: Counter[int] = Counter()

    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        bg = grid_color(x)
        line_rows, line_cols = grid_lines(x, bg)
        row_windows = windows(line_rows)
        col_windows = windows(line_cols)
        selected = pairwise_select(x, bg, row_windows, col_windows)
        if selected is None:
            rows.append({"idx": idx, "status": "no_pair", "shape": f"{x.shape[0]}x{x.shape[1]}"})
            continue
        pair_rr, pair_cc, pair_meta = selected
        pair_pred = render_output(x, bg, pair_rr, pair_cc)
        pair_ok = np.array_equal(pair_pred, y)
        pair_pass += int(pair_ok)

        row_stats, col_stats = axis_scores(x, bg, row_windows, col_windows)
        ranked_rows = sorted(row_stats, key=lambda s: (s["score"], s["active_lines"], -s["start"]), reverse=True)
        ranked_cols = sorted(col_stats, key=lambda s: (s["score"], s["active_lines"], -s["start"]), reverse=True)
        row_target_rank_hist[next((i for i, s in enumerate(ranked_rows) if tuple(s["win"]) == pair_rr), -1)] += 1
        col_target_rank_hist[next((i for i, s in enumerate(ranked_cols) if tuple(s["win"]) == pair_cc), -1)] += 1

        row: dict[str, Any] = {
            "idx": idx,
            "shape": f"{x.shape[0]}x{x.shape[1]}",
            "pair_ok": pair_ok,
            "pair_rr": ",".join(map(str, pair_rr)),
            "pair_cc": ",".join(map(str, pair_cc)),
            "pair_score": pair_meta["pair_score"],
            "row_window_count": len(row_windows),
            "col_window_count": len(col_windows),
        }
        for mode in modes:
            rr = choose_axis(row_stats, mode)
            cc = choose_axis(col_stats, mode)
            ok = rr is not None and cc is not None and np.array_equal(render_output(x, bg, rr, cc), y)
            match_pair = rr == pair_rr and cc == pair_cc
            mode_pass[mode] += int(ok)
            mode_match_pair[mode] += int(match_pair)
            row[f"{mode}_ok"] = ok
            row[f"{mode}_rr"] = "" if rr is None else ",".join(map(str, rr))
            row[f"{mode}_cc"] = "" if cc is None else ",".join(map(str, cc))
            if not ok and len(fails[mode]) < 8:
                fails[mode].append(
                    {
                        "idx": idx,
                        "expected": y.tolist(),
                        "pred": None if rr is None or cc is None else render_output(x, bg, rr, cc).tolist(),
                        "pair_rr": pair_rr,
                        "pair_cc": pair_cc,
                        "axis_rr": rr,
                        "axis_cc": cc,
                    }
                )
        rows.append(row)

    with (EXP_DIR / "axis_selector_audit.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_mode = max(modes, key=lambda m: mode_pass[m])
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "audit_complete",
        "task_id": TASK_ID,
        "hypothesis": "task185の4x4 lattice selectorはrow/col windowを独立scoreで選べれば、pairwise window scoringを避けられる。",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "current_public_best_lb": CURRENT_PUBLIC_BEST_LB,
        "example_count": len(examples),
        "pairwise_reference_pass": pair_pass,
        "mode_pass": mode_pass,
        "mode_match_pair": mode_match_pair,
        "best_axis_mode": best_mode,
        "best_axis_pass": mode_pass[best_mode],
        "row_target_rank_hist": dict(sorted(row_target_rank_hist.items())),
        "col_target_rank_hist": dict(sorted(col_target_rank_hist.items())),
        "fail_examples": fails[best_mode],
        "decision": (
            "If best_axis_pass is full, implement cheaper axis-separable ONNX selector; "
            "otherwise task185 still needs pairwise window interaction or another target."
        ),
        "submission_decision": "no_submit: selector audit only",
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "leakage_risk": "low: input-only geometry audit.",
        "overfitting_risk": "medium-low: independent scoring is a structural simplification; no raw coordinate table emitted.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task185のpairwise window scoringが高costだったため、row/col windowを独立にscoreして同じselectorを再現できるか確認する。

## 結果

- pairwise_reference_pass: `{pair_pass}/{len(examples)}`
- best_axis_mode: `{best_mode}`
- best_axis_pass: `{mode_pass[best_mode]}/{len(examples)}`
- mode_pass: `{mode_pass}`

## 判断

full passならaxis-separable selectorとしてONNX化へ進む。未達ならpairwise相互作用が必要で、task185は別の低cost selector案か他taskへpivotする。

## リスク

- leakage risk: low。入力geometryだけを使う監査。
- overfitting risk: medium-low。raw coordinate tableではないが、selector tie-breakはarc-gen分布に依存しうる。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
