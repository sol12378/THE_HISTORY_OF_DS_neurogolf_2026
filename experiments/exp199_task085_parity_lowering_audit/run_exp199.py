from __future__ import annotations

import json
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, grid_to_array, load_task


EXP_DIR = ROOT / "experiments" / "exp199_task085_parity_lowering_audit"
TASK_ID = 85


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def rule_relative(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for color in [int(v) for v in np.unique(x) if int(v) != 0]:
        for comp in components(x == color):
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            r0, r1 = min(rs), max(rs) + 1
            c0, c1 = min(cs), max(cs) + 1
            if r1 - r0 != 3:
                continue
            if len(comp) != 3 * (c1 - c0):
                continue
            mid = r0 + 1
            for c in range(c0, c1):
                if (c - c0) % 2 == 1:
                    out[mid, c] = 0
    return out


def rule_global(x: np.ndarray, erase_col_parity: int) -> np.ndarray:
    out = x.copy()
    for color in [int(v) for v in np.unique(x) if int(v) != 0]:
        for comp in components(x == color):
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            r0, r1 = min(rs), max(rs) + 1
            c0, c1 = min(cs), max(cs) + 1
            if r1 - r0 != 3:
                continue
            if len(comp) != 3 * (c1 - c0):
                continue
            mid = r0 + 1
            for c in range(c0, c1):
                if c % 2 == erase_col_parity:
                    out[mid, c] = 0
    return out


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = examples_for(task, -1)
    left_parity = Counter()
    width_hist = Counter()
    bar_count_hist = Counter()
    global_pass = {0: 0, 1: 0}
    relative_pass = 0
    for ex in examples:
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        bars = 0
        for color in [int(v) for v in np.unique(x) if int(v) != 0]:
            for comp in components(x == color):
                rs = [r for r, _ in comp]
                cs = [c for _, c in comp]
                r0, r1 = min(rs), max(rs) + 1
                c0, c1 = min(cs), max(cs) + 1
                if r1 - r0 == 3 and len(comp) == 3 * (c1 - c0):
                    bars += 1
                    left_parity[c0 % 2] += 1
                    width_hist[c1 - c0] += 1
        bar_count_hist[bars] += 1
        if np.array_equal(rule_relative(x), y):
            relative_pass += 1
        for parity in [0, 1]:
            if np.array_equal(rule_global(x, parity), y):
                global_pass[parity] += 1
    result = {
        "exp_id": "exp199_task085_parity_lowering_audit",
        "date": "2026-06-10",
        "status": "audit_complete",
        "task_id": TASK_ID,
        "example_count": len(examples),
        "relative_rule_pass": relative_pass,
        "global_parity_pass": global_pass,
        "left_parity_hist": dict(left_parity),
        "width_hist": dict(width_hist),
        "bar_count_hist": dict(bar_count_hist),
        "decision": "If one global parity passes all examples, lower with fixed checkerboard; otherwise run-left parity detection is required.",
        "elapsed_s": round(time.time() - t0, 3),
        "leakage_risk": "low: structural audit only.",
        "overfitting_risk": "low: no model emitted.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXP_DIR / "notes.md").write_text(
        "# exp199_task085_parity_lowering_audit\n\n"
        "## 目的\n\n"
        "task085のrelative parity ruleがglobal checkerboard parityで近似可能か確認する。\n\n"
        "## 結果\n\n"
        f"- relative_rule_pass: `{relative_pass}/{len(examples)}`\n"
        f"- global_parity_pass: `{global_pass}`\n"
        f"- left_parity_hist: `{dict(left_parity)}`\n"
        f"- width_hist: `{dict(width_hist)}`\n\n"
        "## 判断\n\n"
        + result["decision"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
