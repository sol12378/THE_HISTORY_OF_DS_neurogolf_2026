from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task


EXP_ID = "exp052_task020_d4_orbit_rule"
EXP_DIR = ROOT / "experiments" / EXP_ID
TARGET_TASK = 20


@dataclass(frozen=True)
class EvalRow:
    example_idx: int
    split: str
    status: str
    target_color: int | str
    selected_orbit: str
    changed_count: int
    reason: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def d4_orbit(pos: tuple[int, int], center: tuple[int, int] = (2, 2)) -> frozenset[tuple[int, int]]:
    r, c = pos
    cr, cc = center
    dr, dc = r - cr, c - cc
    pts = {
        (cr + dr, cc + dc),
        (cr + dr, cc - dc),
        (cr - dr, cc + dc),
        (cr - dr, cc - dc),
        (cr + dc, cc + dr),
        (cr + dc, cc - dr),
        (cr - dc, cc + dr),
        (cr - dc, cc - dr),
    }
    return frozenset((rr, cc) for rr, cc in pts if 0 <= rr < 5 and 0 <= cc < 5)


def orbit_library() -> list[set[tuple[int, int]]]:
    out = {d4_orbit((r, c)) for r in range(5) for c in range(5)}
    return [set(x) for x in sorted(out, key=lambda s: (len(s), sorted(s)))]


def rel_positions(grid: np.ndarray, color: int, b: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, _, _ = b
    return {(int(r - r0), int(c - c0)) for r, c in np.argwhere(grid == color)}


def apply_rule(x: np.ndarray, orbits: list[set[tuple[int, int]]]) -> tuple[np.ndarray, int | str, str, str]:
    b = bbox(x != 0)
    if b is None:
        return x.copy(), "", "", "empty"
    r0, c0, r1, c1 = b
    if (r1 - r0, c1 - c0) != (5, 5):
        return x.copy(), "", "", f"bbox_shape={(r1-r0, c1-c0)}"
    candidates = []
    for color in [int(v) for v in np.unique(x) if int(v) != 0]:
        pos = rel_positions(x, color, b)
        center_extra = {(2, 2)} if (2, 2) in pos else set()
        for orbit in orbits:
            allowed = orbit | center_extra
            if not pos <= allowed:
                continue
            missing = orbit - pos
            if len(missing) != 3:
                continue
            ok = True
            for rr, cc in missing:
                if x[r0 + rr, c0 + cc] != 0:
                    ok = False
                    break
            if ok:
                candidates.append((color, orbit, missing))
    if len(candidates) != 1:
        return x.copy(), "", "", f"candidate_count={len(candidates)}"
    color, orbit, missing = candidates[0]
    out = x.copy()
    for rr, cc in missing:
        out[r0 + rr, c0 + cc] = color
    return out, color, str(sorted(orbit)), "ok"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TARGET_TASK)
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    orbits = orbit_library()
    rows: list[EvalRow] = []
    split_counts = Counter()
    split_pass = Counter()
    fail_reasons = Counter()
    for idx, ex in enumerate(examples):
        split = "train" if idx < train_n else ("test" if idx < train_n + test_n else "arc-gen")
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred, color, orbit, reason = apply_rule(x, orbits)
        ok = np.array_equal(pred, y)
        split_counts[split] += 1
        if ok:
            split_pass[split] += 1
        else:
            fail_reasons[reason] += 1
        rows.append(EvalRow(idx, split, "pass" if ok else "fail", color, orbit, int((pred != x).sum()), reason if not ok else "ok"))
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if all(row.status == "pass" for row in rows) else "partial",
        "target_task": TARGET_TASK,
        "orbit_count": len(orbits),
        "orbits": [sorted(list(orbit)) for orbit in orbits],
        "split_counts": dict(split_counts),
        "split_pass": dict(split_pass),
        "total_pass": sum(1 for row in rows if row.status == "pass"),
        "total_examples": len(rows),
        "fail_reasons": dict(fail_reasons),
        "decision": "If full pass, lower D4 orbit completion to ONNX; otherwise inspect failure reasons and add role constraints.",
        "leakage_risk": "low: rule is generated from D4 symmetry, not arc-gen label lookup.",
        "overfitting_risk": "low-to-medium: task-specific 5x5 orbit assumption.",
    }
    with (EXP_DIR / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (EXP_DIR / "eval_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(EvalRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([row.__dict__ for row in rows])
    notes = f"""# {EXP_ID}

## 仮説

task020は5x5 bbox中心まわりのD4 orbit completionで説明できる。

## 結果

- status: {result["status"]}
- pass: {result["total_pass"]}/{len(rows)}
- split pass: {dict(split_pass)} / {dict(split_counts)}
- fail reasons: {dict(fail_reasons)}

## 解釈

train template暗記ではなく、D4対称性から候補orbitを全列挙した。full passならsignature lookupを明示ruleへ置換できる。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
