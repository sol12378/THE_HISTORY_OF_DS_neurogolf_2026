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


EXP_ID = "exp051_task020_orbit_completion_miner"
EXP_DIR = ROOT / "experiments" / EXP_ID
TARGET_TASK = 20


@dataclass(frozen=True)
class EvalRow:
    example_idx: int
    split: str
    status: str
    target_color: int | str
    selected_template: str
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


def rel_positions(grid: np.ndarray, color: int, b: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, _, _ = b
    return {(int(r - r0), int(c - c0)) for r, c in np.argwhere(grid == color)}


def extract_orbit_templates(examples: list[dict]) -> Counter[tuple[tuple[int, int], ...]]:
    templates: Counter[tuple[tuple[int, int], ...]] = Counter()
    for ex in examples:
        x = arr(ex["input"])
        y = arr(ex["output"])
        b = bbox((x != 0) | (y != 0))
        if b is None:
            continue
        changed = x != y
        added_colors = set(int(v) for v in y[changed].ravel())
        for color in added_colors:
            before = rel_positions(x, color, b)
            after = rel_positions(y, color, b)
            if before < after:
                templates[tuple(sorted(after))] += 1
    return templates


def apply_templates(x: np.ndarray, templates: list[set[tuple[int, int]]]) -> tuple[np.ndarray, int | str, str, str]:
    b = bbox(x != 0)
    if b is None:
        return x.copy(), "", "", "empty input"
    r0, c0, r1, c1 = b
    out = x.copy()
    colors = [int(v) for v in np.unique(x) if int(v) != 0]
    candidates = []
    for color in colors:
        pos = rel_positions(x, color, b)
        for tidx, tmpl in enumerate(templates):
            missing = tmpl - pos
            extra = pos - tmpl
            if not missing or extra:
                continue
            ok = True
            for rr, cc in missing:
                ar, ac = r0 + rr, c0 + cc
                if ar < 0 or ac < 0 or ar >= out.shape[0] or ac >= out.shape[1] or out[ar, ac] != 0:
                    ok = False
                    break
            if ok:
                candidates.append((len(missing), color, tidx, missing))
    if len(candidates) != 1:
        return x.copy(), "", "", f"candidate_count={len(candidates)}"
    _, color, tidx, missing = candidates[0]
    for rr, cc in missing:
        out[r0 + rr, c0 + cc] = color
    return out, color, str(sorted(templates[tidx])), "ok"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TARGET_TASK)
    train = task["train"]
    test = task["test"]
    arc = task["arc-gen"]
    templates_counter = extract_orbit_templates(train)
    templates = [set(tmpl) for tmpl, _ in templates_counter.most_common()]
    examples = train + test + arc
    rows: list[EvalRow] = []
    split_counts = Counter()
    split_pass = Counter()
    for idx, ex in enumerate(examples):
        split = "train" if idx < len(train) else ("test" if idx < len(train) + len(test) else "arc-gen")
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred, color, tmpl, reason = apply_templates(x, templates)
        ok = np.array_equal(pred, y)
        split_counts[split] += 1
        if ok:
            split_pass[split] += 1
        rows.append(
            EvalRow(
                example_idx=idx,
                split=split,
                status="pass" if ok else "fail",
                target_color=color,
                selected_template=tmpl,
                changed_count=int((pred != x).sum()),
                reason=reason if not ok else "ok",
            )
        )
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if all(row.status == "pass" for row in rows) else "partial",
        "target_task": TARGET_TASK,
        "train_template_count": len(templates),
        "templates": [{"positions": sorted(list(tmpl)), "train_count": count} for tmpl, count in templates_counter.most_common()],
        "split_counts": dict(split_counts),
        "split_pass": dict(split_pass),
        "total_pass": sum(1 for row in rows if row.status == "pass"),
        "total_examples": len(rows),
        "decision": "If all pass, implement this orbit completion rule in cost-aware ONNX; otherwise add role inference beyond train templates.",
        "leakage_risk": "low-to-medium: templates learned from train only, evaluated on test/all arc-gen.",
        "overfitting_risk": "medium: task-specific orbit templates; needs holdout if generalized.",
    }
    with (EXP_DIR / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (EXP_DIR / "eval_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(EvalRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([row.__dict__ for row in rows])
    notes = f"""# {EXP_ID}

## 仮説

task020は、trainから得られる色ごとの相対位置orbit templateを補完するruleで説明できる。

## 結果

- status: {result["status"]}
- templates: {len(templates)}
- pass: {result["total_pass"]}/{len(rows)}
- split pass: {dict(split_pass)} / {dict(split_counts)}

## 解釈

train由来templateだけでtest/all arc-genへ汎化するかを検証した。通ればsignature lookupではなく、人間可読なorbit completion ruleとして次にONNX化する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
