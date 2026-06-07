from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp079_task366_panel_overlay_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 366


@dataclass(frozen=True)
class CandidateEval:
    candidate: str
    status: str
    train_pass: int
    train_total: int
    test_pass: int
    test_total: int
    arc_pass: int
    arc_total: int
    total_pass: int
    total: int
    mean_cell_accuracy: float
    fail_examples: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def combine(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    if mode == "first":
        return a
    if mode == "second":
        return b
    if mode == "nonzero_first":
        return np.where(a != 0, a, b)
    if mode == "nonzero_second":
        return np.where(b != 0, b, a)
    if mode == "same_else_zero":
        return np.where(a == b, a, 0)
    if mode == "diff_first_else_zero":
        return np.where(a != b, a, 0)
    if mode == "diff_second_else_zero":
        return np.where(a != b, b, 0)
    if mode == "first_if_second_zero_else_zero":
        return np.where(b == 0, a, 0)
    if mode == "second_if_first_zero_else_zero":
        return np.where(a == 0, b, 0)
    if mode == "max":
        return np.maximum(a, b)
    if mode == "min":
        return np.minimum(a, b)
    raise ValueError(mode)


def split_panels(x: np.ndarray, out_shape: tuple[int, int], axis: str) -> tuple[np.ndarray, np.ndarray] | None:
    oh, ow = out_shape
    if axis == "vertical":
        if x.shape[1] != ow or x.shape[0] != oh * 2:
            return None
        return x[:oh, :], x[oh:, :]
    if axis == "horizontal":
        if x.shape[0] != oh or x.shape[1] != ow * 2:
            return None
        return x[:, :ow], x[:, ow:]
    return None


def panel_overlay(x: np.ndarray, out_shape: tuple[int, int], axis: str, mode: str) -> np.ndarray | None:
    panels = split_panels(x, out_shape, axis)
    if panels is None:
        return None
    return combine(panels[0], panels[1], mode)


def evaluate(task: dict[str, Any], name: str, predictor: Callable[[np.ndarray, tuple[int, int]], np.ndarray | None]) -> CandidateEval:
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    pass_counts = Counter()
    total_counts = Counter()
    cell_accs: list[float] = []
    fails: list[dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        total_counts[split] += 1
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred = predictor(x, y.shape)
        ok = pred is not None and pred.shape == y.shape and np.array_equal(pred, y)
        if ok:
            pass_counts[split] += 1
            cell_accs.append(1.0)
        else:
            if pred is not None and pred.shape == y.shape:
                cell_accs.append(float(np.mean(pred == y)))
            else:
                cell_accs.append(0.0)
            if len(fails) < 20:
                fails.append({"idx": idx, "split": split, "input": tuple(x.shape), "output": tuple(y.shape)})
    total_pass = sum(pass_counts.values())
    total = sum(total_counts.values())
    return CandidateEval(
        candidate=name,
        status="full_pass" if total_pass == total else "partial",
        train_pass=pass_counts["train"],
        train_total=total_counts["train"],
        test_pass=pass_counts["test"],
        test_total=total_counts["test"],
        arc_pass=pass_counts["arc-gen"],
        arc_total=total_counts["arc-gen"],
        total_pass=total_pass,
        total=total,
        mean_cell_accuracy=float(np.mean(cell_accs)),
        fail_examples=json.dumps(fails, ensure_ascii=False),
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    modes = [
        "first",
        "second",
        "nonzero_first",
        "nonzero_second",
        "same_else_zero",
        "diff_first_else_zero",
        "diff_second_else_zero",
        "first_if_second_zero_else_zero",
        "second_if_first_zero_else_zero",
        "max",
        "min",
    ]
    rows: list[CandidateEval] = []
    for axis in ["vertical", "horizontal"]:
        for mode in modes:
            rows.append(
                evaluate(
                    task,
                    f"{axis}_{mode}",
                    lambda x, shape, axis=axis, mode=mode: panel_overlay(x, shape, axis, mode),
                )
            )
    rows.sort(key=lambda r: (-r.total_pass, -r.mean_cell_accuracy, r.candidate))
    best = rows[0]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "rule_found" if best.status == "full_pass" else "no_simple_panel_overlay_rule",
        "hypothesis": "task366は上下/左右2パネルを重ねるoverlay ruleとして説明できる可能性がある。",
        "best_candidate": asdict(best),
        "candidate_count": len(rows),
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: no ONNX emitted",
        "decision": "full-passならSlice+Where lowering候補。partialならpanel関係はあるがmask/color-roleを追加する。",
        "leakage_risk": "low: hand-written panel overlay candidates only.",
        "overfitting_risk": "low for full axis/mode rule; medium if extended with branch conditions.",
    }
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CandidateEval.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`task366` を上下/左右2パネルoverlay ruleとして説明できるか調べる。

## 結果

- candidates: `{len(rows)}`
- best: `{best.candidate}` = `{best.total_pass}/{best.total}`
- mean cell accuracy: `{best.mean_cell_accuracy:.4f}`

## 判断

full-pass候補があれば `Slice + Where` loweringへ進む。なければpanel overlayにmask/color-role branchを追加する。

## Risk

- leakage risk: low。手書きcandidateのみ。
- overfitting risk: medium。branch追加時はall-arc full pass必須。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
