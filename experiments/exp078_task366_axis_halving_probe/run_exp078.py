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


EXP_ID = "exp078_task366_axis_halving_probe"
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


def reduce_pair(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
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
    if mode == "diff_else_zero_first":
        return np.where(a != b, a, 0)
    if mode == "diff_else_zero_second":
        return np.where(a != b, b, 0)
    if mode == "max":
        return np.maximum(a, b)
    if mode == "min":
        return np.minimum(a, b)
    raise ValueError(mode)


def axis_halve(x: np.ndarray, out_shape: tuple[int, int], axis: str, mode: str, offset: int = 0) -> np.ndarray | None:
    oh, ow = out_shape
    if axis == "rows":
        if ow != x.shape[1] or oh * 2 + offset > x.shape[0]:
            return None
        rows = []
        for r in range(oh):
            rr = offset + r * 2
            rows.append(reduce_pair(x[rr], x[rr + 1], mode))
        return np.stack(rows, axis=0)
    if axis == "cols":
        if oh != x.shape[0] or ow * 2 + offset > x.shape[1]:
            return None
        cols = []
        for c in range(ow):
            cc = offset + c * 2
            cols.append(reduce_pair(x[:, cc], x[:, cc + 1], mode))
        return np.stack(cols, axis=1)
    return None


def border_crop_then_halve(x: np.ndarray, out_shape: tuple[int, int], axis: str, mode: str) -> np.ndarray | None:
    # Some generated shapes have an odd extra border row/col. Try dropping one side before pair reduction.
    candidates: list[np.ndarray] = [x]
    if axis == "rows" and x.shape[0] == out_shape[0] * 2 + 1:
        candidates.extend([x[:-1, :], x[1:, :]])
    if axis == "cols" and x.shape[1] == out_shape[1] * 2 + 1:
        candidates.extend([x[:, :-1], x[:, 1:]])
    for cand in candidates:
        pred = axis_halve(cand, out_shape, axis, mode)
        if pred is not None:
            return pred
    return None


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
        "diff_else_zero_first",
        "diff_else_zero_second",
        "max",
        "min",
    ]
    rows: list[CandidateEval] = []
    for axis in ["rows", "cols"]:
        for mode in modes:
            rows.append(
                evaluate(
                    task,
                    f"{axis}_{mode}",
                    lambda x, shape, axis=axis, mode=mode: axis_halve(x, shape, axis, mode),
                )
            )
            rows.append(
                evaluate(
                    task,
                    f"{axis}_{mode}_drop_border_if_odd",
                    lambda x, shape, axis=axis, mode=mode: border_crop_then_halve(x, shape, axis, mode),
                )
            )
    rows.sort(key=lambda r: (-r.total_pass, -r.mean_cell_accuracy, r.candidate))
    best = rows[0]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "rule_found" if best.status == "full_pass" else "no_simple_axis_halving_rule",
        "hypothesis": "task366は片軸が半分になるため、2-row/2-col pair reductionで説明できる可能性がある。",
        "best_candidate": asdict(best),
        "candidate_count": len(rows),
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: no ONNX emitted",
        "decision": "full-pass候補がなければ、axis halving単体では不足。次はpair reductionにobject mask/source copyを加える。",
        "leakage_risk": "low: hand-written pair reductions only.",
        "overfitting_risk": "low for rejected candidates; medium if partial candidates are extended.",
    }
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CandidateEval.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`task366` の片軸半分出力を、2-row/2-col pair reductionで説明できるか調べる。

## 結果

- candidates: `{len(rows)}`
- best: `{best.candidate}` = `{best.total_pass}/{best.total}`
- mean cell accuracy: `{best.mean_cell_accuracy:.4f}`

## 判断

full-passがなければ、単純axis halvingでは不足。次はobject mask/source copyを加えたpair compilerへ進む。

## Risk

- leakage risk: low。手書きcandidateのみ。
- overfitting risk: low。採用候補なしなら提出しない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
