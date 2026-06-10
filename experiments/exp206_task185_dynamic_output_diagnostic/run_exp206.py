from __future__ import annotations

import json
import pathlib
import sys
from datetime import date
from typing import Any

import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_neurogolf_utils, load_task  # noqa: E402


EXP_ID = "exp206_task185_dynamic_output_diagnostic"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185
SOURCE_EXP = ROOT / "experiments" / "exp204_task185_dynamic_axis_candidate_probe"


def tensor_summary(arr: np.ndarray) -> dict[str, Any]:
    nz = np.argwhere(arr != 0)
    return {
        "shape": list(arr.shape),
        "nonzero_count": int(nz.shape[0]),
        "channel_nonzero": {str(c): int((arr[0, c] != 0).sum()) for c in range(arr.shape[1]) if int((arr[0, c] != 0).sum())},
        "top_left_argmax": np.argmax(arr[0, :, :6, :6], axis=0).astype(int).tolist(),
        "top_left_max": np.max(arr[0, :, :6, :6], axis=0).astype(float).tolist(),
    }


def diff_summary(pred: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    diff = pred != expected
    idx = np.argwhere(diff)
    sample = []
    for loc in idx[:40]:
        n, c, r, col = [int(x) for x in loc]
        sample.append({"n": n, "c": c, "r": r, "col": col, "pred": float(pred[n, c, r, col]), "expected": float(expected[n, c, r, col])})
    return {
        "diff_count": int(idx.shape[0]),
        "pred_extra_count": int(((pred != 0) & (expected == 0)).sum()),
        "missing_count": int(((pred == 0) & (expected != 0)).sum()),
        "sample": sample,
    }


def run_variant(raw_path: pathlib.Path, benchmark: dict[str, Any]) -> dict[str, Any]:
    sess = ort.InferenceSession(raw_path.read_bytes(), providers=["CPUExecutionProvider"])
    pred = sess.run(None, {"input": benchmark["input"]})[0]
    expected = benchmark["output"]
    return {
        "variant": raw_path.stem,
        "equal": bool(np.array_equal(pred, expected)),
        "pred_summary": tensor_summary(pred),
        "expected_summary": tensor_summary(expected),
        "diff": diff_summary(pred, expected),
    }


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    task = load_task(TASK_ID)
    example = examples_for(task, 1)[0]
    benchmark = utils.convert_to_numpy(example)
    assert benchmark is not None
    variants = [
        SOURCE_EXP / "dynamic_axis_basic.onnx",
        SOURCE_EXP / "dynamic_axis_nonzero_only.onnx",
        SOURCE_EXP / "dynamic_axis_score_nonzero_core_basic.onnx",
        SOURCE_EXP / "dynamic_axis_score_nonzero_default_bg.onnx",
    ]
    rows = [run_variant(path, benchmark) for path in variants if path.exists()]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "diagnostic_complete",
        "task_id": TASK_ID,
        "source_exp": str(SOURCE_EXP.relative_to(ROOT)),
        "example_index": 0,
        "rows": rows,
        "decision": "Use diff pattern to decide whether exp204 needs bg-channel suppression, output crop placement, or selector-index correction.",
        "submission_decision": "no_submit: diagnostic only",
        "leakage_risk": "low: compares candidate output to validation example for debugging.",
        "overfitting_risk": "medium: example-0 diagnostic only; fixes must still full-validate.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp204 dynamic candidateのexample 0 mismatchをtensor diffで診断する。

## 結果

- variants: `{[row['variant'] for row in rows]}`
- equal: `{ {row['variant']: row['equal'] for row in rows} }`
- diff_count: `{ {row['variant']: row['diff']['diff_count'] for row in rows} }`

## 判断

diff patternに応じて、bg-channel suppression、出力位置、selector-index correctionのどれを次に直すか決める。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
