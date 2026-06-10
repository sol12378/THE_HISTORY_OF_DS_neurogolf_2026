from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter
from datetime import date
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, grid_to_array, load_neurogolf_utils, load_task  # noqa: E402


EXP_ID = "exp207_task185_dynamic_intermediate_diagnostic"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 185
SOURCE_EXP = ROOT / "experiments" / "exp204_task185_dynamic_axis_candidate_probe"


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


def special_score(arr: np.ndarray, bg: int, axis: int, win: tuple[int, ...]) -> int:
    vals = arr[np.asarray(win), :] if axis == 0 else arr[:, np.asarray(win)]
    return int(((vals != 0) & (vals != bg)).sum())


def choose_axis(arr: np.ndarray, bg: int, axis: int, wins: list[tuple[int, ...]]) -> tuple[int, ...]:
    return max(wins, key=lambda win: (special_score(arr, bg, axis, win), -win[0]))


def render_lattice(arr: np.ndarray, bg: int, rr: tuple[int, ...], cc: tuple[int, ...]) -> np.ndarray:
    mat = np.zeros((4, 4), dtype=np.int64)
    for i, r in enumerate(rr):
        for j, c in enumerate(cc):
            v = int(arr[r, c])
            mat[i, j] = 0 if v in (0, bg) else v
    return mat


def add_outputs(raw: bytes, names: list[str]) -> bytes:
    model = onnx.load_model_from_string(raw)
    existing = {out.name for out in model.graph.output}
    produced = {node_out for node in model.graph.node for node_out in node.output}
    for name in names:
        if name not in existing and name in produced:
            elem_type = TensorProto.INT64 if name in {"row_idx", "col_idx"} else TensorProto.FLOAT
            model.graph.output.append(helper.make_tensor_value_info(name, elem_type, None))
    return model.SerializeToString()


def summarize_array(x: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(x)
    flat = arr.ravel()
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(flat.min()) if flat.size else None,
        "max": float(flat.max()) if flat.size else None,
        "nonzero": int((arr != 0).sum()),
        "sample": flat[:40].astype(float).tolist(),
    }


def run_variant(path: pathlib.Path, benchmark: dict[str, Any], py_rr: tuple[int, ...], py_cc: tuple[int, ...]) -> dict[str, Any]:
    output_names = ["output", "row_idx", "col_idx", "lattice_4x4", "sum2", "small", "small_nonzero", "small_out"]
    raw = add_outputs(path.read_bytes(), output_names[1:])
    sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    available = [out.name for out in sess.get_outputs()]
    values = sess.run(available, {"input": benchmark["input"]})
    by_name = dict(zip(available, values))
    row_idx = by_name.get("row_idx")
    col_idx = by_name.get("col_idx")
    row_positions = [] if row_idx is None else np.unique(row_idx[0, 0, :, 0]).astype(int).tolist()
    col_positions = [] if col_idx is None else np.unique(col_idx[0, 0, 0, :]).astype(int).tolist()
    return {
        "variant": path.stem,
        "available_outputs": available,
        "row_positions": row_positions,
        "col_positions": col_positions,
        "py_rr": list(py_rr),
        "py_cc": list(py_cc),
        "row_match": row_positions == list(py_rr),
        "col_match": col_positions == list(py_cc),
        "summaries": {name: summarize_array(value) for name, value in by_name.items() if name != "output"},
        "output_equal": bool(np.array_equal(by_name["output"], benchmark["output"])),
    }


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    task = load_task(TASK_ID)
    example = examples_for(task, 1)[0]
    benchmark = utils.convert_to_numpy(example)
    assert benchmark is not None
    arr = grid_to_array(example["input"])
    bg = grid_color(arr)
    line_rows, line_cols = grid_lines(arr, bg)
    py_rr = choose_axis(arr, bg, 0, windows(line_rows))
    py_cc = choose_axis(arr, bg, 1, windows(line_cols))
    py_lattice = render_lattice(arr, bg, py_rr, py_cc)
    variants = [
        SOURCE_EXP / "dynamic_axis_basic.onnx",
        SOURCE_EXP / "dynamic_axis_score_nonzero_core_basic.onnx",
        SOURCE_EXP / "dynamic_axis_score_nonzero_default_bg.onnx",
    ]
    rows = [run_variant(path, benchmark, py_rr, py_cc) for path in variants if path.exists()]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "diagnostic_complete",
        "task_id": TASK_ID,
        "source_exp": str(SOURCE_EXP.relative_to(ROOT)),
        "example_index": 0,
        "python_bg": bg,
        "python_rr": list(py_rr),
        "python_cc": list(py_cc),
        "python_lattice": py_lattice.tolist(),
        "rows": rows,
        "decision": "If row/col index matches Python, fix core/output mapping. If not, fix ArgMax-to-template mapping or window ordering.",
        "submission_decision": "no_submit: intermediate diagnostic only",
        "leakage_risk": "low: diagnostic on validation example.",
        "overfitting_risk": "medium: example-0 diagnostic only.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp204 dynamic candidateの中間テンソルを出力化し、example 0でONNXの`row_idx`/`col_idx`/`lattice_4x4`がPython selectorと一致するか確認する。

## 結果

- python_rr: `{list(py_rr)}`
- python_cc: `{list(py_cc)}`
- row_match: `{ {row['variant']: row['row_match'] for row in rows} }`
- col_match: `{ {row['variant']: row['col_match'] for row in rows} }`
- output_equal: `{ {row['variant']: row['output_equal'] for row in rows} }`

## 判断

indexが一致していればcore/output mappingを修正する。不一致ならArgMax-to-template mappingかwindow orderingを修正する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
