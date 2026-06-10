from __future__ import annotations

import json
import pathlib
import sys
from datetime import date

import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp226_task300_max_color_mask4x3_cost_probe.run_exp226 import build_candidate  # noqa: E402
from experiments.phase1_rewrite_utils import grid_to_array, load_neurogolf_utils, load_task, one_hot_padded  # noqa: E402


EXP_ID = "exp227_task300_exp226_output_diagnostic"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 300


def onehot_to_grid(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y)
    return np.argmax(arr[0], axis=0).astype(np.int64)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    task = load_task(TASK_ID)
    ex = (task["train"] + task["test"] + task["arc-gen"])[0]
    x = one_hot_padded(ex["input"])
    expected = grid_to_array(ex["output"])
    raw = build_candidate()
    session = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    out = session.run(None, {"input": x})[0]
    pred = onehot_to_grid(out)
    expected_padded = np.zeros((30, 30), dtype=np.int64)
    expected_padded[: expected.shape[0], : expected.shape[1]] = expected

    # Re-run through official converter for a direct sanity check on output decoding.
    converted = utils.convert_to_numpy(ex)
    official_out = utils.run_network(session, converted["input"])

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "expected_shape": list(expected.shape),
        "expected_top_left": expected_padded[:6, :6].tolist(),
        "pred_top_left_argmax": pred[:6, :6].tolist(),
        "pred_nonzero": np.argwhere(pred != 0).tolist()[:50],
        "pred_nonzero_values": [int(pred[tuple(rc)]) for rc in np.argwhere(pred != 0).tolist()[:50]],
        "official_out_type": str(type(official_out)),
        "official_out_repr": str(official_out)[:1000],
        "argmax_equal_padded": bool(np.array_equal(pred, expected_padded)),
        "decision": "Use this to identify whether exp226 mismatch is color selection, bbox index, or output decoding.",
        "submission_decision": "no_submit: diagnostic only",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp226のtask300 ONNX候補がexample 0でmismatchする原因を、出力gridの位置/色から診断する。

## 結果

- argmax_equal_padded: `{result['argmax_equal_padded']}`
- expected_top_left: `{result['expected_top_left']}`
- pred_top_left_argmax: `{result['pred_top_left_argmax']}`

## 判断

result.jsonのtop-left比較を見て、色選択・bbox index・出力decodeのどこがずれているかを決める。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
