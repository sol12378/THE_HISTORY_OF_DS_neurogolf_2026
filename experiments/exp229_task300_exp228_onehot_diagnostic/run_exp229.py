from __future__ import annotations

import json
import pathlib
import sys
from datetime import date

import numpy as np
import onnx
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp228_task300_max_color_mask4x3_bgfix_cost_probe.run_exp228 import build_candidate  # noqa: E402
from experiments.phase1_rewrite_utils import load_neurogolf_utils, load_task, one_hot_padded  # noqa: E402


EXP_ID = "exp229_task300_exp228_onehot_diagnostic"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 300


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    task = load_task(TASK_ID)
    ex = (task["train"] + task["test"] + task["arc-gen"])[0]
    raw = build_candidate()
    model = utils.sanitize_model(onnx.load_model_from_string(raw))
    sanitized_raw = model.SerializeToString()
    session = ort.InferenceSession(sanitized_raw, providers=["CPUExecutionProvider"])
    x = one_hot_padded(ex["input"])
    expected = one_hot_padded(ex["output"])
    out = session.run(None, {"input": x})[0]
    diff = out - expected
    nonzero_diff = np.argwhere(np.abs(diff) > 1e-5)
    coords = []
    for item in nonzero_diff[:80]:
        b, ch, r, c = [int(v) for v in item]
        coords.append({"ch": ch, "r": r, "c": c, "out": float(out[b, ch, r, c]), "expected": float(expected[b, ch, r, c])})
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "diff_count": int(nonzero_diff.shape[0]),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "out_channel_sums_top4x3": [float(out[0, ch, :4, :3].sum()) for ch in range(10)],
        "expected_channel_sums_top4x3": [float(expected[0, ch, :4, :3].sum()) for ch in range(10)],
        "out_channel_sums_full": [float(out[0, ch].sum()) for ch in range(10)],
        "expected_channel_sums_full": [float(expected[0, ch].sum()) for ch in range(10)],
        "first_diffs": coords,
        "decision": "Inspect channel sums/diffs to fix exp228 one-hot mismatch.",
        "submission_decision": "no_submit: diagnostic only",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp228のmismatchをone-hot tensor差分で診断する。

## 結果

- diff_count: `{result['diff_count']}`
- max_abs_diff: `{result['max_abs_diff']}`
- out_channel_sums_top4x3: `{result['out_channel_sums_top4x3']}`
- expected_channel_sums_top4x3: `{result['expected_channel_sums_top4x3']}`

## 判断

result.jsonのfirst_diffsを見て、背景channelまたはpadding/channel順の問題を修正する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
