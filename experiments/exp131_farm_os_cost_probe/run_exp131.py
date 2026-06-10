from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.neurogolf_farm import CostExtractor
from experiments.phase1_rewrite_utils import load_neurogolf_utils, make_model, score_model


EXP_DIR = ROOT / "experiments" / "exp131_farm_os_cost_probe"


def identity_model() -> bytes:
    return make_model([helper.make_node("Identity", ["input"], ["output"])], [], "exp131_identity")


def channel_gather_model() -> bytes:
    idx = numpy_helper.from_array(np.arange(10, dtype=np.int64), "idx")
    node = helper.make_node("Gather", ["input", "idx"], ["output"], axis=1)
    return make_model([node], [idx], "exp131_channel_gather")


def one_by_one_conv_model() -> bytes:
    weight = np.zeros((10, 10, 1, 1), dtype=np.float32)
    for c in range(10):
        weight[c, c, 0, 0] = 1.0
    node = helper.make_node("Conv", ["input", "weight"], ["output"])
    return make_model([node], [numpy_helper.from_array(weight, "weight")], "exp131_conv1x1")


def full_grid_where_model() -> bytes:
    cond = numpy_helper.from_array(np.ones((1, 10, 30, 30), dtype=np.bool_), "cond")
    zeros = numpy_helper.from_array(np.zeros((1, 10, 30, 30), dtype=np.float32), "zeros")
    node = helper.make_node("Where", ["cond", "input", "zeros"], ["output"])
    return make_model([node], [cond, zeros], "exp131_full_grid_where")


def small_where_expand_model() -> bytes:
    cond = numpy_helper.from_array(np.ones((1, 10, 1, 1), dtype=np.bool_), "cond")
    zeros = numpy_helper.from_array(np.zeros((1, 10, 1, 1), dtype=np.float32), "zeros")
    shape = numpy_helper.from_array(np.asarray([1, 10, 30, 30], dtype=np.int64), "shape")
    where = helper.make_node("Where", ["cond", "input", "zeros"], ["small"])
    expand = helper.make_node("Expand", ["small", "shape"], ["output"])
    return make_model([where, expand], [cond, zeros, shape], "exp131_small_where_expand")


def static_slice_pad_model() -> bytes:
    starts = numpy_helper.from_array(np.asarray([0, 0, 0, 0], dtype=np.int64), "starts")
    ends = numpy_helper.from_array(np.asarray([1, 10, 3, 3], dtype=np.int64), "ends")
    axes = numpy_helper.from_array(np.asarray([0, 1, 2, 3], dtype=np.int64), "axes")
    pads = numpy_helper.from_array(np.asarray([0, 0, 0, 0, 0, 0, 27, 27], dtype=np.int64), "pads")
    zero = numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "zero")
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["crop"]),
        helper.make_node("Pad", ["crop", "pads", "zero"], ["output"], mode="constant"),
    ]
    return make_model([nodes[0], nodes[1]], [starts, ends, axes, pads, zero], "exp131_static_slice_pad", opset_version=11)


def infer_and_save(raw: bytes, path: Path) -> None:
    model = onnx.load_model_from_string(raw)
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        pass
    path.write_bytes(model.SerializeToString())


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    extractor = CostExtractor()
    cases = [
        ("identity", identity_model, "cheap control"),
        ("channel_gather", channel_gather_model, "cheap channel permutation control"),
        ("conv1x1", one_by_one_conv_model, "small param local mask control"),
        ("full_grid_where", full_grid_where_model, "conditional full-grid reject control"),
        ("small_where_expand", small_where_expand_model, "small Where but full-grid Expand risk control"),
        ("static_slice_pad_3x3", static_slice_pad_model, "crop floor control"),
    ]
    rows = []
    for name, factory, purpose in cases:
        raw = factory()
        model_path = EXP_DIR / f"{name}.onnx"
        infer_and_save(raw, model_path)
        signal = extractor.assess_onnx_path(model_path).as_dict()
        try:
            official_cost, _, trace_status = score_model(utils, raw, 1, name, EXP_DIR)
        except Exception as exc:
            official_cost = None
            trace_status = f"score_error:{str(exc)[:160]}"
        rows.append(
            {
                "case": name,
                "purpose": purpose,
                "official_cost": official_cost,
                "trace_status": trace_status,
                **signal,
            }
        )

    csv_path = EXP_DIR / "cost_probe_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "exp": "exp131_farm_os_cost_probe",
        "status": "cost_probe_complete",
        "n_cases": len(rows),
        "rows": rows,
        "outputs": {"cost_probe_rows": str(csv_path)},
        "elapsed_s": round(time.time() - t0, 3),
        "decision": "Numeric cost proxy is now recorded, but official score_network remains the acceptance source.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_notes(result)


def write_notes(result: dict[str, object]) -> None:
    lines = [
        "# exp131_farm_os_cost_probe",
        "",
        "## Hypothesis",
        "",
        "レビュー指摘に従い、cost extractorを現行cost式寄りの数値proxyへ修正すれば、cheap controlとhigh-risk controlを実測前により妥当に分離できる。",
        "",
        "## Result",
        "",
    ]
    for row in result["rows"]:
        lines.append(
            f"- {row['case']}: band `{row['predicted_cost_band']}`, proxy `{row['cost_proxy']}`, official `{row['official_cost']}`, reject `{row['hard_reject']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "これはscore-producing実験ではなくfarm OSの校正実験。accepted/rejectedの判断はcost extractor単体ではなく、official score_network、full-arc validation、bundle ledgerを接続して行う必要がある。",
            "",
            "## Leakage / Overfitting Risk",
            "",
            "task001に対するcost probeであり、出力正解性は評価していない。LBやlocal estimateへ加算しない。",
        ]
    )
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
