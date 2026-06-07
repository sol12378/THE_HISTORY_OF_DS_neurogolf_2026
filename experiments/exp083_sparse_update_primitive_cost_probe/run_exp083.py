from __future__ import annotations

import csv
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import infer_static_ok, load_neurogolf_utils, make_model, score_model, sha256  # noqa: E402


EXP_ID = "exp083_sparse_update_primitive_cost_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 366


@dataclass(frozen=True)
class PrimitiveRow:
    primitive: str
    updates: int
    status: str
    memory: int | str
    params: int | str
    cost: int | str
    file_bytes: int
    sha256: str
    static_reason: str
    score_reason: str
    notes: str


def init_i(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.int64), name)


def init_f(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.float32), name)


def scatter_indices(n: int) -> np.ndarray:
    rows = []
    for i in range(n):
        color = 1 + (i % 9)
        r = (i * 3) % 30
        c = (i * 7) % 30
        rows.append([0, color, r, c])
    return np.asarray(rows, dtype=np.int64)


def model_constantofshape_scatter(n: int) -> bytes:
    nodes = [
        helper.make_node("ConstantOfShape", ["shape"], ["zero_grid"], value=numpy_helper.from_array(np.asarray([0.0], dtype=np.float32))),
        helper.make_node("ScatterND", ["zero_grid", "indices", "updates"], ["output"]),
    ]
    inits = [
        init_i("shape", np.asarray([1, 10, 30, 30])),
        init_i("indices", scatter_indices(n)),
        init_f("updates", np.ones((n,), dtype=np.float32)),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_cos_scatter_{n}", opset_version=11)


def model_input_zero_scatter(n: int) -> bytes:
    nodes = [
        helper.make_node("Mul", ["input", "zero"], ["zero_grid"]),
        helper.make_node("ScatterND", ["zero_grid", "indices", "updates"], ["output"]),
    ]
    inits = [
        init_f("zero", np.asarray([0.0])),
        init_i("indices", scatter_indices(n)),
        init_f("updates", np.ones((n,), dtype=np.float32)),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_inputzero_scatter_{n}", opset_version=11)


def model_sparse_channel_gather(n: int) -> bytes:
    # Not a task solution: measures small coordinate/channel gathering without writing
    # back to 30x30. The final Tile only satisfies the fixed output shape contract.
    nodes = [
        helper.make_node("Gather", ["input", "rows"], ["row_g"], axis=2),
        helper.make_node("Gather", ["row_g", "cols"], ["patch"], axis=3),
        helper.make_node("ReduceSum", ["patch"], ["scalar"], axes=[2, 3], keepdims=1),
        helper.make_node("Tile", ["scalar", "repeats"], ["output"]),
    ]
    inits = [
        init_i("rows", np.arange(min(n, 30), dtype=np.int64)),
        init_i("cols", np.arange(min(n, 30), dtype=np.int64)),
        init_i("repeats", np.asarray([1, 1, 30, 30])),
    ]
    return make_model(nodes, inits, f"{EXP_ID}_gather_{n}", opset_version=11)


def smoke_runtime(raw: bytes) -> None:
    sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    y = sess.run(None, {"input": np.zeros((1, 10, 30, 30), dtype=np.float32)})[0]
    if y.shape != (1, 10, 30, 30):
        raise RuntimeError(f"bad output shape {y.shape}")


def score_raw(utils: Any, name: str, n: int, raw: bytes, notes: str) -> PrimitiveRow:
    try:
        smoke_runtime(raw)
        model = onnx.load_model_from_string(raw)
        ok, static_reason = infer_static_ok(model)
        if not ok:
            return PrimitiveRow(name, n, "rejected_static", "", "", "", len(raw), sha256(raw), static_reason, "", notes)
        memory, params, score_reason = score_model(utils, raw, TASK_ID, f"{name}_{n}", EXP_DIR)
        if memory is None or params is None:
            return PrimitiveRow(name, n, "score_failed", "", "", "", len(raw), sha256(raw), static_reason, score_reason, notes)
        return PrimitiveRow(name, n, "scored", memory, params, memory + params, len(raw), sha256(raw), static_reason, score_reason, notes)
    except Exception as exc:
        return PrimitiveRow(name, n, "runtime_failed", "", "", "", len(raw), sha256(raw), "", str(exc)[:180], notes)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    rows: list[PrimitiveRow] = []
    for n in [7, 16, 49, 100]:
        variants = [
            ("constantofshape_scatternd", model_constantofshape_scatter(n), "zero grid via ConstantOfShape + sparse ScatterND"),
            ("input_zero_scatternd", model_input_zero_scatter(n), "zero grid via input*0 + sparse ScatterND"),
        ]
        if n <= 49:
            variants.append(("small_gather_tile_proxy", model_sparse_channel_gather(n), "small Gather then Tile to fixed output; cost proxy only"))
        for name, raw, notes in variants:
            (EXP_DIR / f"{name}_{n}.onnx").write_bytes(raw)
            rows.append(score_raw(utils, name, n, raw, notes))

    with (EXP_DIR / "primitive_costs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PrimitiveRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])

    scored = [r for r in rows if r.status == "scored"]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "cost_probe_ready",
        "hypothesis": "task366のsmall-patch loweringでは、最大7 marker cells / 49 patch cells程度の sparse update primitive が600級に近い可能性がある。",
        "rows": [asdict(r) for r in rows],
        "min_scored_cost": min((int(r.cost) for r in scored), default=None),
        "decision": "If even tiny ScatterND costs thousands, task366 needs a non-writeback representation or another lane for score production.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: primitive cost probe only",
        "leakage_risk": "low: synthetic primitive probe, no labels.",
        "overfitting_risk": "low: cost-only diagnostic.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`task366` のsmall-patch / sparse-coordinate lowering候補として、少数 `ScatterND` 更新や小さい `Gather` の公式costを測る。

## 結果

- scored variants: `{len(scored)}/{len(rows)}`
- min scored cost: `{result["min_scored_cost"]}`

## 判断

これは提出候補ではない。小さいsparse updateでも数千costなら、task366を600級へ落とすには別表現が必要。

## Risk

- leakage risk: low。合成primitiveのみ。
- overfitting risk: low。cost-only診断。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
