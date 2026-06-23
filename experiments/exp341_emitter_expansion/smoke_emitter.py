"""P2-2/P2-1 smoke: each newly-emitted primitive must lower to a valid ONNX that
passes the static gate (infer_static_ok), runs in ORT, and scores. Also checks the
size-conditional gate relax (P2-1)."""
import pathlib
import sys

WS = pathlib.Path(r"D:\Data_Science\autoresearch\competitions\neurogolf-2026\workspace")
sys.path.insert(0, str(WS))  # so `import experiments.phase1_rewrite_utils` works (farm import style)
sys.path.insert(0, str(WS / "experiments"))

import numpy as np
import onnx
import onnxruntime as ort
import phase1_rewrite_utils as P
import neurogolf_calib as C
from neurogolf_farm.ir import IRNode, IRProgram, PrimitiveKind, TensorSpec
from neurogolf_farm.emitter import emit_candidate
from neurogolf_farm.cost_extractor import CostExtractor

CLIMB = WS / "experiments" / "exp337_autonomous_climb"
utils = P.load_neurogolf_utils()


def identity_grid(h=30, w=30):
    grid = np.zeros((1, h, w, 2), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            grid[0, r, c, 0] = (2 * c + 1) / w - 1.0  # x (col)
            grid[0, r, c, 1] = (2 * r + 1) / h - 1.0  # y (row)
    return grid


full = TensorSpec("output", (1, 10, 30, 30), "float32")
cond = np.zeros((1, 10, 30, 30), dtype=bool); cond[0, :, 0, 0] = True
fill = np.zeros((1, 10, 30, 30), dtype=np.float32); fill[0, 5, 0, 0] = 1.0

PROGRAMS = {
    "RECOLOR_DIRECT": IRProgram("1", "recolor", "", (IRNode("g", PrimitiveKind.RECOLOR_DIRECT, "Gather",
        output=full, attrs={"order": [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]}),)),
    "RECOLOR_CAST": IRProgram("1", "recolor", "", (IRNode("gc", PrimitiveKind.RECOLOR_CAST, "Gather",
        output=full, attrs={"order": list(range(10)), "to": int(onnx.TensorProto.UINT8)}),)),
    "COMPUTED_SLICE_PAD": IRProgram("1", "crop", "", (IRNode("sp", PrimitiveKind.COMPUTED_SLICE_PAD, "SlicePad",
        output=full, attrs={"starts": [0, 0, 2, 2], "ends": [1, 10, 12, 12], "pads": [0, 0, 0, 0, 0, 0, 20, 20]}),)),
    "SMALL_LOCAL_MASK": IRProgram("1", "mask", "", (IRNode("w", PrimitiveKind.SMALL_LOCAL_MASK, "Where",
        output=full, attrs={"cond": cond.tolist(), "fill": fill.tolist()}),)),
    "GRID_SAMPLE": IRProgram("1", "geo", "", (IRNode("gs", PrimitiveKind.GRID_SAMPLE, "GridSample",
        output=full, attrs={"grid": identity_grid().tolist(), "mode": "nearest"}),)),
}

dummy = np.zeros((1, 10, 30, 30), dtype=np.float32)
dummy[0, 3, 5, 7] = 1.0
print(f"{'primitive':18} {'emit':>8} {'static':>7} {'ort':>5} {'cost':>7}")
for name, prog in PROGRAMS.items():
    cand = emit_candidate(prog)
    if cand.raw is None:
        print(f"{name:18} {'SKIP':>8} reason={cand.reason[:50]}")
        continue
    model = onnx.load_model_from_string(cand.raw)
    ok, reason = P.infer_static_ok(model)
    try:
        sess = ort.InferenceSession(cand.raw, providers=["CPUExecutionProvider"])
        sess.run(["output"], {"input": dummy})
        ortok = "ok"
    except Exception as exc:
        ortok = f"FAIL:{str(exc)[:30]}"
    m, p, _ = C.score_model_calibrated(utils, cand.raw, 1, name, CLIMB)
    cost = (m + p) if m is not None else "n/a"
    print(f"{name:18} {'gen':>8} {('ok' if ok else reason[:30]):>7} {ortok:>5} {str(cost):>7}")

# P2-1 gate-relax check: SPARSE_WRITEBACK small -> allowed; full-grid -> rejected
ce = CostExtractor()
small = IRProgram("1", "sw", "", (IRNode("s", PrimitiveKind.SPARSE_WRITEBACK, "ScatterND",
    output=TensorSpec("patch", (1, 10, 3, 3), "float32")),))
big = IRProgram("1", "sw", "", (IRNode("s", PrimitiveKind.SPARSE_WRITEBACK, "ScatterND",
    output=TensorSpec("output", (1, 10, 30, 30), "float32")),))
print(f"\nP2-1 gate relax: SPARSE_WRITEBACK small(3x3) hard_reject={ce.assess_ir(small).hard_reject} "
      f"(expect False) | full-grid hard_reject={ce.assess_ir(big).hard_reject} (expect True)")
