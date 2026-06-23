"""Phase-2 step 0 (zero-risk diagnostic): for the top-N highest-cost union artifacts,
split cost into memory(activations) vs params(initializer element-count), and report
graph stats + the largest intermediate tensors. This tells us which lever applies:
  - memory-bound  -> dtype minimization (float32 intermediates -> uint8/bool) helps
  - params-bound  -> sparse / smaller-table representation helps (dtype does NOT,
                     because calculate_params counts ELEMENTS not bytes)
"""
import json
import math
import pathlib
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
EXPROOT = HERE.parent
sys.path.insert(0, str(EXPROOT))

import onnx
import onnxruntime as ort
import phase1_rewrite_utils as P
import neurogolf_calib as C

CLIMB = EXPROOT / "exp337_autonomous_climb"
UNION = EXPROOT / "exp336_public_lb7117_task_best_union" / "submission.zip"
TOP_N = 30

led = json.loads((CLIMB / "baseline_ledger.json").read_text(encoding="utf-8"))
costs = {int(k): int(v) for k, v in led["costs"].items()}
raws = C._read_zip(UNION)
utils = P.load_neurogolf_utils()

top = sorted(costs, key=lambda t: -costs[t])[:TOP_N]
print(f"top-{TOP_N} cost tasks (cost desc). cost = memory(bytes) + params(elements)")
print(f"{'tid':>4} {'cost':>8} {'memory':>8} {'params':>8} {'mem%':>5} {'#nodes':>6}  bound        top_ops")

mem_bound, par_bound = [], []
for tid in top:
    raw = raws[tid]
    m, p, reason = C.score_model_calibrated(utils, raw, tid, "anat", CLIMB)
    if m is None:
        print(f"{tid:>4} {costs[tid]:>8}  score failed: {reason}")
        continue
    model = onnx.load_model_from_string(raw)
    ops = Counter(n.op_type for n in model.graph.node)
    n_nodes = len(model.graph.node)
    mem_pct = 100.0 * m / (m + p) if (m + p) else 0
    bound = "memory" if m >= p else "params"
    (mem_bound if bound == "memory" else par_bound).append(tid)
    top_ops = ",".join(f"{k}:{v}" for k, v in ops.most_common(4))
    print(f"{tid:>4} {m + p:>8} {m:>8} {p:>8} {mem_pct:>4.0f}% {n_nodes:>6}  {bound:<11}  {top_ops}")

print(f"\nmemory-bound (dtype-min candidates): {len(mem_bound)} -> {sorted(mem_bound)}")
print(f"params-bound (sparse/table candidates): {len(par_bound)} -> {sorted(par_bound)}")
print(f"sum cost top{TOP_N} = {sum(costs[t] for t in top)}  "
      f"(headroom if all -> cost<=250: ~{sum(P.point(250) - P.point(costs[t]) for t in top):.1f} pts)")
