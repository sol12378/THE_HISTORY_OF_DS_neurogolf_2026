"""P2-2 value test: GridSample inducer. Detect tasks whose output is a content-
independent pixel permutation/remap of the input (output[r,c] = input[f(r,c)], same
f across examples), fit the sampling grid, emit a single-op GridSample candidate, and
gate it (static + full-arc(-1) + official rescore). Report any task solved cheaper
than the union. Single op => no intermediates => cost ~= grid params (<=1800)."""
import json
import pathlib
import sys
from collections import defaultdict

WS = pathlib.Path(r"D:\Data_Science\autoresearch\competitions\neurogolf-2026\workspace")
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS / "experiments"))

import numpy as np
import phase1_rewrite_utils as P
import neurogolf_calib as C
from neurogolf_farm.ir import IRNode, IRProgram, PrimitiveKind, TensorSpec
from neurogolf_farm.emitter import emit_candidate

CLIMB = WS / "experiments" / "exp337_autonomous_climb"
UNION = WS / "experiments" / "exp336_public_lb7117_task_best_union" / "submission.zip"
led = json.loads((CLIMB / "baseline_ledger.json").read_text(encoding="utf-8"))
costs = {int(k): int(v) for k, v in led["costs"].items()}
raws = C._read_zip(UNION)
utils = P.load_neurogolf_utils()


def color_grid(g):
    arr = -np.ones((30, 30), dtype=int)
    for r, row in enumerate(g[:30]):
        for c, v in enumerate(row[:30]):
            arr[r, c] = int(v)
    return arr


def fit_grid(examples):
    ins = [color_grid(e["input"]) for e in examples]
    outs = [color_grid(e["output"]) for e in examples]
    pos_by_color = []
    for inp in ins:
        d = defaultdict(set)
        for sr in range(30):
            for sc in range(30):
                d[inp[sr, sc]].add((sr, sc))
        pos_by_color.append(d)
    grid = np.full((1, 30, 30, 2), -3.0, dtype=np.float32)  # out-of-bounds -> samples 0
    for r in range(30):
        for c in range(30):
            ocs = [o[r, c] for o in outs]
            if all(x == -1 for x in ocs):
                continue  # padding cell everywhere -> background (0)
            if any(x == -1 for x in ocs):
                return None  # inconsistent output footprint -> not a clean remap
            cands = None
            for i, oc in enumerate(ocs):
                m = pos_by_color[i].get(oc, set())
                cands = m if cands is None else (cands & m)
                if not cands:
                    return None
            sr, sc = (r, c) if (r, c) in cands else sorted(cands)[0]
            grid[0, r, c, 0] = (2 * sc + 1) / 30 - 1.0
            grid[0, r, c, 1] = (2 * sr + 1) / 30 - 1.0
    return grid


order = sorted(costs, key=lambda t: -costs[t])
fits = wins = 0
results = []
for tid in order:
    if tid in C.NEGATIVE_PAD_TASKS:
        continue
    try:
        ex = P.examples_for(P.load_task(tid), 10)  # fit on a diverse subset
    except Exception:
        continue
    grid = fit_grid(ex)
    if grid is None:
        continue
    fits += 1
    prog = IRProgram(str(tid), "geo", "", (IRNode("gs", PrimitiveKind.GRID_SAMPLE, "GridSample",
        output=TensorSpec("output", (1, 10, 30, 30), "float32"),
        attrs={"grid": grid.tolist(), "mode": "nearest"}),))
    cand = emit_candidate(prog)
    if cand.raw is None:
        continue
    base = P.BaseTask(tid, costs[tid], P.point(costs[tid]), "union", "union", "", raws[tid])
    try:
        ev, acc = P.evaluate_candidate(utils, cand, base, -1, CLIMB)
    except Exception:
        continue
    valid = "pass" in ev.validation_status and ev.validation_status.endswith("0_fail")
    if valid:
        cc = ev.candidate_cost if isinstance(ev.candidate_cost, int) else None
        cheaper = cc is not None and cc < costs[tid]
        wins += int(cheaper)
        results.append((tid, costs[tid], cc, cheaper, acc is not None))
        tag = "*** WIN ***" if cheaper else "valid-but-not-cheaper"
        print(f"task{tid:03d}: union={costs[tid]:>7} gridsample={cc} {tag}")

print(f"\nscanned non-frozen tasks | fit(permutation-like)={fits} | full-arc valid GridSample wins(cheaper)={wins}")
print("WINS:", [(t, uc, gc) for t, uc, gc, ch, _ in results if ch])
