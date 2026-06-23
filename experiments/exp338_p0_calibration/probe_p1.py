"""Characterize P1-1: over a task sample, how many of the 16 builders GENERATE a
candidate, how many PASS full-arc, and the cheapest candidate cost vs the union
baseline. Distinguishes 'union already cheaper' (expected) from a generation bug."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
EXPROOT = HERE.parent
sys.path.insert(0, str(EXPROOT))
sys.path.insert(0, str(EXPROOT.parent.parent / "scripts"))

import phase1_rewrite_utils as P
import neurogolf_climb as climb

CLIMB = EXPROOT / "exp337_autonomous_climb"
led = json.loads((CLIMB / "baseline_ledger.json").read_text(encoding="utf-8"))
costs = {int(k): int(v) for k, v in led["costs"].items()}
points = {int(k): float(v) for k, v in led["points"].items()}

utils = P.load_neurogolf_utils()
raws = climb._read_zip(EXPROOT / "exp336_public_lb7117_task_best_union" / "submission.zip")

# sample: 40 tasks spread across the headroom range (ascending points)
order = sorted(points, key=lambda t: points[t])
sample = order[::10][:40]

gen_total = pass_total = cheaper_total = 0
examples_cache = {}
rows = []
for tid in sample:
    if tid in climb.C.NEGATIVE_PAD_TASKS:
        continue
    try:
        examples = P.examples_for(P.load_task(tid), 12)
    except Exception:
        continue
    cands = climb.dsl_candidates(tid, examples, "")
    base = climb.make_base(tid, raws[tid], costs[tid], points[tid])
    n_gen = len(cands)
    gen_total += n_gen
    best_pass_cost = None
    n_pass = 0
    for c in cands:
        try:
            ev, acc = P.evaluate_candidate(utils, c, base, -1, CLIMB)
        except Exception:
            continue
        if isinstance(ev.candidate_cost, int) and "pass" in ev.validation_status and ev.validation_status.endswith("0_fail"):
            n_pass += 1
            if best_pass_cost is None or ev.candidate_cost < best_pass_cost:
                best_pass_cost = ev.candidate_cost
    pass_total += n_pass
    cheaper = best_pass_cost is not None and best_pass_cost < costs[tid]
    cheaper_total += int(cheaper)
    if n_gen or n_pass:
        rows.append((tid, costs[tid], n_gen, n_pass, best_pass_cost, cheaper))

print(f"sample={len(sample)} tasks | total generated={gen_total} | full-arc pass={pass_total} | "
      f"tasks with a cheaper valid candidate={cheaper_total}")
print("tid   base_cost  #gen #pass  best_pass_cost  cheaper?")
for tid, bc, ng, npass, bpc, ch in rows:
    print(f"{tid:>4} {bc:>10} {ng:>5} {npass:>5}  {str(bpc):>14}  {ch}")
