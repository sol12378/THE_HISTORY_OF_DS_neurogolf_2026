"""Test the enumerative solver core: rule-ification + lowering + gate over a sample."""
import json
import pathlib
import sys

WS = pathlib.Path(r"D:\Data_Science\autoresearch\competitions\neurogolf-2026\workspace")
sys.path.insert(0, str(WS / "experiments"))

import phase1_rewrite_utils as P
import neurogolf_calib as C
import neurogolf_solver as S

CLIMB = WS / "experiments" / "exp337_autonomous_climb"
UNION = WS / "experiments" / "exp336_public_lb7117_task_best_union" / "submission.zip"
led = json.loads((CLIMB / "baseline_ledger.json").read_text(encoding="utf-8"))
costs = {int(k): int(v) for k, v in led["costs"].items()}
raws = C._read_zip(UNION)
utils = P.load_neurogolf_utils()

# sample: the 22 permutation-like tasks + a few high-cost
sample = [194, 214, 152, 83, 211, 142, 87, 373, 116, 164, 172, 210, 326, 18, 187, 233, 2, 205]
ruleified = wins = lowered = 0
for tid in sample:
    if tid in C.NEGATIVE_PAD_TASKS:
        continue
    ex = P.examples_for(P.load_task(tid), -1)  # ALL arc-gen for verification
    progs = S.enumerate_programs(ex, max_depth=2)
    if not progs:
        print(f"task{tid:03d}: no program found (union={costs[tid]})")
        continue
    ruleified += 1
    # pick shortest program, lower + gate
    progs.sort(key=len)
    best_cost = None
    for prog in progs[:4]:
        cand = S.program_to_candidate(tid, prog, ex)
        if cand is None or cand.raw is None:
            continue
        base = P.BaseTask(tid, costs[tid], P.point(costs[tid]), "union", "union", "", raws[tid])
        try:
            evl, acc = P.evaluate_candidate(utils, cand, base, -1, CLIMB)
        except Exception:
            continue
        if "pass" in evl.validation_status and evl.validation_status.endswith("0_fail") and isinstance(evl.candidate_cost, int):
            lowered += 1
            if best_cost is None or evl.candidate_cost < best_cost:
                best_cost = evl.candidate_cost
            break
    win = best_cost is not None and best_cost < costs[tid]
    wins += int(win)
    tag = "*** WIN ***" if win else ("lowered" if best_cost else "ruleified-not-lowered")
    print(f"task{tid:03d}: union={costs[tid]:>6} solver_cost={best_cost} progs={len(progs)} ex={S.tuple_str(progs[0]) if hasattr(S,'tuple_str') else progs[0]} {tag}")

print(f"\nsample={len(sample)} | ruleified={ruleified} | lowered+valid={lowered} | WINS={wins}")
