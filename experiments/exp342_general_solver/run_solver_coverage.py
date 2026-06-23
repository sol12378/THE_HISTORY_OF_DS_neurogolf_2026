"""P3-1 / Layer-D: measure the solver's rule-ification rate over all 393 non-frozen
tasks and run the completeness critic (classify WHY each task is stuck). Enumerative
core over all tasks (fast: fit on a subset, re-verify survivors on FULL arc-gen); the
LLM proposer optionally augments a high-cost-unsolved subset (time-boxed).

Usage: python run_solver_coverage.py [--llm N]   (N = LLM budget for unsolved tasks)
"""
import json
import pathlib
import sys
import time

WS = pathlib.Path(r"D:\Data_Science\autoresearch\competitions\neurogolf-2026\workspace")
sys.path.insert(0, str(WS / "experiments"))

import phase1_rewrite_utils as P
import neurogolf_calib as C
import neurogolf_solver as S

USE_LLM = 0
if "--llm" in sys.argv:
    USE_LLM = int(sys.argv[sys.argv.index("--llm") + 1])

CLIMB = WS / "experiments" / "exp337_autonomous_climb"
UNION = WS / "experiments" / "exp336_public_lb7117_task_best_union" / "submission.zip"
led = json.loads((CLIMB / "baseline_ledger.json").read_text(encoding="utf-8"))
costs = {int(k): int(v) for k, v in led["costs"].items()}
raws = C._read_zip(UNION)
utils = P.load_neurogolf_utils()

tasks = [t for t in sorted(costs) if t not in C.NEGATIVE_PAD_TASKS]
ruleified, lowered, wins = [], [], []
stuck = {"no_rule": [], "rule_not_lowerable": [], "lowered_not_cheaper": []}

t_start = time.time()
for i, tid in enumerate(tasks, 1):
    full = P.examples_for(P.load_task(tid), -1)
    fit = P.examples_for(P.load_task(tid), 6)
    progs = [p for p in S.enumerate_programs(fit, max_depth=2) if S.verify_program(full, p)]
    if not progs:
        stuck["no_rule"].append(tid)
        continue
    ruleified.append(tid)
    progs.sort(key=len)
    best = None
    any_lowered = False
    for prog in progs[:3]:
        cand = S.program_to_candidate(tid, prog, full)
        if cand is None or cand.raw is None:
            continue
        base = P.BaseTask(tid, costs[tid], P.point(costs[tid]), "union", "union", "", raws[tid])
        try:
            ev, acc = P.evaluate_candidate(utils, cand, base, -1, CLIMB)
        except Exception:
            continue
        if isinstance(ev.candidate_cost, int) and ev.validation_status.endswith("0_fail") and "pass" in ev.validation_status:
            any_lowered = True
            best = ev.candidate_cost if best is None else min(best, ev.candidate_cost)
    if not any_lowered:
        stuck["rule_not_lowerable"].append(tid)
    else:
        lowered.append(tid)
        if best < costs[tid]:
            wins.append((tid, costs[tid], best))
        else:
            stuck["lowered_not_cheaper"].append(tid)
    if i % 80 == 0:
        print(f"  ...{i}/{len(tasks)} ruleified={len(ruleified)} lowered={len(lowered)} wins={len(wins)} ({time.time()-t_start:.0f}s)")

print("\n==== ENUMERATIVE coverage (Layer-A, no LLM) ====")
print(f"tasks={len(tasks)} | ruleified={len(ruleified)} ({100*len(ruleified)/len(tasks):.1f}%) | "
      f"lowered+valid={len(lowered)} | WINS(cheaper than union)={len(wins)}")
print("completeness critic (stuck reasons):")
for k, v in stuck.items():
    print(f"  {k}: {len(v)}")
print(f"WINS: {wins}")

# LLM augmentation on the hardest unsolved (highest union cost, no enumerative rule)
if USE_LLM:
    import neurogolf_llm_proposer as L
    client = L.get_client()
    print(f"\n==== LLM augmentation (budget={USE_LLM}, client={'LIVE' if client else 'DOWN'}) ====")
    targets = sorted(stuck["no_rule"], key=lambda t: -costs[t])[:USE_LLM]
    llm_ruleified = []
    for tid in targets:
        full = P.examples_for(P.load_task(tid), -1)
        progs = L.propose_programs(full[:6], client=client, samples=1)
        good = [p for p in progs if S.verify_program(full, p)]
        if good:
            llm_ruleified.append((tid, good[0]))
            print(f"  task{tid:03d} (union={costs[tid]}): LLM ruleified -> {good[0]}")
    print(f"LLM newly ruleified: {len(llm_ruleified)}/{len(targets)} hardest-unsolved tasks")
