"""Rebuild the baseline with the calibrated scorer to a SEPARATE ledger file, then
compare against the current (sentinel-laden) baseline. Non-destructive."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
EXPROOT = HERE.parent  # workspace/experiments (where modules + exp33x dirs live)
sys.path.insert(0, str(EXPROOT))

import phase1_rewrite_utils as P
import neurogolf_calib as C

CLIMB = EXPROOT / "exp337_autonomous_climb"
UNION = EXPROOT / "exp336_public_lb7117_task_best_union" / "submission.zip"
OUT = CLIMB / "baseline_ledger_calibrated.json"

utils = P.load_neurogolf_utils()
res = C.rebuild_baseline_calibrated(utils, UNION, OUT, log=print, resume=True)

old = json.loads((CLIMB / "baseline_ledger.json").read_text(encoding="utf-8"))
old_costs = {int(k): int(v) for k, v in old["costs"].items()}
old_sent = sorted(t for t, c in old_costs.items() if c >= 10**9)

print("=" * 64)
print(f"OLD baseline total = {old['total']:.2f}   sentinels = {len(old_sent)} {old_sent}")
print(f"NEW baseline total = {res['total']:.2f}   sentinels = {len(res['sentinels'])} {res['sentinels']}")
print(f"delta              = {res['total'] - old['total']:+.2f}")
# show what changed (the negative-pad 7)
new_costs = res["costs"]
print("changed tasks (old sentinel -> new cost/points):")
for t in sorted(C.NEGATIVE_PAD_TASKS):
    print(f"  task{t:03d}: cost {old_costs.get(t)} -> {new_costs.get(t)}  points -> {res['points'][t]:.3f}")
# verify the 393 are unchanged vs official
unchanged = sum(1 for t, c in new_costs.items() if c < 10**9 and t not in C.NEGATIVE_PAD_TASKS
                and old_costs.get(t) == c)
print(f"non-negpad tasks with IDENTICAL cost vs old official = {unchanged}/393")
print(f"completion: 0 sentinels? {len(res['sentinels']) == 0}   total in [7116,7118]? {7116 <= res['total'] <= 7118}")
