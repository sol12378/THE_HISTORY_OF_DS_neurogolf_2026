"""Phase-0 acceptance checks for the climb engine wiring (no Kaggle, no slow sweep)."""
import pathlib
import sys

COMP = pathlib.Path(r"D:\Data_Science\autoresearch\competitions\neurogolf-2026")
sys.path.insert(0, str(COMP / "scripts"))
sys.path.insert(0, str(COMP / "workspace" / "experiments"))

import neurogolf_climb as climb
import neurogolf_calib as C

logs = []
log = logs.append

utils = climb.P.load_neurogolf_utils()
raws, points, costs = climb.load_bundle(utils, log)
total = sum(points.values())
sentinels = [t for t, c in costs.items() if c >= 10**9]

print(f"[1] load_bundle: tasks={len(points)} total={total:.2f} sentinels={len(sentinels)}")
print(f"[2] all 400 raws present: {len(raws) == 400}")
print(f"[3] negative-pad set frozen in sweep: {sorted(C.NEGATIVE_PAD_TASKS)}")

# safety valve emulation (climb.main lines): refuse auto-submit if start_total < FLOOR-50
floor = climb.SUBMIT_FLOOR
would_refuse = total < floor - 50
print(f"[4] safety valve: SUBMIT_FLOOR={floor} start_total={total:.2f} "
      f"-> would_refuse_autosubmit={would_refuse} (expect False = valve releases)")

# submit gate emulation: only submit when total > last_submitted AND total >= floor
last_submitted = total  # init in main
print(f"[5] submit gate at baseline: total>{last_submitted:.2f}? "
      f"{total > last_submitted + climb.MIN_SUBMIT_DELTA} (expect False = no resubmit of floor)")

# freeze check: emulate sweep order and confirm the 7 are skipped
order = sorted(points, key=lambda t: points[t])
skipped_frozen = [t for t in order if t in C.NEGATIVE_PAD_TASKS]
attempted = [t for t in order if t not in C.NEGATIVE_PAD_TASKS and costs[t] < 10**9]
print(f"[6] frozen tasks that sweep will skip: {sorted(skipped_frozen)} "
      f"(count={len(skipped_frozen)}, expect 7)")
print(f"[7] improvable task count (non-frozen, non-sentinel): {len(attempted)} (expect 393)")

ok = (len(points) == 400 and len(sentinels) == 0 and len(raws) == 400
      and not would_refuse and len(skipped_frozen) == 7 and len(attempted) == 393
      and 7116 <= total <= 7180)
print(f"\nPHASE-0 WIRING ALL-GREEN: {ok}")
