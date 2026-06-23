"""End-to-end test of the Qwen LLM proposer: propose -> deterministic verify."""
import pathlib
import sys
import time

WS = pathlib.Path(r"D:\Data_Science\autoresearch\competitions\neurogolf-2026\workspace")
sys.path.insert(0, str(WS / "experiments"))

import phase1_rewrite_utils as P
import neurogolf_solver as S
import neurogolf_llm_proposer as L

client = L.get_client()
print(f"LLM client: {'LIVE' if client else 'UNAVAILABLE'}")

for tid in [87, 116, 164, 373, 211]:
    ex = P.examples_for(P.load_task(tid), -1)
    fit_ex = ex[:6]  # show the LLM a few; verify on ALL
    t0 = time.time()
    progs = L.propose_programs(fit_ex, client=client, samples=1)
    dt = time.time() - t0
    verified = [p for p in progs if S.verify_program(ex, p)]
    print(f"task{tid:03d}: proposed={len(progs)} verified={len(verified)} ({dt:.1f}s) "
          f"{verified[0] if verified else (progs[0] if progs else 'none')}")
