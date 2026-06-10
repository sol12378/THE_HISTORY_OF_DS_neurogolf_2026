from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.public_zero_probe_utils import build_probe


def main() -> None:
    result = build_probe(
        exp_dir=ROOT / "experiments" / "exp190_risk_next4g_failure_bisection_probe",
        exp_id="exp190_risk_next4g_failure_bisection_probe",
        purpose="Prepare next high-risk-inventory fail-stub probe for task238/112/377/177 while exp188 scores.",
        base_exp=ROOT / "experiments" / "exp178_task285_b035_repair_probe",
        base_public_lb=6005.93,
        target_points={
            238: 13.927565965061614,
            112: 14.010141445218116,
            377: 13.966336894339644,
            177: 13.978195575521427,
        },
        submission_decision="hold_until_exp188_result",
        overfitting_risk="medium: risk-inventory prioritization may be less calibrated than subset-sum consensus.",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
