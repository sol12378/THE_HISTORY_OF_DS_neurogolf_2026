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
        exp_dir=ROOT / "experiments" / "exp184_risk_next4e_failure_bisection_probe",
        exp_id="exp184_risk_next4e_failure_bisection_probe",
        purpose="Prepare next high-risk-inventory fail-stub probe for task284/300/379/340 while exp183 scores.",
        base_exp=ROOT / "experiments" / "exp178_task285_b035_repair_probe",
        base_public_lb=6005.93,
        target_points={
            284: 13.475808050254287,
            300: 13.74137338597163,
            379: 13.8086718845216,
            340: 13.838875786203188,
        },
        submission_decision="hold_until_exp183_result",
        overfitting_risk="medium: risk-inventory prioritization may be less calibrated than subset-sum consensus.",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
