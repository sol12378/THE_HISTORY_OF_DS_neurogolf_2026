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
        exp_dir=ROOT / "experiments" / "exp183_risk_next4d_failure_bisection_probe",
        exp_id="exp183_risk_next4d_failure_bisection_probe",
        purpose="Prepare next high-risk-inventory fail-stub probe for task187/204/198/364 while exp182 scores.",
        base_exp=ROOT / "experiments" / "exp178_task285_b035_repair_probe",
        base_public_lb=6005.93,
        target_points={
            187: 13.435307852707972,
            204: 13.442636685599203,
            198: 13.449576458609688,
            364: 13.737345222483407,
        },
        submission_decision="hold_until_exp182_result",
        overfitting_risk="medium: risk-inventory prioritization may be less calibrated than subset-sum consensus.",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
