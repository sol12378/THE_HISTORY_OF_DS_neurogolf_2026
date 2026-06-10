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
        exp_dir=ROOT / "experiments" / "exp181_risk_next4b_failure_bisection_probe",
        exp_id="exp181_risk_next4b_failure_bisection_probe",
        purpose="Prepare next high-risk-inventory fail-stub probe for task281/203/126/159 while exp179 scores.",
        base_exp=ROOT / "experiments" / "exp178_task285_b035_repair_probe",
        base_public_lb=6005.93,
        target_points={
            281: 13.60794357751076,
            203: 13.610350127195805,
            126: 13.636839844664504,
            159: 13.597994922615115,
        },
        submission_decision="hold_until_exp179_result",
        overfitting_risk="medium: risk-inventory prioritization may be less calibrated than subset-sum consensus.",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
