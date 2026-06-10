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
        exp_dir=ROOT / "experiments" / "exp188_risk_next4f_failure_bisection_probe",
        exp_id="exp188_risk_next4f_failure_bisection_probe",
        purpose="Prepare next high-risk-inventory fail-stub probe for task328/301/306/387 while exp184 scores.",
        base_exp=ROOT / "experiments" / "exp178_task285_b035_repair_probe",
        base_public_lb=6005.93,
        target_points={
            328: 13.871957907660665,
            301: 13.91804208355036,
            306: 13.926448091389123,
            387: 13.940291846175448,
        },
        submission_decision="hold_until_exp184_result",
        overfitting_risk="medium: risk-inventory prioritization may be less calibrated than subset-sum consensus.",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
