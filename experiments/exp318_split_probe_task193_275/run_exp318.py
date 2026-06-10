from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.public_zero_probe_utils import build_probe


EXP_ID = "exp318_split_probe_task193_275"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp297_exp262_skip_task048_336_fresh_candidates"
BASE_PUBLIC_LB = 6008.96
TARGET_POINTS = {
    193: 14.109558240338645,
    275: 13.662869436848247,
}


def main() -> None:
    result = build_probe(
        exp_dir=EXP_DIR,
        exp_id=EXP_ID,
        purpose=(
            "Split follow-up for exp315: fail-stub task193 and task275 only. "
            "If the LB drop is near 27.7724 these tasks are public-scoring alive; "
            "if the drop is near 0 or partial, this pair contains public-zero task(s)."
        ),
        base_exp=BASE_EXP,
        base_public_lb=BASE_PUBLIC_LB,
        target_points=TARGET_POINTS,
        submission_decision="submit_probe_after_sanity; do not submit another bisection probe until this score completes",
        overfitting_risk=(
            "medium: public diagnostic split of exp315 ambiguity; any repair must use full validation "
            "and correctness-first candidates, not public-score-only adoption."
        ),
    )
    result["date"] = "2026-06-11"
    result["exp315_context"] = {
        "missing_drop_vs_all_alive": 27.760763322976885,
        "tested_pair": [193, 275],
        "pair_points_sum": sum(TARGET_POINTS.values()),
        "alternative_pair": [243, 101],
        "interpretation": (
            "Observed LB near 5981.19 means task193/task275 are alive and the alternative pair becomes stronger. "
            "Observed LB near 6008.96 means both are public-zero. Intermediate drop identifies a single zero."
        ),
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
