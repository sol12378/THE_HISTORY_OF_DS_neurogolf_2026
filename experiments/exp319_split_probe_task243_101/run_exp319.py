from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.public_zero_probe_utils import build_probe


EXP_ID = "exp319_split_probe_task243_101"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp297_exp262_skip_task048_336_fresh_candidates"
BASE_PUBLIC_LB = 6008.96
TARGET_POINTS = {
    243: 13.874532744845478,
    101: 13.904485714156337,
}


def main() -> None:
    result = build_probe(
        exp_dir=EXP_DIR,
        exp_id=EXP_ID,
        purpose=(
            "Split follow-up for exp315 after exp318 proved task193/task275 alive: "
            "fail-stub task243 and task101 only. If the LB drop is near 27.7790 they are alive; "
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
        "tested_pair": [243, 101],
        "pair_points_sum": sum(TARGET_POINTS.values()),
        "prior_split_exp318": "task193/task275 matched all-alive and are excluded",
        "interpretation": (
            "Observed LB near 5981.18 means task243/task101 are alive and another collision explains exp315. "
            "Observed LB near 6008.96 means both are public-zero. Intermediate drop identifies a single zero."
        ),
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
