from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp124_focused_surgery_fourth_pass_limited import run_exp124 as runner  # noqa: E402


runner.EXP_ID = "exp125_focused_surgery_fifth_pass_limited"
runner.EXP_DIR = ROOT / "experiments" / runner.EXP_ID
runner.PREV_EXPS = [
    ROOT / "experiments" / "exp109_task048_artifact_surgery_sweep",
    ROOT / "experiments" / "exp110_focused_surgery_rulehit_cropish_sweep",
    ROOT / "experiments" / "exp111_focused_surgery_second_pass",
    ROOT / "experiments" / "exp122_focused_surgery_third_pass_limited",
    ROOT / "experiments" / "exp124_focused_surgery_fourth_pass_limited",
]
runner.BASE_ZIP_EXP = runner.PREV_EXPS[-1]
runner.OUTPUT_ZIP = runner.EXP_DIR / "submission.zip"
runner.CURRENT_LOCAL_ESTIMATE = 6282.954985728787


if __name__ == "__main__":
    runner.main()
