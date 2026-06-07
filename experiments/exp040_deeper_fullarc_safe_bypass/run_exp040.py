from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "experiments" / "exp039_deep_fullarc_safe_bypass"))

import run_exp039 as e39  # noqa: E402


e39.EXP_ID = "exp040_deeper_fullarc_safe_bypass"
e39.EXP_DIR = ROOT / "experiments" / e39.EXP_ID
e39.BASE_EXP = ROOT / "experiments" / "exp039_deep_fullarc_safe_bypass"
e39.MAX_PASSES_PER_TASK = 16


if __name__ == "__main__":
    e39.main()
