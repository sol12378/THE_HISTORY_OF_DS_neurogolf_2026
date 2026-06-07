from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "experiments" / "exp039_deep_fullarc_safe_bypass"))

import run_exp039 as e39  # noqa: E402


TARGET_TASKS = {145}


e39.EXP_ID = "exp041_task145_deeper_mul_chain"
e39.EXP_DIR = ROOT / "experiments" / e39.EXP_ID
e39.BASE_EXP = ROOT / "experiments" / "exp040_deeper_fullarc_safe_bypass"
e39.MAX_PASSES_PER_TASK = 32


def target_tasks_only(exp_dir: pathlib.Path) -> set[int]:
    return set(TARGET_TASKS)


e39.read_selected_tasks = target_tasks_only
e39.read_fullarc_ok_tasks = target_tasks_only


if __name__ == "__main__":
    e39.main()
