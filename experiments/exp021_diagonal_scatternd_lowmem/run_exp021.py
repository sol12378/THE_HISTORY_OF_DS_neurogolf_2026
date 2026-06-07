from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.exp019_boundary_flood_fill_campaign import run_exp019 as campaign
from experiments.phase1_rewrite_utils import ROOT


campaign.EXP_ID = "exp021_diagonal_scatternd_lowmem"
campaign.EXP_DIR = ROOT / "experiments" / campaign.EXP_ID
campaign.OUTPUT_ZIP = campaign.EXP_DIR / "submission.zip"


if __name__ == "__main__":
    campaign.main()
