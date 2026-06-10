from __future__ import annotations

import json
import pathlib
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp233_task300_channel_gather_mask4x3_cost_probe.run_exp233 import build_candidate  # noqa: E402
from experiments.phase1_rewrite_utils import (  # noqa: E402
    Candidate,
    evaluate_candidate,
    load_base_tasks,
    load_neurogolf_utils,
    point,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp234_task300_max_color_submit_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 300
BASE_EXP = ROOT / "experiments" / "exp178_task285_b035_repair_probe"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
BASE_PUBLIC_LB = 6005.93


def load_base_raws() -> dict[int, bytes]:
    with zipfile.ZipFile(BASE_EXP / "submission.zip") as zf:
        return {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_base_tasks()
    base = base_tasks[TASK_ID]
    candidate = Candidate(
        TASK_ID,
        "max_color_channel_gather_mask4x3",
        base.route,
        build_candidate(),
        "generated",
        "max-color channel GatherElements to 1ch mask, crop4x3 then channelize",
    )
    eval_row, raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
    raws = load_base_raws()
    if raw is not None and eval_row.status == "improved":
        raws[TASK_ID] = raw
    write_zip(OUTPUT_ZIP, raws)
    sanity = zip_sanity(OUTPUT_ZIP)
    delta = 0.0
    expected_lb = BASE_PUBLIC_LB
    if isinstance(eval_row.candidate_points, float):
        delta = float(eval_row.candidate_points) - float(eval_row.baseline_points)
        expected_lb += delta
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "zip_ready" if raw is not None and eval_row.status == "improved" and sanity["count"] == 400 else "not_ready",
        "purpose": "Submit-safe probe for task300 max-color mask4x3 cost improvement on top of exp178 current public best.",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_public_lb": BASE_PUBLIC_LB,
        "candidate_eval": asdict(eval_row),
        "expected_local_delta": delta,
        "expected_lb_if_public_pass": expected_lb,
        "zip_sanity": sanity,
        "outputs": {"submission_zip": str(OUTPUT_ZIP.relative_to(ROOT))},
        "elapsed_s": round(time.time() - started, 3),
        "submission_decision": "submit_single_task_delta" if raw is not None and eval_row.status == "improved" else "no_submit",
        "leakage_risk": "low: input-only structural rule validated on all local examples; no public source raw.",
        "overfitting_risk": "medium-low: rule derived from all available examples; hidden shape/color edge should be calibrated by this single-task submission.",
        "private_risk": "medium: current base exp178 contains high-risk source repairs; this delta itself is rule-based.",
        "sha256_task300": sha256(raw) if raw is not None else "",
        "task300_old_points": base.points,
        "task300_new_points": point(eval_row.candidate_cost) if isinstance(eval_row.candidate_cost, int) else "",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp178 current public bestに、exp233のtask300 max-color mask4x3 improved candidateを1 taskだけ差し替え、LB較正提出用zipを作る。

## 結果

- status: `{result['status']}`
- validation: `{eval_row.validation_status}`
- task300 cost: `{eval_row.baseline_cost}` -> `{eval_row.candidate_cost}`
- expected_local_delta: `{delta}`
- expected_lb_if_public_pass: `{expected_lb}`
- zip_sanity: `{sanity}`

## 判断

zip_readyならKaggleへsingle-task deltaとして提出する。採点待ち中は次のPhase C候補探索を進める。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
