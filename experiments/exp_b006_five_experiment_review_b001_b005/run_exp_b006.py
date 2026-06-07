from __future__ import annotations

import json
import pathlib
from datetime import date


ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP_ID = "exp_b006_five_experiment_review_b001_b005"
EXP_DIR = ROOT / "experiments" / EXP_ID

EXPS = [
    ROOT / "experiments" / "exp_b001_rule_replacement_backlog" / "result.json",
    ROOT / "experiments" / "exp_b002_p0_explainable_rule_sweep" / "result.json",
    ROOT / "experiments" / "exp_b003_teacher_gain_p0_rule_sweep" / "result.json",
    ROOT / "experiments" / "exp_b004_teacher_gain_p0_structural_taxonomy" / "result.json",
    ROOT / "experiments" / "exp_b005_sparse_neighborhood_fill_sweep" / "result.json",
]


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def notes_text(result: dict) -> str:
    lines = [
        f"# {EXP_ID}",
        "",
        "## 目的",
        "",
        "exp_b001-b005が、LB最大化、全task cost<=250、7700到達に本当に有意義だったかを問い直す。",
        "",
        "## 評価",
        "",
        "| exp | verdict | reason | next action |",
        "|---|---|---|---|",
    ]
    for row in result["reviews"]:
        lines.append(f"| `{row['exp_id']}` | {row['verdict']} | {row['reason']} | {row['next_action']} |")
    lines.extend(
        [
            "",
            "## 結論",
            "",
            result["conclusion"],
            "",
            "## 次のPDCA",
            "",
        ]
    )
    for action in result["next_pdca"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    payloads = [load(path) for path in EXPS]
    reviews = [
        {
            "exp_id": "exp_b001_rule_replacement_backlog",
            "verdict": "useful",
            "reason": "400-task backlog, cost<=250 direction, and strict seed LB calibration were established. LB 5929.89 gives a safer baseline than exp041.",
            "next_action": "Keep exp005 strict seed as calibration baseline and only add small validated deltas.",
        },
        {
            "exp_id": "exp_b002_p0_explainable_rule_sweep",
            "verdict": "partly useful but target definition was wrong",
            "reason": "It swept queue-top signature tasks, not teacher-gain P0. The no-hit result still rejects overly simple D4/rectangle/line rules for broad signature tasks.",
            "next_action": "Do not use queue-rank P0 alone; prioritize teacher-gain P0 plus strict high-cost tasks separately.",
        },
        {
            "exp_id": "exp_b003_teacher_gain_p0_rule_sweep",
            "verdict": "useful negative result",
            "reason": "Corrected target set to 18 teacher-gain P0 tasks. No full hit, but task020 D4 remained the best partial and showed sparse/orbit direction.",
            "next_action": "Continue with richer object-role grammar rather than simple geometry primitives.",
        },
        {
            "exp_id": "exp_b004_teacher_gain_p0_structural_taxonomy",
            "verdict": "highly useful",
            "reason": "Classified P0 tasks into grammar needs. Most are sparse color-role fill or shape/object crop, which are plausible cost<=250 targets.",
            "next_action": "Use taxonomy to choose grammar modules and avoid random rule additions.",
        },
        {
            "exp_id": "exp_b005_sparse_neighborhood_fill_sweep",
            "verdict": "useful negative result",
            "reason": "Pure local-neighbor counts produced zero train-fit candidates, proving P0 sparse fill needs object-role/bbox-local/symmetry features.",
            "next_action": "Add bbox-local coordinate and object-role predicates; stop using raw neighbor-count-only rules.",
        },
    ]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "review_complete",
        "reviewed_experiments": [p["exp_id"] for p in payloads],
        "reviews": reviews,
        "conclusion": (
            "The five experiments were meaningful as a course-correction block, not as score-producing work. "
            "They established a Kaggle-calibrated strict baseline, corrected the target set, and rejected three weak grammar families. "
            "For the cost<=250 objective, the next PDCA must focus on object-role plus bbox-local coordinate rules, then tiny static-mask/ScatterND lowering."
        ),
        "next_pdca": [
            "exp_b007: build bbox-local/object-role feature miner for sparse background fill tasks, starting with task020/126/251/90/37.",
            "Require candidate rules to state target color role and target coordinates; train-fit is not enough unless all arc-gen passes.",
            "Only after full pass, implement lowering with tiny coordinate/mask plan and submit a single-task delta bundle.",
            "Track every accepted candidate against cost<=250, not merely against current artifact.",
        ],
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: review experiment",
        "leakage_risk": "low: review only.",
        "overfitting_risk": "low: review only.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXP_DIR / "notes.md").write_text(notes_text(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
