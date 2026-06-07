from __future__ import annotations

import json
import pathlib
import sys
import zipfile
from dataclasses import asdict

import onnx

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "experiments"))

from phase1_rewrite_utils import (  # noqa: E402
    Candidate,
    evaluate_candidate,
    load_base_tasks,
    load_neurogolf_utils,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp032_task187_component_lowering"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
TASK_ID = 187
ARC_GEN_SAMPLE = 20


def remove_redundant_and(raw: bytes) -> bytes:
    model = onnx.load_model_from_string(raw)

    # In the task187 public artifact, safe_name_75 is already masked by
    # safe_name_16, so And(safe_name_16, safe_name_75) is equivalent to
    # safe_name_75. Removing this node saves one 1x1x30x30 bool tensor.
    for node in model.graph.node:
        for idx, input_name in enumerate(node.input):
            if input_name == "safe_name_76":
                node.input[idx] = "safe_name_75"

    kept = [node for node in model.graph.node if list(node.output) != ["safe_name_76"]]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return model.SerializeToString()


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_base_tasks(BASE_EXP)
    base_task = base_tasks[TASK_ID]
    base_raws: dict[int, bytes] = {}
    with zipfile.ZipFile(BASE_EXP / "submission.zip") as zf:
        for name in zf.namelist():
            task_id = int(pathlib.Path(name).stem.replace("task", ""))
            base_raws[task_id] = zf.read(name)

    candidate_raw = remove_redundant_and(base_raws[TASK_ID])
    candidate = Candidate(
        TASK_ID,
        "task187_surgery_remove_redundant_and",
        base_task.route,
        candidate_raw,
        "generated",
        "replace safe_name_76 with safe_name_75 and remove redundant And node",
    )
    evaluation, accepted_raw = evaluate_candidate(utils, candidate, base_task, ARC_GEN_SAMPLE, EXP_DIR)

    selected_rows = []
    bundle_sanity = {}
    if accepted_raw is not None:
        base_raws[TASK_ID] = accepted_raw
        write_zip(EXP_DIR / "submission.zip", base_raws)
        bundle_sanity = zip_sanity(EXP_DIR / "submission.zip")
        selected_rows.append(asdict(evaluation))
        (EXP_DIR / "task187.onnx").write_bytes(accepted_raw)

    base_local = sum(task.points for task in base_tasks.values())
    delta = 0.0
    if selected_rows:
        delta = float(evaluation.candidate_points) - float(evaluation.baseline_points)
    local_estimate = base_local + delta
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-06",
        "status": evaluation.status,
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_local_estimate": base_local,
        "task_id": TASK_ID,
        "arc_gen_sample": ARC_GEN_SAMPLE,
        "evaluation": asdict(evaluation),
        "improved_task_count": len(selected_rows),
        "local_estimate_delta": delta,
        "local_estimate": local_estimate,
        "gap_to_6500": 6500.0 - local_estimate,
        "submission_decision": "submit_threshold_reached" if local_estimate >= 6500.0 else "no_submit: below 6500 threshold",
        "zip_sanity": bundle_sanity,
        "leakage_risk": "medium: graph surgery preserves an existing public artifact's semantics; task187 rule was validated separately, but base still includes high-risk lookup artifacts.",
        "overfitting_risk": "medium: candidate is validated on train/test/arc-gen sample20, not full private-like holdout.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## Hypothesis

task187の既存artifactには、exp031で確認したcomponent-fill規則を実装する過程で冗長なmask nodeが残っている。`safe_name_75` はすでに `safe_name_16` でmask済みなので、`safe_name_76 = And(safe_name_16, safe_name_75)` を `safe_name_75` に置換すればcostを下げられる。

## Result

- baseline cost: `{evaluation.baseline_cost}`
- candidate cost: `{evaluation.candidate_cost}`
- validation: `{evaluation.validation_status}`
- status: `{evaluation.status}`
- local estimate: `{local_estimate:.6f}`
- delta: `{delta:.6f}`
- gap to 6500: `{6500.0 - local_estimate:.6f}`
- submission decision: `{result["submission_decision"]}`

## Interpretation

1 nodeのgraph surgeryでtask187は `105313 -> {evaluation.candidate_cost}` に改善した。
6500 thresholdには届かないため提出しない。
この方向は大きな点数にはならないが、既存artifactの小さい冗長削除としてsubmit-safe候補の積み上げに使える。

## Risks

- leakage risk: {result["leakage_risk"]}
- overfitting risk: {result["overfitting_risk"]}
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
