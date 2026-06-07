from __future__ import annotations

import csv
import json
import pathlib
import sys
import zipfile
from dataclasses import asdict
from typing import Iterable

import onnx

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "experiments"))

from phase1_rewrite_utils import (  # noqa: E402
    Candidate,
    BaseTask,
    candidate_fields,
    evaluate_candidate,
    load_base_tasks,
    load_neurogolf_utils,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp033_redundant_and_surgery_sweep"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp032_task187_component_lowering"
FALLBACK_BASE_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
ARC_GEN_SAMPLE = 20
TOP_K = 120


def base_exp_dir() -> pathlib.Path:
    return BASE_EXP if (BASE_EXP / "submission.zip").exists() else FALLBACK_BASE_EXP


def top_task_ids(base: dict[int, BaseTask], top_k: int) -> list[int]:
    return [item.task_id for item in sorted(base.values(), key=lambda x: (-x.cost, x.task_id))[:top_k]]


def replace_all_inputs(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for idx, input_name in enumerate(node.input):
            if input_name == old:
                node.input[idx] = new
    for output in model.graph.output:
        if output.name == old:
            output.name = new


def remove_node_by_output(model: onnx.ModelProto, output_name: str) -> None:
    kept = [node for node in model.graph.node if output_name not in set(node.output)]
    del model.graph.node[:]
    model.graph.node.extend(kept)


def and_surgery_candidates(task_id: int, raw: bytes, route: str) -> Iterable[Candidate]:
    model = onnx.load_model_from_string(raw)
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "And" or len(node.input) != 2 or len(node.output) != 1:
            continue
        output_name = node.output[0]
        for input_idx, input_name in enumerate(node.input):
            candidate_model = onnx.load_model_from_string(raw)
            replace_all_inputs(candidate_model, output_name, input_name)
            remove_node_by_output(candidate_model, output_name)
            yield Candidate(
                task_id,
                f"redundant_and_node{idx}_to_input{input_idx}",
                route,
                candidate_model.SerializeToString(),
                "generated",
                f"replace {output_name}=And({node.input[0]},{node.input[1]}) with {input_name}",
            )


def load_raws(exp_dir: pathlib.Path) -> dict[int, bytes]:
    raws: dict[int, bytes] = {}
    with zipfile.ZipFile(exp_dir / "submission.zip") as zf:
        for name in zf.namelist():
            task_id = int(pathlib.Path(name).stem.replace("task", ""))
            raws[task_id] = zf.read(name)
    return raws


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    source_exp = base_exp_dir()
    utils = load_neurogolf_utils()
    base = load_base_tasks(FALLBACK_BASE_EXP)
    raws = load_raws(source_exp)
    if source_exp != FALLBACK_BASE_EXP:
        exp032_result = json.loads((source_exp / "result.json").read_text(encoding="utf-8"))
        ev = exp032_result["evaluation"]
        task_id = int(ev["task_id"])
        old = base[task_id]
        base[task_id] = BaseTask(
            task_id=task_id,
            cost=int(ev["candidate_cost"]),
            points=float(ev["candidate_points"]),
            source=EXP_ID,
            template_name=str(ev["template_name"]),
            route=old.route,
            raw=raws[task_id],
        )

    eval_rows = []
    selected: dict[int, bytes] = {}
    selected_eval = {}
    generated = 0

    for task_id in top_task_ids(base, TOP_K):
        task = base[task_id]
        for candidate in and_surgery_candidates(task_id, raws[task_id], task.route):
            generated += 1
            evaluation, accepted_raw = evaluate_candidate(utils, candidate, task, ARC_GEN_SAMPLE, EXP_DIR)
            eval_rows.append(asdict(evaluation))
            if accepted_raw is None:
                continue
            old = selected_eval.get(task_id)
            if old is None or float(evaluation.candidate_points) > float(old.candidate_points):
                selected[task_id] = accepted_raw
                selected_eval[task_id] = evaluation

    bundle_raws = dict(raws)
    bundle_raws.update(selected)
    sanity = {}
    if selected:
        write_zip(EXP_DIR / "submission.zip", bundle_raws)
        sanity = zip_sanity(EXP_DIR / "submission.zip")

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(eval_rows)

    selected_rows = [asdict(v) for _, v in sorted(selected_eval.items())]
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(selected_rows)

    base_local = sum(task.points for task in base.values())
    delta = sum(float(row["candidate_points"]) - float(row["baseline_points"]) for row in selected_rows)
    local_estimate = base_local + delta
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-06",
        "status": "improved" if selected else "no_gain",
        "base_exp": str(source_exp.relative_to(ROOT)),
        "base_local_estimate": base_local,
        "top_k": TOP_K,
        "arc_gen_sample": ARC_GEN_SAMPLE,
        "candidate_rows": len(eval_rows),
        "generated_candidate_count": generated,
        "improved_task_count": len(selected_rows),
        "local_estimate_delta": delta,
        "local_estimate": local_estimate,
        "gap_to_6500": 6500.0 - local_estimate,
        "submission_decision": "submit_threshold_reached" if local_estimate >= 6500.0 else "no_submit: below 6500 threshold",
        "selected_tasks": [row["task_id"] for row in selected_rows],
        "top_selected": sorted(selected_rows, key=lambda row: float(row["candidate_points"]) - float(row["baseline_points"]), reverse=True)[:20],
        "zip_sanity": sanity,
        "leakage_risk": "medium: graph surgery preserves candidate semantics under sample20 validation, but base includes high-risk lookup artifacts.",
        "overfitting_risk": "medium: node replacement is validated on train/test/arc-gen sample20, not full private-like holdout.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## Hypothesis

exp032で有効だった `And(mask, already_masked)` 型の冗長nodeは、他の高cost artifactにも存在する。各 `And` nodeの出力を片方の入力に置換してもvalidationが通る場合、1 node分のmemory costを削減できる。

## Result

- base: `{result["base_exp"]}`
- top_k: `{TOP_K}`
- candidate rows: `{result["candidate_rows"]}`
- improved tasks: `{result["improved_task_count"]}`
- local estimate: `{result["local_estimate"]:.6f}`
- delta: `{result["local_estimate_delta"]:.6f}`
- gap to 6500: `{result["gap_to_6500"]:.6f}`
- submission decision: `{result["submission_decision"]}`

## Interpretation

改善候補はsample20 validationとofficial-like scoreを通したものだけ採用する。
6500未満なら提出しない。

## Risks

- leakage risk: {result["leakage_risk"]}
- overfitting risk: {result["overfitting_risk"]}
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
