from __future__ import annotations

import csv
import json
import pathlib
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


EXP_ID = "exp067_public_notebook_utilization_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp066_task020_correctness_onnx_lowering"
EXP002_BUILD = ROOT / "experiments" / "exp002_public_blend_6500_fast" / "build_public_blend.py"
KARNAK_CSV = EXP_DIR / "karnak_dataset" / "arc_primitives.csv"
TASK_TARGETS = ROOT / "experiments" / "exp053_all_task_cost_250_600_inventory" / "task_cost_targets.csv"


SAFE_UNIFORM_OPS = {
    "Greater",
    "Less",
    "Equal",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Where",
    "Max",
    "Min",
    "And",
    "Or",
    "Not",
    "Clip",
    "LessOrEqual",
    "GreaterOrEqual",
    "Sum",
}

NOTEBOOKS = [
    {
        "url": "https://www.kaggle.com/code/massimilianoghiotto/convolution-series-part-3",
        "slug": "massimilianoghiotto/convolution-series-part-3",
        "role": "convolution-series public solver lineage; outputs already represented by massimilianoghiotto datasets in exp002",
    },
    {
        "url": "https://www.kaggle.com/code/konbu17/neurogolf-2026-blended-till-4-27",
        "slug": "konbu17/neurogolf-2026-blended-till-4-27",
        "role": "blend source; related konbu17 public outputs already represented in exp002",
    },
    {
        "url": "https://www.kaggle.com/code/vyankteshdwivedi/neurogolf-multi-source-onnx-solver",
        "slug": "vyankteshdwivedi/neurogolf-multi-source-onnx-solver",
        "role": "multi-source ONNX output; already represented in exp002",
    },
    {
        "url": "https://www.kaggle.com/code/seddiktrk/surgical-onnx-precision-parameter-reduction",
        "slug": "seddiktrk/surgical-onnx-precision-parameter-reduction",
        "role": "surgical ONNX compression patterns: unused initializer prune, duplicate initializer dedup, uniform tensor scalarization",
    },
    {
        "url": "https://www.kaggle.com/code/karnakbaevarthur/all-task-description-analysis",
        "slug": "karnakbaevarthur/all-task-description-analysis",
        "role": "task transformation description library for compiler lane routing",
    },
    {
        "url": "https://www.kaggle.com/code/magmacot/neurogolf-new-blending",
        "slug": "magmacot/neurogolf-new-blending",
        "role": "blend output; already represented in exp002",
    },
    {
        "url": "https://www.kaggle.com/code/needless090/neurogolf-4250",
        "slug": "needless090/neurogolf-4250",
        "role": "baseline public output; already represented in exp002",
    },
]


@dataclass(frozen=True)
class NotebookUseRow:
    slug: str
    already_in_exp002: bool
    local_asset_status: str
    action: str
    role: str


@dataclass(frozen=True)
class SurgeryAuditRow:
    task_id: int
    cost: int
    local_points: float
    initializer_count: int
    initializer_params: int
    unused_count: int
    unused_params: int
    duplicate_groups: int
    duplicate_wasted_params: int
    uniform_safe_count: int
    uniform_safe_saveable_params: int
    total_seddik_style_saveable: int
    primary_category: str
    transformations: str
    cost_gap_to_600: float
    cost_gap_to_250: float


def read_manifest(exp_dir: pathlib.Path) -> dict[int, dict[str, str]]:
    manifest = exp_dir / "selected_manifest.csv"
    rows: dict[int, dict[str, str]] = {}
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["task_id"])] = row
    return rows


def model_raws(exp_dir: pathlib.Path) -> dict[int, bytes]:
    with zipfile.ZipFile(exp_dir / "submission.zip") as zf:
        return {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}


def source_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def notebook_use_rows() -> list[NotebookUseRow]:
    exp002 = source_text(EXP002_BUILD).lower()
    rows = []
    for nb in NOTEBOOKS:
        slug = nb["slug"]
        author, name = slug.split("/", 1)
        already = author.lower() in exp002 or name.lower() in exp002
        if "seddiktrk" in slug:
            status = "pulled_ipynb"
            action = "convert patterns into strict-seed surgery audit"
        elif "karnakbaevarthur" in slug:
            status = "pulled_ipynb_and_dataset"
            action = "join task descriptions to cost target queue"
        elif already:
            status = "already_blended_or_related_source"
            action = "do not re-submit; mine accepted ONNX structure and source provenance"
        else:
            status = "not_found"
            action = "review manually before using"
        rows.append(NotebookUseRow(slug, already, status, action, nb["role"]))
    return rows


def read_karnak() -> dict[int, dict[str, str]]:
    if not KARNAK_CSV.exists():
        return {}
    out: dict[int, dict[str, str]] = {}
    with KARNAK_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            m = re.search(r"(\d+)", row["Task_ID"])
            if m:
                out[int(m.group(1))] = row
    return out


def read_cost_targets() -> dict[int, dict[str, str]]:
    if not TASK_TARGETS.exists():
        return {}
    out: dict[int, dict[str, str]] = {}
    with TASK_TARGETS.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["task_id"])] = row
    return out


def init_params(init: onnx.TensorProto) -> int:
    arr = numpy_helper.to_array(init)
    return int(arr.size)


def initializer_key(init: onnx.TensorProto) -> tuple[str, tuple[int, ...], bytes]:
    arr = numpy_helper.to_array(init)
    return (arr.dtype.str, tuple(arr.shape), arr.tobytes())


def surgery_audit_row(task_id: int, raw: bytes, manifest: dict[str, str], karnak: dict[str, str], target: dict[str, str]) -> SurgeryAuditRow:
    model = onnx.load_from_string(raw)
    used = {inp for node in model.graph.node for inp in node.input if inp}
    consumers: dict[str, list[str]] = defaultdict(list)
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                consumers[inp].append(node.op_type)
    inits = list(model.graph.initializer)
    params_by_name = {init.name: init_params(init) for init in inits}
    unused = [init for init in inits if init.name not in used]

    groups: dict[tuple[str, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    for init in inits:
        groups[initializer_key(init)].append(init.name)
    dup_groups = [names for names in groups.values() if len(names) > 1]
    duplicate_wasted = sum(sum(params_by_name[name] for name in names[1:]) for names in dup_groups)

    uniform_safe_count = 0
    uniform_saveable = 0
    for init in inits:
        if init.name not in used:
            continue
        arr = numpy_helper.to_array(init)
        if arr.size <= 1:
            continue
        if not np.all(arr == arr.flat[0]):
            continue
        ops = consumers.get(init.name, [])
        if ops and all(op in SAFE_UNIFORM_OPS for op in ops):
            uniform_safe_count += 1
            uniform_saveable += int(arr.size - 1)

    cost = int(float(manifest["cost"]))
    local_points = float(manifest["local_points"])
    return SurgeryAuditRow(
        task_id=task_id,
        cost=cost,
        local_points=local_points,
        initializer_count=len(inits),
        initializer_params=sum(params_by_name.values()),
        unused_count=len(unused),
        unused_params=sum(init_params(init) for init in unused),
        duplicate_groups=len(dup_groups),
        duplicate_wasted_params=duplicate_wasted,
        uniform_safe_count=uniform_safe_count,
        uniform_safe_saveable_params=uniform_saveable,
        total_seddik_style_saveable=sum(init_params(init) for init in unused) + duplicate_wasted + uniform_saveable,
        primary_category=karnak.get("Primary_Category", ""),
        transformations=karnak.get("All_Used_Transformations", ""),
        cost_gap_to_600=float(target.get("gain_to_600", 0.0) or 0.0),
        cost_gap_to_250=float(target.get("gain_to_250", 0.0) or 0.0),
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(BASE_EXP)
    raws = model_raws(BASE_EXP)
    karnak = read_karnak()
    targets = read_cost_targets()

    use_rows = notebook_use_rows()
    audit_rows = [
        surgery_audit_row(task_id, raws[task_id], manifest[task_id], karnak.get(task_id, {}), targets.get(task_id, {}))
        for task_id in sorted(raws)
    ]
    audit_rows.sort(key=lambda r: (r.total_seddik_style_saveable, r.cost_gap_to_250), reverse=True)

    with (EXP_DIR / "notebook_use_map.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(NotebookUseRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in use_rows])
    with (EXP_DIR / "seddik_surgery_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SurgeryAuditRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in audit_rows])

    category_counts = Counter(row.get("Primary_Category", "") for row in karnak.values())
    transformation_counts = Counter()
    for row in karnak.values():
        for item in row.get("All_Used_Transformations", "").split("|"):
            item = item.strip()
            if item:
                transformation_counts[item] += 1

    high_priority = [
        asdict(r)
        for r in audit_rows
        if r.total_seddik_style_saveable > 0 or r.cost_gap_to_250 >= 2.0
    ][:30]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "audit_ready",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "notebooks": [asdict(r) for r in use_rows],
        "seddik_audit": {
            "tasks_scanned": len(audit_rows),
            "tasks_with_unused": sum(1 for r in audit_rows if r.unused_params > 0),
            "tasks_with_duplicate_initializers": sum(1 for r in audit_rows if r.duplicate_wasted_params > 0),
            "tasks_with_uniform_safe": sum(1 for r in audit_rows if r.uniform_safe_saveable_params > 0),
            "total_unused_params": sum(r.unused_params for r in audit_rows),
            "total_duplicate_wasted_params": sum(r.duplicate_wasted_params for r in audit_rows),
            "total_uniform_safe_saveable_params": sum(r.uniform_safe_saveable_params for r in audit_rows),
            "top_candidates": high_priority[:10],
        },
        "karnak_description_library": {
            "rows": len(karnak),
            "primary_category_counts": dict(category_counts),
            "top_transformations": dict(transformation_counts.most_common(20)),
        },
        "decision": "Use Seddik-style surgery as a strict-seed post-pass queue, and Karnak descriptions as compiler-lane priors. Do not treat public blend outputs as submit-safe unless converted into explicit rules or official-valid deltas.",
        "leakage_risk": "low for audit; public notebook outputs remain teacher/provenance only unless revalidated.",
        "overfitting_risk": "low for audit; future use must pass full arc-gen and LB calibration.",
        "outputs": {
            "notebook_use_map": "notebook_use_map.csv",
            "seddik_surgery_audit": "seddik_surgery_audit.csv",
            "source_notebooks": "source_notebooks/",
            "karnak_dataset": "karnak_dataset/",
        },
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

ユーザー提示の公開notebook群を、既存blendの再利用ではなく、strict-safe compiler / surgery / task-priorへ変換する。

## 結果

- 既にexp002へ含まれていた系統: `massimilianoghiotto`, `konbu17`, `vyankteshdwivedi`, `magmacot`, `needless090`
- 新規取得: `seddiktrk/surgical-onnx-precision-parameter-reduction`
- 新規取得: `karnakbaevarthur/all-task-description-analysis`
- Karnak dataset: `arc_primitives.csv/json` を取得
- Seddik-style audit scanned: {len(audit_rows)} tasks
- unused params total: {sum(r.unused_params for r in audit_rows)}
- duplicate wasted params total: {sum(r.duplicate_wasted_params for r in audit_rows)}
- uniform safe saveable params total: {sum(r.uniform_safe_saveable_params for r in audit_rows)}

## 解釈

公開blendはexp002で既にかなり取り込み済み。今後の有意義な使い方は、提出済みartifactをそのまま混ぜることではなく、Seddik式の外科的圧縮をstrict seed post-passへ入れること、Karnakのtask descriptionをcompiler laneのpriorにすること。

## Decision

次は `seddik_surgery_audit.csv` の上位taskをfull-arc gated surgery候補にし、Karnak category/transformationsを `exp054` laneや `exp053` cost queueへjoinする。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
