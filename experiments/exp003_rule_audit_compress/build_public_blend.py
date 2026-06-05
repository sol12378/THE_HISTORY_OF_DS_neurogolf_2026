from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
ARTIFACT_DIR = ROOT / "data" / "cache" / "public_artifacts"
EXP_DIR = ROOT / "experiments" / "exp003_rule_audit_compress"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
SCORE_CACHE_PATH = EXP_DIR / "score_cache.jsonl"
TASK_RE = re.compile(r"^task(\d{3})\.onnx$")
MAX_ONNX_BYTES = int(1.44 * 1024 * 1024)
BANNED_OPS = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION"}
NUM_TASKS = 400


SOURCES = [
    {
        "label": "massimilianoghiotto_6254",
        "kind": "dataset",
        "ref": "massimilianoghiotto/neurogolf2026-6254",
        "priority": 0,
    },
    {
        "label": "afr1ste_5689_artifact",
        "kind": "dataset",
        "ref": "afr1ste/neurogolf-5689-51-current-rules-open-artifact",
        "priority": 1,
    },
    {
        "label": "jonathanchan_v3",
        "kind": "dataset",
        "ref": "jonathanchan/ngc26-public-v3",
        "priority": 2,
    },
    {
        "label": "konbu17_v117",
        "kind": "dataset",
        "ref": "konbu17/neurogolf-2026-blended-401-v117",
        "priority": 3,
    },
    {
        "label": "sigmaborov_test_golf",
        "kind": "dataset",
        "ref": "sigmaborov/test-golf",
        "priority": 4,
    },
    {
        "label": "needless090_4250_output",
        "kind": "kernel_output",
        "ref": "needless090/neurogolf-4250",
        "priority": 5,
    },
    {
        "label": "vyanktesh_multi_source_output",
        "kind": "kernel_output",
        "ref": "vyankteshdwivedi/neurogolf-multi-source-onnx-solver",
        "priority": 6,
    },
    {
        "label": "magmacot_new_blending_output",
        "kind": "kernel_output",
        "ref": "magmacot/neurogolf-new-blending",
        "priority": 7,
    },
    {
        "label": "beicicc_6645_artifact",
        "kind": "dataset",
        "ref": "beicicc/neurogolf-6645-39-open-submission-artifact",
        "priority": 8,
    },
    {
        "label": "afr1ste_6335_artifact",
        "kind": "dataset",
        "ref": "afr1ste/neurogolf-6335-19-controlled-public-artifact",
        "priority": 9,
    },
    {
        "label": "afr1ste_6323_artifact",
        "kind": "dataset",
        "ref": "afr1ste/neurogolf-6323-15-controlled-public-artifact",
        "priority": 10,
    },
    {
        "label": "agentzz_6284_zip",
        "kind": "dataset",
        "ref": "agentzz/neurogolf-6284-zip",
        "priority": 11,
    },
    {
        "label": "afr1ste_6285_artifact",
        "kind": "dataset",
        "ref": "afr1ste/neurogolf-6285-95-open-submission-artifact",
        "priority": 12,
    },
    {
        "label": "massimilianoghiotto_6208",
        "kind": "dataset",
        "ref": "massimilianoghiotto/neurogolf2026-6208",
        "priority": 13,
    },
    {
        "label": "octaviograu_manual_v205",
        "kind": "dataset",
        "ref": "octaviograu/neurogolf-manual-rewrites-v205",
        "priority": 14,
    },
    {
        "label": "biohack44_6113_bundle",
        "kind": "dataset",
        "ref": "biohack44/neurogolf-6113-bundle",
        "priority": 15,
    },
    {
        "label": "massimilianoghiotto_6112",
        "kind": "dataset",
        "ref": "massimilianoghiotto/neurogolf2026-6112",
        "priority": 16,
    },
    {
        "label": "jsrdcht_6029_bundle",
        "kind": "dataset",
        "ref": "jsrdcht/neurogolf-6029-submission-bundle",
        "priority": 17,
    },
    {
        "label": "franksunp_blended_best",
        "kind": "dataset",
        "ref": "franksunp/neurogolf-blended-best-models",
        "priority": 18,
    },
    {
        "label": "konbu17_blend_source_v360",
        "kind": "dataset",
        "ref": "konbu17/neurogolf-2026-blend-source-v3-6-0",
        "priority": 19,
    },
    {
        "label": "kojimar_5800_minimal_blend",
        "kind": "dataset",
        "ref": "kojimar/neurogolf-5800-55-minimal-onnx-blend-assets",
        "priority": 20,
    },
    {
        "label": "scottweeden_trace_dfa_output",
        "kind": "kernel_output",
        "ref": "scottweeden/neurogolf-trace-language-dfa-solvers",
        "priority": 21,
    },
    {
        "label": "nadeem_6252_baseline_output",
        "kind": "kernel_output",
        "ref": "nadeembinshajahan/6252-lb-neurogolf-6252-baseline",
        "priority": 22,
    },
    {
        "label": "biohack44_superior_blend_output",
        "kind": "kernel_output",
        "ref": "biohack44/neurogolf-superior-blend-notebook",
        "priority": 23,
    },
    {
        "label": "biohack44_super_best_output",
        "kind": "kernel_output",
        "ref": "biohack44/neurogolf-super-blend-best-public-score",
        "priority": 24,
    },
    {
        "label": "nadeem_6151_surgery_output",
        "kind": "kernel_output",
        "ref": "nadeembinshajahan/6151-lb-neurogolf-2026-stable-fp16-surgery",
        "priority": 25,
    },
    {
        "label": "octaviograu_6154_rewrites_output",
        "kind": "kernel_output",
        "ref": "octaviograu/6154-71-onnx-rewrites-hand-built-solvers",
        "priority": 26,
    },
    {
        "label": "biohack44_6115_prune_output",
        "kind": "kernel_output",
        "ref": "biohack44/neurogolf-2026-fp16-surgery-prune-blend-6115",
        "priority": 27,
    },
    {
        "label": "nadeem_6130_prune_output",
        "kind": "kernel_output",
        "ref": "nadeembinshajahan/neurogolf-2026-fp16-surgery-prune-blend-6130",
        "priority": 28,
    },
    {
        "label": "biohack44_6080_output",
        "kind": "kernel_output",
        "ref": "biohack44/neurogolf-new-best-public-v2-6080-ish",
        "priority": 29,
    },
    {
        "label": "biohack44_6078_output",
        "kind": "kernel_output",
        "ref": "biohack44/neurogolf-new-best-public-6078-ish",
        "priority": 30,
    },
    {
        "label": "haoranran_6100_output",
        "kind": "kernel_output",
        "ref": "haoranran/fp16-graph-surgery-v2-highest-public-6100",
        "priority": 31,
    },
    {
        "label": "franksunp_super_blend_v2_output",
        "kind": "kernel_output",
        "ref": "franksunp/neurogolf-super-blend-v2",
        "priority": 32,
    },
]


@dataclass
class Candidate:
    task_id: int
    filename: str
    source_label: str
    source_ref: str
    source_priority: int
    relative_path: str
    raw: bytes
    sha256: str = ""
    file_bytes: int = 0
    normalized_bytes: int = 0
    status: str = "pending"
    reject_reason: str = ""
    params: int | None = None
    memory_bytes: int | None = None
    cost: int | None = None
    simple_cost: int | None = None
    local_points: float | None = None
    validation_status: str = "not_run"
    validation_pass: int = 0
    validation_fail: int = 0
    normalized_raw: bytes | None = None


@dataclass
class SourceStatus:
    label: str
    kind: str
    ref: str
    status: str
    path: str = ""
    message: str = ""
    task_files: int = 0


def run_cmd(cmd: list[str], timeout: int = 600) -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return proc.returncode == 0, proc.stdout
    except Exception as exc:
        return False, repr(exc)


def kaggle_exe() -> str:
    exe = ROOT / ".venv" / "Scripts" / "kaggle.exe"
    return str(exe) if exe.exists() else "kaggle"


def source_dir(source: dict[str, Any]) -> pathlib.Path:
    return ARTIFACT_DIR / source["label"]


def ensure_source(source: dict[str, Any], force_download: bool) -> SourceStatus:
    label = source["label"]
    dest = source_dir(source)
    dest.mkdir(parents=True, exist_ok=True)
    existing = list(dest.rglob("task*.onnx")) + list(dest.rglob("submission.zip"))
    if existing and not force_download:
        return SourceStatus(label, source["kind"], source["ref"], "cached", str(dest))

    cmd: list[str]
    if source["kind"] == "dataset":
        cmd = [
            kaggle_exe(),
            "datasets",
            "download",
            source["ref"],
            "-p",
            str(dest),
            "--unzip",
            "-o",
        ]
    elif source["kind"] == "kernel_output":
        cmd = [
            kaggle_exe(),
            "kernels",
            "output",
            source["ref"],
            "-p",
            str(dest),
            "-o",
        ]
    else:
        return SourceStatus(label, source["kind"], source["ref"], "blocked", str(dest), "unknown source kind")

    timeout = int(source.get("timeout", 45 if source["kind"] == "kernel_output" else 600))
    ok, out = run_cmd(cmd, timeout=timeout)
    status = "downloaded" if ok else "blocked"
    return SourceStatus(label, source["kind"], source["ref"], status, str(dest), out[-1000:])


def iter_onnx_blobs(base: pathlib.Path) -> list[tuple[str, bytes]]:
    blobs: list[tuple[str, bytes]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".onnx":
            blobs.append((str(path.relative_to(base)), path.read_bytes()))
        elif path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    for member in sorted(zf.namelist()):
                        if pathlib.PurePosixPath(member).name.endswith(".onnx"):
                            rel = f"{path.relative_to(base)}::{member}"
                            blobs.append((rel, zf.read(member)))
            except zipfile.BadZipFile:
                continue
    return blobs


def normalize_io(model: onnx.ModelProto) -> onnx.ModelProto | None:
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        return None
    rename: dict[str, str] = {}
    old_in = model.graph.input[0].name
    old_out = model.graph.output[0].name
    if old_in != "input":
        rename[old_in] = "input"
        model.graph.input[0].name = "input"
    if old_out != "output":
        rename[old_out] = "output"
        model.graph.output[0].name = "output"
    if not rename:
        return model
    for node in model.graph.node:
        node.input[:] = [rename.get(x, x) for x in node.input]
        node.output[:] = [rename.get(x, x) for x in node.output]
        for attr in node.attribute:
            if attr.HasField("g"):
                rename_graph(attr.g, rename)
            for graph in attr.graphs:
                rename_graph(graph, rename)
    for value in list(model.graph.value_info):
        value.name = rename.get(value.name, value.name)
    for init in model.graph.initializer:
        init.name = rename.get(init.name, init.name)
    return model


def rename_graph(graph: onnx.GraphProto, rename: dict[str, str]) -> None:
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        value.name = rename.get(value.name, value.name)
    for init in graph.initializer:
        init.name = rename.get(init.name, init.name)
    for node in graph.node:
        node.input[:] = [rename.get(x, x) for x in node.input]
        node.output[:] = [rename.get(x, x) for x in node.output]
        for attr in node.attribute:
            if attr.HasField("g"):
                rename_graph(attr.g, rename)
            for subgraph in attr.graphs:
                rename_graph(subgraph, rename)


def infer_static_ok(model: onnx.ModelProto) -> tuple[bool, str]:
    if model.functions:
        return False, "functions are not allowed"
    for node in model.graph.node:
        op = node.op_type.upper()
        if op in BANNED_OPS or "SEQUENCE" in op:
            return False, f"banned op {node.op_type}"
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return False, "subgraphs are not allowed"
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return False, f"custom domain {opset.domain}"
    try:
        onnx.checker.check_model(model, full_check=True)
        graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    except Exception as exc:
        return False, f"shape/check failed: {str(exc)[:160]}"
    seen: set[str] = set()
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value.name in seen:
            return False, f"duplicate value_info {value.name}"
        seen.add(value.name)
        if not value.type.HasField("tensor_type"):
            continue
        shape = value.type.tensor_type.shape
        if not shape:
            return False, f"missing shape {value.name}"
        for dim in shape.dim:
            if dim.HasField("dim_param"):
                return False, f"dynamic shape {value.name}"
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                return False, f"bad dim {value.name}"
    return True, "ok"


def load_neurogolf_utils():
    path = DATA_DIR / "neurogolf_utils" / "neurogolf_utils.py"
    spec = importlib.util.spec_from_file_location("neurogolf_utils_exp002", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["neurogolf_utils_exp002"] = module
    spec.loader.exec_module(module)
    return module


def task_examples(task_id: int, arc_gen_sample: int) -> list[dict[str, Any]]:
    path = DATA_DIR / f"task{task_id:03d}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["train"] + data["test"] + data["arc-gen"][:arc_gen_sample]


def validate_sample(utils: Any, session: ort.InferenceSession, examples: list[dict[str, Any]]) -> tuple[str, int, int]:
    passed = 0
    failed = 0
    for example in examples:
        benchmark = utils.convert_to_numpy(example)
        if benchmark is None:
            continue
        try:
            result = utils.run_network(session, benchmark["input"])
        except Exception:
            failed += 1
            continue
        if np.array_equal(result, benchmark["output"]):
            passed += 1
        else:
            failed += 1
    return ("pass" if failed == 0 else "fail"), passed, failed


def score_candidate(cand: Candidate, utils: Any, arc_gen_sample: int, validate: bool) -> Candidate:
    if not cand.sha256:
        cand.sha256 = hashlib.sha256(cand.raw).hexdigest()
    if not cand.file_bytes:
        cand.file_bytes = len(cand.raw)
    if cand.file_bytes > MAX_ONNX_BYTES:
        cand.status = "rejected"
        cand.reject_reason = "file too large"
        return cand
    match = TASK_RE.match(cand.filename)
    if not match or not (1 <= int(match.group(1)) <= NUM_TASKS):
        cand.status = "rejected"
        cand.reject_reason = "bad task filename"
        return cand
    try:
        model = onnx.load_model_from_string(cand.raw)
    except Exception as exc:
        cand.status = "rejected"
        cand.reject_reason = f"parse failed: {str(exc)[:160]}"
        return cand
    model = normalize_io(model)
    if model is None:
        cand.status = "rejected"
        cand.reject_reason = "not single input/output"
        return cand
    ok, reason = infer_static_ok(model)
    if not ok:
        cand.status = "rejected"
        cand.reject_reason = reason
        return cand
    sanitized = utils.sanitize_model(model)
    if sanitized is None:
        cand.status = "rejected"
        cand.reject_reason = "sanitize failed"
        return cand
    normalized_raw = sanitized.SerializeToString()
    cand.normalized_raw = normalized_raw
    cand.normalized_bytes = len(normalized_raw)
    try:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.profile_file_prefix = str(EXP_DIR / f"profile_task{cand.task_id:03d}_{cand.source_label}")
        session = ort.InferenceSession(normalized_raw, options, providers=["CPUExecutionProvider"])
    except Exception as exc:
        cand.status = "rejected"
        cand.reject_reason = f"ort load failed: {str(exc)[:160]}"
        return cand

    examples = task_examples(cand.task_id, arc_gen_sample)
    if validate:
        cand.validation_status, cand.validation_pass, cand.validation_fail = validate_sample(utils, session, examples)
    else:
        benchmark = utils.convert_to_numpy(examples[0])
        if benchmark is not None:
            try:
                utils.run_network(session, benchmark["input"])
            except Exception as exc:
                cand.status = "rejected"
                cand.reject_reason = f"profile run failed: {str(exc)[:160]}"
                return cand
        cand.validation_status = "not_run"

    trace_path = session.end_profiling()
    memory, params = utils.score_network(sanitized, trace_path)
    try:
        pathlib.Path(trace_path).unlink(missing_ok=True)
    except OSError:
        pass
    if memory is None or params is None:
        cand.status = "rejected"
        cand.reject_reason = "cost failed"
        return cand
    cand.memory_bytes = int(memory)
    cand.params = int(params)
    cand.cost = cand.memory_bytes + cand.params
    cand.simple_cost = cand.params + cand.normalized_bytes
    cand.local_points = max(1.0, 25.0 - math.log(max(1, cand.cost)))
    if cand.validation_status == "fail":
        cand.status = "rejected"
        cand.reject_reason = "sample validation failed"
    else:
        cand.status = "accepted"
    return cand


SCORE_FIELDS = [
    "sha256",
    "file_bytes",
    "normalized_bytes",
    "status",
    "reject_reason",
    "params",
    "memory_bytes",
    "cost",
    "simple_cost",
    "local_points",
    "validation_status",
    "validation_pass",
    "validation_fail",
]


def score_cache_key(cand: Candidate, arc_gen_sample: int, validate: bool) -> str:
    return f"{cand.task_id}:{cand.sha256}:{arc_gen_sample}:{int(validate)}"


def load_score_cache(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.pop("cache_key", "")
            if key:
                cache[key] = row
    return cache


def apply_score_cache(cand: Candidate, row: dict[str, Any]) -> Candidate:
    for field_name in SCORE_FIELDS:
        if field_name in row:
            setattr(cand, field_name, row[field_name])
    return cand


def append_score_cache(path: pathlib.Path, key: str, cand: Candidate) -> None:
    row = {"cache_key": key}
    row.update({field_name: getattr(cand, field_name) for field_name in SCORE_FIELDS})
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize_normalized_raw(cand: Candidate, utils: Any) -> Candidate:
    if cand.normalized_raw is not None:
        return cand
    model = onnx.load_model_from_string(cand.raw)
    model = normalize_io(model)
    if model is None:
        raise RuntimeError(f"selected task{cand.task_id:03d} could not be normalized")
    sanitized = utils.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError(f"selected task{cand.task_id:03d} could not be sanitized")
    cand.normalized_raw = sanitized.SerializeToString()
    return cand


def candidate_sort_key(cand: Candidate) -> tuple[int, int, int, str]:
    validation_rank = 0 if cand.validation_status == "pass" else 1
    cost_rank = int(cand.cost) if cand.cost is not None else 10**18
    return (validation_rank, cost_rank, cand.source_priority, cand.sha256)


def collect_candidates(source_statuses: list[SourceStatus], source_map: dict[str, dict[str, Any]]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for status in source_statuses:
        if status.status not in {"cached", "downloaded"}:
            continue
        source = source_map[status.label]
        for rel, raw in iter_onnx_blobs(pathlib.Path(status.path)):
            entry_name = rel.split("::")[-1].replace("\\", "/")
            name = pathlib.PurePosixPath(entry_name).name
            match = TASK_RE.match(name)
            if not match:
                continue
            task_id = int(match.group(1))
            if not (1 <= task_id <= NUM_TASKS):
                continue
            candidates.append(
                Candidate(
                    task_id=task_id,
                    filename=name,
                    source_label=status.label,
                    source_ref=status.ref,
                    source_priority=source["priority"],
                    relative_path=rel,
                    raw=raw,
                )
            )
    return candidates


def write_candidate_manifest(path: pathlib.Path, candidates: list[Candidate]) -> None:
    fields = [
        "task_id",
        "filename",
        "source_label",
        "source_ref",
        "relative_path",
        "sha256",
        "file_bytes",
        "normalized_bytes",
        "status",
        "reject_reason",
        "params",
        "memory_bytes",
        "cost",
        "simple_cost",
        "local_points",
        "validation_status",
        "validation_pass",
        "validation_fail",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cand in candidates:
            writer.writerow({field: getattr(cand, field) for field in fields})


def write_selected_manifest(path: pathlib.Path, selected: dict[int, Candidate]) -> None:
    fields = [
        "task_id",
        "filename",
        "source_label",
        "source_ref",
        "relative_path",
        "sha256",
        "file_bytes",
        "normalized_bytes",
        "params",
        "memory_bytes",
        "cost",
        "simple_cost",
        "local_points",
        "validation_status",
        "validation_pass",
        "validation_fail",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task_id in sorted(selected):
            cand = selected[task_id]
            writer.writerow({field: getattr(cand, field) for field in fields})


def write_zip(selected: dict[int, Candidate]) -> None:
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task_id in sorted(selected):
            raw = selected[task_id].normalized_raw
            if raw is None:
                raise RuntimeError(f"selected task{task_id:03d} missing normalized bytes")
            zf.writestr(f"task{task_id:03d}.onnx", raw)


def load_source_statuses(path: pathlib.Path) -> list[SourceStatus]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [SourceStatus(**row) for row in rows]


def save_source_statuses(path: pathlib.Path, statuses: list[SourceStatus]) -> None:
    path.write_text(json.dumps([s.__dict__ for s in statuses], ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--arc-gen-sample", type=int, default=20)
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--no-validate", action="store_false", dest="validate")
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--limit-sources", type=int, default=0)
    args = parser.parse_args()

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    source_map = {s["label"]: s for s in SOURCES}
    status_path = EXP_DIR / "source_status.json"

    if args.skip_download:
        source_statuses = load_source_statuses(status_path)
    else:
        source_statuses = []
        for source in SOURCES[: args.limit_sources or None]:
            print(f"Ensuring source {source['label']} ({source['ref']})", flush=True)
            status = ensure_source(source, args.force_download)
            status.task_files = len([p for p in pathlib.Path(status.path).rglob("task*.onnx")]) if status.path else 0
            source_statuses.append(status)
            save_source_statuses(status_path, source_statuses)
            print(f"  {status.status}: {status.task_files} task*.onnx files", flush=True)
        save_source_statuses(status_path, source_statuses)

    utils = load_neurogolf_utils()
    all_candidates = collect_candidates(source_statuses, source_map)
    if args.limit_tasks:
        all_candidates = [c for c in all_candidates if c.task_id <= args.limit_tasks]
    print(f"Collected {len(all_candidates)} candidates", flush=True)

    score_cache = load_score_cache(SCORE_CACHE_PATH)
    cache_hits = 0
    cache_misses = 0
    scored: list[Candidate] = []
    by_task_raw = Counter(c.task_id for c in all_candidates)
    for idx, cand in enumerate(all_candidates, start=1):
        if idx % 25 == 0:
            print(f"Scoring {idx}/{len(all_candidates)}", flush=True)
        cand.sha256 = hashlib.sha256(cand.raw).hexdigest()
        cand.file_bytes = len(cand.raw)
        cache_key = score_cache_key(cand, args.arc_gen_sample, args.validate)
        if cache_key in score_cache:
            scored.append(apply_score_cache(cand, score_cache[cache_key]))
            cache_hits += 1
        else:
            scored_cand = score_candidate(cand, utils, args.arc_gen_sample, args.validate)
            append_score_cache(SCORE_CACHE_PATH, cache_key, scored_cand)
            scored.append(scored_cand)
            cache_misses += 1

    accepted_by_task: dict[int, list[Candidate]] = defaultdict(list)
    for cand in scored:
        if cand.status == "accepted":
            accepted_by_task[cand.task_id].append(cand)

    selected: dict[int, Candidate] = {}
    for task_id, candidates in accepted_by_task.items():
        selected[task_id] = sorted(candidates, key=candidate_sort_key)[0]
    for cand in selected.values():
        materialize_normalized_raw(cand, utils)

    write_candidate_manifest(EXP_DIR / "candidate_manifest.csv", scored)
    write_selected_manifest(EXP_DIR / "selected_manifest.csv", selected)
    write_zip(selected)

    source_selected = Counter(c.source_label for c in selected.values())
    reject_counts = Counter(c.reject_reason for c in scored if c.status == "rejected")
    validation_counts = Counter(c.validation_status for c in selected.values())
    selected_score = sum(float(c.local_points or 0) for c in selected.values())
    missing_tasks = [i for i in range(1, NUM_TASKS + 1) if i not in selected]
    top_expensive = sorted(selected.values(), key=lambda c: int(c.cost or 0), reverse=True)[:25]
    result = {
        "exp_id": "exp003_rule_audit_compress",
        "status": "local_estimate_complete" if selected_score >= 6500 else "local_estimate_below_target",
        "date": "2026-06-05",
        "target": 6500,
        "local_estimate": selected_score,
        "selected_task_count": len(selected),
        "candidate_count": len(scored),
        "score_cache_hits": cache_hits,
        "score_cache_misses": cache_misses,
        "accepted_candidate_count": sum(1 for c in scored if c.status == "accepted"),
        "missing_task_count": len(missing_tasks),
        "missing_tasks": missing_tasks,
        "source_status": [s.__dict__ for s in source_statuses],
        "source_selected_counts": dict(source_selected),
        "validation_counts": dict(validation_counts),
        "reject_counts_top": reject_counts.most_common(30),
        "raw_candidate_task_coverage": len(by_task_raw),
        "submission_zip": str(OUTPUT_ZIP.relative_to(ROOT)),
        "submission_zip_bytes": OUTPUT_ZIP.stat().st_size,
        "top_expensive_tasks": [
            {
                "task_id": c.task_id,
                "source": c.source_label,
                "cost": c.cost,
                "points": c.local_points,
                "file_bytes": c.file_bytes,
            }
            for c in top_expensive
        ],
        "leakage_risk": "高: public artifact/sourceを利用した最速LB寄せ。private benchmarkやrule updateに弱い可能性がある。",
        "overfitting_risk": "高: public LB実績のある公開ONNX束をtask別にblendする方針であり、汎化目的ではない。",
        "rule_audit": {
            "compress_allowed": True,
            "still_rejected": ["Loop", "Scan", "NonZero", "Unique", "Script", "Function", "model functions", "subgraphs", "dynamic shapes", "custom domains"],
            "purpose": "Compressだけを許可した場合にlocal estimateと採用sourceがどれだけ変化するかを確認する。",
        },
        "next_actions": [
            "local estimateが6500以上なら、同じfilterでexp004 full arc-gen validationへ進む。",
            "6500未満なら、strict routeのtop expensive task rewriteを優先する。",
            "小さいzipのKaggle submit可否確認はユーザー確認後に行う。",
        ],
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
