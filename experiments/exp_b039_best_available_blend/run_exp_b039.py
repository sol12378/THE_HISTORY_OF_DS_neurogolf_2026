"""
exp_b039: 全有効ソースを対象とした最善ブレンド

知見まとめ:
- golfドメイン(beicicc)は2026-04-30より禁止 → 使用不可
- static cost削減はExpand/Scalarトリック → official ORT costは変わらない
- jonathanchan_v3がtask133で genuine改善: 193921→151176
- 現状: local 6282.81, LB 5930.40, gap=-352

戦略:
1. 全タスクで全有効ソース(標準ONNX only)をスキャン
2. static cost が安い候補のみ official ORT cost をプロファイリング
3. official cost が best より安く、train+test validation をパスしたもののみ採用
4. cost=0 タスクは完璧なので絶対に上書きしない
"""
from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import sys
import time
import tempfile
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
import importlib.util

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
ARTIFACT_DIR = ROOT / "data" / "cache" / "public_artifacts"
EXP_DIR = ROOT / "experiments" / "exp_b039_best_available_blend"

BANNED_OPS_SET = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
ALLOWED_DOMAINS = {"", "ai.onnx"}
NUM_TASKS = 400
MAX_BYTES = int(1.44 * 1024 * 1024)

UTILS_DIR = DATA_DIR / "neurogolf_utils"
sys.path.insert(0, str(UTILS_DIR))
spec = importlib.util.spec_from_file_location("neurogolf_utils", UTILS_DIR / "neurogolf_utils.py")
ng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ng)

# 全ての有効ソース（golf禁止、Compress禁止）
SOURCES: dict[str, pathlib.Path] = {}

def _add_source(name: str):
    d = ARTIFACT_DIR / name
    if (d / "submission").exists():
        SOURCES[name] = d / "submission"
    elif d.exists():
        SOURCES[name] = d

for n in [
    "jonathanchan_v3", "biohack44_6113_bundle", "biohack44_6115_prune_output",
    "nadeem_6151_surgery_output", "nadeem_6130_prune_output",
    "jsrdcht_6029_bundle", "needless090_onnx_v31", "agentzz_6284_zip",
    "jvlegend_conservative_v136_plus13", "octaviograu_manual_v205",
    "octaviograu_6154_rewrites_output", "konbu17_v117", "konbu17_blend_source_v360",
    "konbu17_blended_341", "konbu17_blended_4_27", "franksunp_super_blend_v2_output",
    "franksunp_blended_best", "franksunp_v18_multi_source", "franksunp_blended_v24",
    "massimilianoghiotto_6208", "massimilianoghiotto_6110", "massimilianoghiotto_6254_new",
    "seddik_surgical_output", "massimilianoghiotti_conv_part3",
    "biohack44_superior_blend_output", "biohack44_super_best_output",
    "vyanktesh_multi_source_output", "nadeem_6252_baseline_output",
    "magmacot_new_blending_output", "haoranran_6100_output",
    "tonylica_google_public", "jsrdcht_6029_bundle",
]:
    _add_source(n)

BEST_ZIP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "submission.zip"
NEW_ZIP = EXP_DIR / "submission.zip"
BEST_MANIFEST = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "selected_manifest.csv"

DT_BYTES = {1: 4, 2: 1, 3: 1, 4: 2, 5: 4, 6: 8, 7: 8, 8: 16, 9: 1, 10: 4, 11: 8, 12: 4, 13: 4}


def grid_to_f32(grid):
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if r < 30 and c < 30 and 0 <= color < 10:
                arr[0, color, r, c] = 1.0
    return arr


def f32_to_grid(arr):
    result = []
    for r in range(arr.shape[2]):
        row = []
        for c in range(arr.shape[3]):
            colors = [col for col in range(arr.shape[1]) if arr[0, col, r, c] == 1.0]
            row.append(colors[0] if len(colors) == 1 else (11 if colors else 10))
        while row and row[-1] == 10:
            row.pop()
        result.append(row)
    while result and not result[-1]:
        result.pop()
    return result


def static_cost(data: bytes) -> int | None:
    try:
        model = onnx.load_model_from_string(data)
        params = mem = 0
        for t in model.graph.initializer:
            if not t.dims:
                continue
            n = math.prod(t.dims)
            params += n
            mem += n * DT_BYTES.get(t.data_type, 4)
        for node in model.graph.node:
            if node.op_type != "Constant":
                continue
            for attr in node.attribute:
                if attr.name == "value":
                    n = math.prod(attr.t.dims) if attr.t.dims else 1
                    params += n
        return mem + params
    except:
        return None


def is_valid(data: bytes) -> tuple[bool, str]:
    if len(data) > MAX_BYTES:
        return False, "too_large"
    try:
        model = onnx.load_model_from_string(data)
    except Exception as e:
        return False, f"parse:{e}"
    if model.functions:
        return False, "functions"
    for node in model.graph.node:
        if node.op_type.upper() in BANNED_OPS_SET:
            return False, f"banned:{node.op_type}"
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return False, "subgraph"
    for opset in model.opset_import:
        if opset.domain not in ALLOWED_DOMAINS:
            return False, f"custom_domain:{opset.domain}"
    real_inputs = [i for i in model.graph.input
                   if not any(i.name == t.name for t in model.graph.initializer)]
    if len(real_inputs) != 1 or len(model.graph.output) != 1:
        return False, "not_single_io"
    try:
        onnx.checker.check_model(model)
        onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        return False, f"shape:{e}"
    return True, "ok"


def normalize_io(model):
    real_inputs = [i for i in model.graph.input
                   if not any(i.name == t.name for t in model.graph.initializer)]
    if not real_inputs or not model.graph.output:
        return model
    rename = {}
    if real_inputs[0].name != "input":
        rename[real_inputs[0].name] = "input"
        real_inputs[0].name = "input"
    if model.graph.output[0].name != "output":
        rename[model.graph.output[0].name] = "output"
        model.graph.output[0].name = "output"
    if not rename:
        return model
    for node in model.graph.node:
        node.input[:] = [rename.get(x, x) for x in node.input]
        node.output[:] = [rename.get(x, x) for x in node.output]
    for v in model.graph.value_info:
        v.name = rename.get(v.name, v.name)
    for t in model.graph.initializer:
        t.name = rename.get(t.name, t.name)
    return model


def validate(model_bytes: bytes, task: dict) -> bool:
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(model_bytes, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
    except Exception:
        return False
    for split in ("train", "test"):
        for ex in task.get(split, []):
            inp_g, out_g = ex["input"], ex["output"]
            if not inp_g or max(len(inp_g), max(len(r) for r in inp_g)) > 30:
                return False
            try:
                pred = sess.run(None, {inp_name: grid_to_f32(inp_g)})[0]
                if f32_to_grid(pred) != out_g:
                    return False
            except Exception:
                return False
    return True


def official_cost_safe(model_bytes: bytes, ex_input) -> int | None:
    tmp_pfx = pathlib.Path(tempfile.mktemp(suffix="", prefix="ng_"))
    try:
        opts = ort.SessionOptions()
        opts.enable_profiling = True
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.profile_file_prefix = str(tmp_pfx)
        sess = ort.InferenceSession(model_bytes, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        sess.run(None, {sess.get_inputs()[0].name: grid_to_f32(ex_input)})
        trace_path = sess.end_profiling()
        model = onnx.load_model_from_string(model_bytes)
        mem, params = ng.score_network(model, trace_path)
        try:
            pathlib.Path(trace_path).unlink(missing_ok=True)
        except:
            pass
        if mem is None or params is None:
            return static_cost(model_bytes)
        return int(mem) + int(params)
    except Exception:
        return None
    finally:
        for p in tmp_pfx.parent.glob(tmp_pfx.name + "*.json"):
            try:
                p.unlink()
            except:
                pass


def pts(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def main():
    t0 = time.time()
    os.makedirs(EXP_DIR, exist_ok=True)

    # Load best costs
    best_costs: dict[int, int] = {}
    with open(BEST_MANIFEST) as f:
        for row in csv.DictReader(f):
            cost_raw = row.get("cost", "")
            cost_val = int(cost_raw) if cost_raw and cost_raw.strip() else 0
            best_costs[int(row["task_id"])] = cost_val

    zero_cost_tasks = {tid for tid, c in best_costs.items() if c == 0}
    print(f"Best manifest: {len(best_costs)} tasks, perfect(cost=0): {sorted(zero_cost_tasks)}")
    print(f"Available sources: {len(SOURCES)}")

    # Per-task: find best candidate across all sources
    updates: dict[int, tuple[str, bytes, int]] = {}

    with zipfile.ZipFile(BEST_ZIP) as bz:
        for tid in range(1, NUM_TASKS + 1):
            fname = f"task{tid:03d}.onnx"

            # Never touch perfect tasks
            if tid in zero_cost_tasks:
                continue

            best_c = best_costs.get(tid, 0)
            if best_c == 0:
                # Not in manifest but check if in zip
                try:
                    best_data = bz.read(fname)
                    sc = static_cost(best_data)
                    best_c = sc if sc else 999999
                except:
                    best_c = 999999

            # Find the cheapest static-cost valid candidate across all sources
            candidates: list[tuple[str, bytes, int]] = []
            for src_label, src_dir in SOURCES.items():
                p = src_dir / fname
                if not p.exists():
                    continue
                try:
                    data = p.read_bytes()
                except:
                    continue
                sc = static_cost(data)
                if sc is None or sc == 0 or sc >= best_c:
                    continue
                # Quick validity check (no shape inference yet)
                try:
                    model = onnx.load_model_from_string(data)
                    domains = {op.domain for op in model.opset_import}
                    if not domains.issubset(ALLOWED_DOMAINS):
                        continue
                    ops = {n.op_type.upper() for n in model.graph.node}
                    if ops & BANNED_OPS_SET:
                        continue
                    if len(data) > MAX_BYTES:
                        continue
                except:
                    continue
                candidates.append((src_label, data, sc))

            if not candidates:
                continue

            # Sort by static cost and try in order
            candidates.sort(key=lambda x: x[2])
            with open(DATA_DIR / f"task{tid:03d}.json") as f:
                task = json.load(f)

            best_update = None
            best_update_cost = best_c

            for src_label, data, sc in candidates[:5]:  # try top 5 cheapest
                ok, reason = is_valid(data)
                if not ok:
                    continue
                try:
                    model = normalize_io(onnx.load_model_from_string(data))
                    model_bytes = model.SerializeToString()
                except:
                    continue

                if not validate(model_bytes, task):
                    continue

                # Official cost
                ex_input = task["train"][0]["input"] if task.get("train") else None
                if ex_input is None:
                    continue
                oc = official_cost_safe(model_bytes, ex_input)
                if oc is None or oc >= best_update_cost:
                    continue

                best_update = (src_label, model_bytes, oc)
                best_update_cost = oc

            if best_update:
                src_label, model_bytes, oc = best_update
                gain = pts(oc) - pts(best_c)
                elapsed = time.time() - t0
                print(f"  task{tid:03d}: ACCEPTED {best_c}->{oc} +{gain:.4f}pts src={src_label} t={elapsed:.0f}s")
                updates[tid] = best_update

    print(f"\nTotal updates: {len(updates)}")

    # Build submission
    manifest_rows = []
    source_counts: dict[str, int] = {}
    actual_gain = 0.0

    with zipfile.ZipFile(BEST_ZIP) as bz, zipfile.ZipFile(NEW_ZIP, "w", zipfile.ZIP_DEFLATED) as nz:
        for tid in range(1, NUM_TASKS + 1):
            fname = f"task{tid:03d}.onnx"
            if tid in updates:
                src_label, model_bytes, oc = updates[tid]
                nz.writestr(fname, model_bytes)
                cost = oc
                source_counts[src_label] = source_counts.get(src_label, 0) + 1
                actual_gain += pts(oc) - pts(best_costs.get(tid, oc))
            else:
                try:
                    data = bz.read(fname)
                    nz.writestr(fname, data)
                    cost = best_costs.get(tid, static_cost(data) or 0)
                    source_counts["exp_b025_best"] = source_counts.get("exp_b025_best", 0) + 1
                except Exception:
                    print(f"  WARN: missing task{tid:03d}")
                    continue
            manifest_rows.append({"task_id": tid, "cost": cost})

    new_total = sum(pts(r["cost"]) for r in manifest_rows)
    prev_total = 6282.812218

    with open(EXP_DIR / "selected_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["task_id", "cost"])
        w.writeheader()
        for r in manifest_rows:
            w.writerow(r)

    result = {
        "exp": "exp_b039_best_available_blend",
        "local_estimate": round(new_total, 6),
        "prev_local": prev_total,
        "delta": round(new_total - prev_total, 6),
        "n_tasks_updated": len(updates),
        "actual_gain": round(actual_gain, 6),
        "source_counts": dict(sorted(source_counts.items(), key=lambda x: -x[1])),
        "elapsed_s": round(time.time() - t0, 1),
        "note": "golf domain banned since 2026-04-30; static cost tricks do not help",
    }

    with open(EXP_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
