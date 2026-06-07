"""
exp_b035_new_source_full_arc_blend

新ソース（konbu17_blended_4_27, seddik_surgical_output）を加えた全arc-gen検証付きblend。
目標: LB6200超え

アプローチ:
1. 全キャッシュ済みソース + 新ソースを収集
2. task別に全候補をstatic validate + full arc-gen validate (neurogolf_utils使用)
3. min cost passing候補を選択
4. Seddik uniform initializer scalarization post-pass (full arc-gen gated)
5. submission.zip 生成
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import sys
import time
import zipfile
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
ARTIFACT_DIR = ROOT / "data" / "cache" / "public_artifacts"
EXP_DIR = ROOT / "experiments" / "exp_b035_new_source_full_arc_blend"
OUTPUT_ZIP = EXP_DIR / "submission.zip"

TASK_RE = re.compile(r"^task(\d{3})\.onnx$")
MAX_ONNX_BYTES = int(1.44 * 1024 * 1024)
BANNED_OPS_SET = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
NUM_TASKS = 400

# neurogolf_utilsのロード
UTILS_DIR = DATA_DIR / "neurogolf_utils"
sys.path.insert(0, str(UTILS_DIR))
spec = importlib.util.spec_from_file_location("neurogolf_utils", UTILS_DIR / "neurogolf_utils.py")
ng_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ng_module)
print("neurogolf_utils loaded")

# 入力変換関数
def grid_to_float32(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if r < 30 and c < 30 and 0 <= color < 10:
                arr[0, color, r, c] = 1.0
    return arr

def float32_to_grid(arr: np.ndarray) -> list[list[int]]:
    _, ch, h, w = arr.shape
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            colors = [col for col in range(ch) if arr[0, col, r, c] == 1.0]
            row.append(colors[0] if len(colors) == 1 else (11 if colors else 10))
        while row and row[-1] == 10:
            row.pop()
        result.append(row)
    while result and not result[-1]:
        result.pop()
    return result


# ソースリスト: (label, base_dir) - base_dirにonnxが直接入っている
# iter_onnx_files がbase_dir以下を再帰的に探す
SOURCES: list[tuple[str, pathlib.Path]] = []

def _add_src(label: str) -> None:
    base = ARTIFACT_DIR / label
    if base.exists():
        SOURCES.append((label, base))
    else:
        print(f"  WARN: source not found: {label}")

# 新ソース（優先度高 - リストの先頭）
_add_src("konbu17_blended_4_27")
_add_src("seddik_surgical_output")
# 既存ソース
_add_src("massimilianoghiotti_6254")
_add_src("massimilianoghiotti_6208")
_add_src("massimilianoghiotti_6112")
_add_src("afr1ste_5689_artifact")
_add_src("afr1ste_6335_artifact")
_add_src("afr1ste_6323_artifact")
_add_src("afr1ste_6285_artifact")
_add_src("beicicc_6645_artifact")
_add_src("jonathanchan_v3")
_add_src("konbu17_v117")
_add_src("konbu17_blend_source_v360")
_add_src("konbu17_blended_341")
_add_src("magmacot_new_blending_output")
_add_src("needless090_4250_output")
_add_src("needless090_onnx_v31")
_add_src("vyanktesh_multi_source_output")
_add_src("nadeem_6252_baseline_output")
_add_src("nadeem_6151_surgery_output")
_add_src("franksunp_blended_best")
_add_src("kojimar_5800_minimal_blend")
_add_src("sigmaborov_test_golf")
_add_src("octaviograu_6154_rewrites_output")
_add_src("octaviograu_manual_v205")
_add_src("jvlegend_conservative_v136_plus13")
_add_src("biohack44_6115_prune_output")
_add_src("biohack44_super_best_output")
_add_src("biohack44_superior_blend_output")
_add_src("haoranran_6100_output")
_add_src("gfarew_solution_pack")

print(f"Total sources: {len(SOURCES)}")

# current best submissionも候補として追加
CURRENT_BEST_ZIP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "submission.zip"


def iter_task_onnx(base: pathlib.Path, task_id: int) -> list[tuple[str, bytes]]:
    """baseディレクトリからtask_id対応のonnxを全て収集"""
    fname = f"task{task_id:03d}.onnx"
    blobs = []
    # 直接ファイル検索（subdirも含む）
    for path in sorted(base.rglob(fname)):
        try:
            blobs.append((str(path.relative_to(base)), path.read_bytes()))
        except Exception:
            pass
    # zipファイル内も検索
    for zip_path in sorted(base.rglob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    if pathlib.PurePosixPath(member).name == fname:
                        blobs.append((f"{zip_path.relative_to(base)}::{member}", zf.read(member)))
        except Exception:
            pass
    return blobs


def static_validate(data: bytes) -> tuple[bool, str, onnx.ModelProto | None]:
    if len(data) > MAX_ONNX_BYTES:
        return False, f"too_large:{len(data)}", None
    try:
        model = onnx.load_model_from_string(data)
    except Exception as e:
        return False, f"parse:{e}", None
    # banned ops
    for node in model.graph.node:
        if node.op_type.upper() in BANNED_OPS_SET:
            return False, f"banned:{node.op_type}", None
    # no functions
    if model.functions:
        return False, "functions_not_allowed", None
    # no subgraphs
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return False, "subgraphs_not_allowed", None
    # only default domain
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return False, f"custom_domain:{opset.domain}", None
    # single input/output
    real_inputs = [i for i in model.graph.input if not any(
        i.name == init.name for init in model.graph.initializer)]
    if len(real_inputs) != 1 or len(model.graph.output) != 1:
        return False, "not_single_io", None
    # shape check
    try:
        onnx.checker.check_model(model)
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        return False, f"shape_error:{e}", None
    for value in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info):
        if not value.type.HasField("tensor_type"):
            continue
        shape = value.type.tensor_type.shape
        if not shape:
            return False, f"missing_shape:{value.name}", None
        for dim in shape.dim:
            if dim.HasField("dim_param"):
                return False, f"dynamic:{value.name}", None
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                return False, f"bad_dim:{value.name}", None
    return True, "ok", model


def normalize_io(model: onnx.ModelProto) -> onnx.ModelProto:
    """input/outputを"input"/"output"に正規化"""
    real_inputs = [i for i in model.graph.input if not any(
        i.name == init.name for init in model.graph.initializer)]
    if not real_inputs or not model.graph.output:
        return model
    rename = {}
    old_in = real_inputs[0].name
    old_out = model.graph.output[0].name
    if old_in != "input":
        rename[old_in] = "input"
        real_inputs[0].name = "input"
    if old_out != "output":
        rename[old_out] = "output"
        model.graph.output[0].name = "output"
    if not rename:
        return model
    for node in model.graph.node:
        node.input[:] = [rename.get(x, x) for x in node.input]
        node.output[:] = [rename.get(x, x) for x in node.output]
    for v in model.graph.value_info:
        v.name = rename.get(v.name, v.name)
    for init in model.graph.initializer:
        init.name = rename.get(init.name, init.name)
    return model


def official_cost_profiling(model_bytes: bytes, example: dict) -> int | None:
    """ORT profilingを使った公式cost計算"""
    import tempfile
    tmp = pathlib.Path(tempfile.mktemp(suffix="", prefix="ng_prof_"))
    try:
        opts = ort.SessionOptions()
        opts.enable_profiling = True
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.profile_file_prefix = str(tmp)
        sess = ort.InferenceSession(model_bytes, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        inp_arr = grid_to_float32(example["input"])
        inp_name = sess.get_inputs()[0].name
        sess.run(None, {inp_name: inp_arr})
        trace_path = sess.end_profiling()
        model = onnx.load_model_from_string(model_bytes)
        mem, params = ng_module.score_network(model, trace_path)
        try:
            pathlib.Path(trace_path).unlink(missing_ok=True)
        except Exception:
            pass
        if mem is None or params is None:
            return None
        return int(mem) + int(params)
    except Exception:
        return None
    finally:
        for p in tmp.parent.glob(tmp.name + "*.json"):
            try:
                p.unlink()
            except Exception:
                pass


def validate_model_examples(model_bytes: bytes, examples: list[tuple]) -> tuple[bool, int, int]:
    """examples: list of (input_grid, output_grid). Returns (all_pass, n_pass, n_fail)"""
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(model_bytes, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
    except Exception:
        return False, 0, len(examples)
    n_pass = n_fail = 0
    for inp_grid, out_grid in examples:
        if max(len(inp_grid), len(inp_grid[0]) if inp_grid else 0) > 30:
            n_fail += 1
            continue
        try:
            inp_arr = grid_to_float32(inp_grid)
            pred_raw = sess.run(None, {inp_name: inp_arr})[0]
            pred_grid = float32_to_grid(pred_raw)
            if pred_grid == out_grid:
                n_pass += 1
            else:
                n_fail += 1
        except Exception:
            n_fail += 1
    return n_fail == 0, n_pass, n_fail


def load_task(task_id: int) -> dict:
    p = DATA_DIR / f"task{task_id:03d}.json"
    with open(p) as f:
        return json.load(f)


def task_examples_list(task: dict) -> list[tuple]:
    pairs = []
    for split in ("train", "test"):
        for ex in task.get(split, []):
            pairs.append((ex["input"], ex["output"]))
    return pairs


def arc_gen_examples_list(task: dict, max_n: int = 40) -> list[tuple]:
    pairs = []
    for ex in task.get("arc_gen", [])[:max_n]:
        pairs.append((ex["input"], ex["output"]))
    return pairs


def scalarize_uniform_initializers(model_bytes: bytes) -> tuple[bytes, int]:
    """Seddik-style uniform initializer scalarization"""
    try:
        model = onnx.load_model_from_string(model_bytes)
        graph = model.graph
        new_inits = []
        new_nodes = []
        replacements = {}
        n_changed = 0

        for init in graph.initializer:
            arr = numpy_helper.to_array(init)
            if arr.size <= 1 or not np.all(arr == arr.flat[0]):
                new_inits.append(init)
                continue
            scalar_name = init.name + "_sc_"
            shape_name = init.name + "_sh_"
            scalar_t = numpy_helper.from_array(np.array(arr.flat[0], dtype=arr.dtype), name=scalar_name)
            shape_t = numpy_helper.from_array(np.array(list(arr.shape), dtype=np.int64), name=shape_name)
            new_inits.extend([scalar_t, shape_t])
            expand_out = init.name + "_ex_"
            replacements[init.name] = expand_out
            new_nodes.append(onnx.helper.make_node("Expand", inputs=[scalar_name, shape_name], outputs=[expand_out]))
            n_changed += 1

        if not replacements:
            return model_bytes, 0

        for node in graph.node:
            node.input[:] = [replacements.get(x, x) for x in node.input]

        del graph.initializer[:]
        graph.initializer.extend(new_inits)
        existing_nodes = list(graph.node)
        del graph.node[:]
        graph.node.extend(new_nodes)
        graph.node.extend(existing_nodes)

        return model.SerializeToString(), n_changed
    except Exception:
        return model_bytes, 0


def points(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def main() -> None:
    t0 = time.time()
    ARC_GEN_MAX = 40

    os.makedirs(EXP_DIR, exist_ok=True)

    # current best costを読む
    best_costs: dict[int, int] = {}
    BEST_MANIFEST = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "selected_manifest.csv"
    if BEST_MANIFEST.exists():
        with open(BEST_MANIFEST) as f:
            for row in csv.DictReader(f):
                tid = int(row["task_id"])
                best_costs[tid] = int(row.get("cost", 0) or 0)
    else:
        print(f"WARN: best manifest not found at {BEST_MANIFEST}")

    manifest_rows = []
    total_points = 0.0
    source_counts: dict[str, int] = {}
    n_improved = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for task_id in range(1, NUM_TASKS + 1):
            task = load_task(task_id)
            tt_pairs = task_examples_list(task)
            arc_pairs = arc_gen_examples_list(task, max_n=ARC_GEN_MAX)

            # task0 example (for profiling)
            if tt_pairs:
                profile_ex = {"input": tt_pairs[0][0], "output": tt_pairs[0][1]}
            else:
                profile_ex = ng_module._TASK_ZERO["train"][0]

            # 全ソースからONNX候補を収集
            all_candidates: list[tuple[str, bytes]] = []

            # current best zip から
            if CURRENT_BEST_ZIP.exists():
                try:
                    with zipfile.ZipFile(CURRENT_BEST_ZIP) as zf:
                        fname = f"task{task_id:03d}.onnx"
                        if fname in zf.namelist():
                            all_candidates.append(("exp_b025_best", zf.read(fname)))
                except Exception:
                    pass

            # 各ソースから
            for label, base_dir in SOURCES:
                blobs = iter_task_onnx(base_dir, task_id)
                for rel_path, data in blobs:
                    all_candidates.append((label, data))

            if not all_candidates:
                print(f"task{task_id:03d}: NO CANDIDATES")
                continue

            # static validate + cost計算 + validation
            best_bytes = None
            best_cost = float("inf")
            best_label = None
            tried = 0
            rejected_static = 0
            rejected_val = 0

            for label, data in all_candidates:
                ok, reason, model = static_validate(data)
                if not ok:
                    rejected_static += 1
                    continue
                model = normalize_io(model)
                model_bytes = model.SerializeToString()
                tried += 1

                # full validation (train+test + arc-gen)
                all_pairs = tt_pairs + arc_pairs
                pass_all, n_pass, n_fail = validate_model_examples(model_bytes, all_pairs)
                if not pass_all:
                    rejected_val += 1
                    continue

                # cost計算
                cost = official_cost_profiling(model_bytes, profile_ex)
                if cost is None:
                    continue

                if cost < best_cost:
                    best_bytes = model_bytes
                    best_cost = cost
                    best_label = label

            if best_bytes is None:
                # arc-gen除外して再試行
                for label, data in all_candidates:
                    ok, reason, model = static_validate(data)
                    if not ok:
                        continue
                    model = normalize_io(model)
                    model_bytes = model.SerializeToString()
                    pass_tt, n_pass, n_fail = validate_model_examples(model_bytes, tt_pairs)
                    if not pass_tt:
                        continue
                    cost = official_cost_profiling(model_bytes, profile_ex)
                    if cost is None:
                        continue
                    if cost < best_cost:
                        best_bytes = model_bytes
                        best_cost = cost
                        best_label = label + "_tt_only"

            if best_bytes is None:
                print(f"task{task_id:03d}: ALL FAILED (tried={tried}, static_rej={rejected_static}, val_rej={rejected_val})")
                continue

            # Seddik post-pass
            new_bytes, n_scalar = scalarize_uniform_initializers(best_bytes)
            if n_scalar > 0:
                all_pairs = tt_pairs + arc_pairs
                pass_all, _, _ = validate_model_examples(new_bytes, all_pairs)
                if pass_all:
                    new_cost = official_cost_profiling(new_bytes, profile_ex)
                    if new_cost is not None and new_cost < best_cost:
                        best_bytes = new_bytes
                        best_cost = new_cost

            fname = f"task{task_id:03d}.onnx"
            zout.writestr(fname, best_bytes)

            pt = points(best_cost)
            total_points += pt
            source_counts[best_label] = source_counts.get(best_label, 0) + 1

            prev_cost = best_costs.get(task_id, best_cost)
            improved = prev_cost > 0 and best_cost < prev_cost
            if improved:
                n_improved += 1

            manifest_rows.append({
                "task_id": task_id,
                "source": best_label,
                "cost": best_cost,
                "points": round(pt, 6),
                "prev_cost": prev_cost,
                "improved": improved,
            })

            elapsed = time.time() - t0
            if task_id % 20 == 0 or improved:
                print(f"task{task_id:03d}: {best_label} cost={best_cost} pts={pt:.4f} "
                      f"{'IMPROVED' if improved else ''} | total={total_points:.2f} t={elapsed:.0f}s")

    # マニフェスト保存
    manifest_path = EXP_DIR / "selected_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "source", "cost", "points", "prev_cost", "improved"])
        w.writeheader()
        w.writerows(manifest_rows)

    result = {
        "exp": "exp_b035_new_source_full_arc_blend",
        "local_estimate": round(total_points, 6),
        "n_tasks": len(manifest_rows),
        "n_improved_vs_b025": n_improved,
        "source_counts": dict(sorted(source_counts.items(), key=lambda x: -x[1])),
        "elapsed_s": round(time.time() - t0, 1),
        "prev_best_local": 6282.812218,
        "delta": round(total_points - 6282.812218, 6),
    }
    with open(EXP_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
