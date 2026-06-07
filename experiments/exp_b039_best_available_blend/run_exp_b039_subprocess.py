"""
exp_b039: subprocess 分離で安全なblend
各モデルのORT profiling をサブプロセスで実行してセグフォルトを防ぐ
"""
from __future__ import annotations

import csv, json, math, os, pathlib, subprocess, sys, time, zipfile, tempfile
import numpy as np, onnx

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
ARTIFACT_DIR = ROOT / "data" / "cache" / "public_artifacts"
EXP_DIR = ROOT / "experiments" / "exp_b039_best_available_blend"
PROBE = EXP_DIR / "probe_task.py"

BANNED = {"LOOP","SCAN","NONZERO","UNIQUE","SCRIPT","FUNCTION","COMPRESS"}
ALLOWED_DOMAINS = {"", "ai.onnx"}
NUM_TASKS = 400
DT_BYTES = {1:4,2:1,3:1,4:2,5:4,6:8,7:8,8:16,9:1,10:4,11:8,12:4,13:4}

SOURCES: dict[str, pathlib.Path] = {}
def _add(name):
    d = ARTIFACT_DIR / name
    sub = d / "submission"
    SOURCES[name] = sub if sub.exists() else d

for n in [
    "jonathanchan_v3", "biohack44_6113_bundle", "biohack44_6115_prune_output",
    "nadeem_6151_surgery_output", "jsrdcht_6029_bundle", "needless090_onnx_v31",
    "agentzz_6284_zip", "jvlegend_conservative_v136_plus13",
    "octaviograu_manual_v205", "octaviograu_6154_rewrites_output",
    "konbu17_v117", "massimilianoghiotto_6208",
]:
    _add(n)

BEST_ZIP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "submission.zip"
NEW_ZIP = EXP_DIR / "submission.zip"
BEST_MANIFEST = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "selected_manifest.csv"

def static_cost(data):
    try:
        model = onnx.load_model_from_string(data)
        params = mem = 0
        for t in model.graph.initializer:
            if not t.dims: continue
            n = math.prod(t.dims); params += n; mem += n * DT_BYTES.get(t.data_type, 4)
        return mem + params
    except: return None

def pts(c): return max(1.0, 25.0 - math.log(max(1, c)))

def probe(tid, src_path, best_cost, timeout=30):
    """Run probe_task.py in subprocess. Returns (ok, new_cost, model_bytes) or (False,...)"""
    try:
        result = subprocess.run(
            [sys.executable, str(PROBE), str(tid), str(src_path), str(best_cost)],
            capture_output=True, text=True, timeout=timeout
        )
        out = result.stdout.strip()
        lines = out.split("\n")
        ok_line = next((l for l in lines if l.startswith("OK:")), None)
        model_line = next((l for l in lines if l.startswith("MODEL:")), None)
        if ok_line and model_line:
            new_cost = int(ok_line[3:])
            model_path = model_line[6:]
            model_bytes = pathlib.Path(model_path).read_bytes()
            try: pathlib.Path(model_path).unlink()
            except: pass
            return True, new_cost, model_bytes
        return False, None, None
    except subprocess.TimeoutExpired:
        return False, None, None
    except Exception:
        return False, None, None

def main():
    t0 = time.time()
    os.makedirs(EXP_DIR, exist_ok=True)

    best_costs: dict[int, int] = {}
    with open(BEST_MANIFEST) as f:
        for row in csv.DictReader(f):
            cr = row.get("cost",""); best_costs[int(row["task_id"])] = int(cr) if cr.strip() else 0

    zero_cost = {t for t,c in best_costs.items() if c == 0}
    print(f"Perfect tasks (skip): {sorted(zero_cost)}")
    print(f"Sources: {list(SOURCES.keys())}")

    updates: dict[int, tuple[str, bytes, int]] = {}

    with zipfile.ZipFile(BEST_ZIP) as bz:
        for tid in range(1, NUM_TASKS+1):
            if tid in zero_cost: continue
            fname = f"task{tid:03d}.onnx"
            try: best_data = bz.read(fname)
            except: continue

            best_c = best_costs.get(tid) if tid in best_costs else (static_cost(best_data) or 999999)
            # Handle cost=0 that means "not in manifest" differently from "perfect"
            if best_c == 0 and tid not in zero_cost:
                best_c = static_cost(best_data) or 999999

            best_update_cost = best_c
            best_update = None

            for src_label, src_dir in SOURCES.items():
                p = src_dir / fname
                if not p.exists(): continue
                # Static pre-filter
                sc = static_cost(p.read_bytes())
                if sc is None or sc == 0 or sc >= best_update_cost:
                    continue

                ok, new_cost, model_bytes = probe(tid, p, best_update_cost)
                if ok and new_cost and new_cost < best_update_cost:
                    best_update_cost = new_cost
                    best_update = (src_label, model_bytes, new_cost)
                    print(f"  task{tid:03d}: {best_c}->{new_cost} src={src_label} +{pts(new_cost)-pts(best_c):.3f}pts t={time.time()-t0:.0f}s")

            if best_update:
                updates[tid] = best_update

    print(f"\nTotal genuine improvements: {len(updates)}")

    # Build submission
    actual_gain = 0.0
    manifest_rows = []
    source_counts: dict[str, int] = {}

    with zipfile.ZipFile(BEST_ZIP) as bz, zipfile.ZipFile(NEW_ZIP, "w", zipfile.ZIP_DEFLATED) as nz:
        for tid in range(1, NUM_TASKS+1):
            fname = f"task{tid:03d}.onnx"
            if tid in updates:
                sl, mb, oc = updates[tid]
                nz.writestr(fname, mb)
                cost = oc
                source_counts[sl] = source_counts.get(sl, 0) + 1
                actual_gain += pts(oc) - pts(best_costs.get(tid, oc))
            else:
                try:
                    data = bz.read(fname)
                    nz.writestr(fname, data)
                    cost = best_costs.get(tid, static_cost(data) or 0)
                    source_counts["exp_b025_best"] = source_counts.get("exp_b025_best", 0) + 1
                except:
                    print(f"WARN: missing task{tid:03d}")
                    continue
            manifest_rows.append({"task_id": tid, "cost": cost})

    new_total = sum(pts(r["cost"]) for r in manifest_rows)

    with open(EXP_DIR / "selected_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["task_id","cost"]); w.writeheader()
        for r in manifest_rows: w.writerow(r)

    result = {
        "exp": "exp_b039_best_available_blend",
        "local_estimate": round(new_total, 6),
        "prev_local": 6282.812218,
        "delta": round(new_total - 6282.812218, 6),
        "n_tasks_updated": len(updates),
        "actual_gain": round(actual_gain, 6),
        "source_counts": dict(sorted(source_counts.items(), key=lambda x: -x[1])),
        "elapsed_s": round(time.time()-t0, 1),
    }
    with open(EXP_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
