from __future__ import annotations

import hashlib
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP_ID = "exp331_task185_gather_axis_cost_shave"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_ZIP = ROOT / "experiments" / "exp297_exp262_skip_task048_336_fresh_candidates" / "submission.zip"
CANDIDATE = EXP_DIR / "gather_axis_float.onnx"
OUT_ZIP = EXP_DIR / "submission.zip"
TASK_NAME = "task185.onnx"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    result = json.loads((EXP_DIR / "result.json").read_text(encoding="utf-8"))
    candidate_raw = CANDIDATE.read_bytes()
    with zipfile.ZipFile(BASE_ZIP, "r") as zin:
        names = sorted(zin.namelist())
        if len(names) != 400:
            raise RuntimeError(f"base zip has {len(names)} files")
        if TASK_NAME not in names:
            raise RuntimeError(f"{TASK_NAME} missing from base zip")
        replaced_old_sha = sha256_bytes(zin.read(TASK_NAME))
        with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                payload = candidate_raw if name == TASK_NAME else zin.read(name)
                zout.writestr(name, payload)
    with zipfile.ZipFile(OUT_ZIP, "r") as zcheck:
        out_names = sorted(zcheck.namelist())
        if out_names != names:
            raise RuntimeError("output zip names differ from base")
        replaced_new_sha = sha256_bytes(zcheck.read(TASK_NAME))
    manifest = {
        "exp_id": EXP_ID,
        "base_zip": str(BASE_ZIP.relative_to(ROOT)),
        "output_zip": str(OUT_ZIP.relative_to(ROOT)),
        "file_count": len(out_names),
        "names_ok": out_names == names,
        "replaced_task": TASK_NAME,
        "old_task_sha256": replaced_old_sha,
        "new_task_sha256": replaced_new_sha,
        "new_task_matches_candidate": replaced_new_sha == sha256_bytes(candidate_raw),
        "expected_public_lb_if_calibrated": result["expected_public_lb_if_calibrated"],
        "expected_local_delta": result["expected_local_delta"],
        "candidate_cost": result["best_row"]["cost"],
        "baseline_cost": result["baseline_cost"],
        "submission_decision": "submit",
    }
    (EXP_DIR / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
