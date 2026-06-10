from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "exp130_public_code_6285_floor"
SOURCE_EXP = ROOT / "experiments" / "exp_b037_beicicc_golf_blend"
SOURCE_ZIP = SOURCE_EXP / "submission.zip"
SOURCE_RESULT = SOURCE_EXP / "result.json"
SOURCE_MANIFEST = SOURCE_EXP / "selected_manifest.csv"
OUT_ZIP = EXP_DIR / "submission.zip"
OUT_MANIFEST = EXP_DIR / "selected_manifest.csv"
MAX_ONNX_BYTES = int(1.44 * 1024 * 1024)
BANNED_OPS = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sanity_zip(path: Path) -> dict[str, object]:
    parse_failures: list[str] = []
    too_large: list[str] = []
    banned: list[str] = []
    golf_domain_count = 0
    op_counts: dict[str, int] = {}
    with zipfile.ZipFile(path) as zf:
        names = sorted(zf.namelist())
        for name in names:
            data = zf.read(name)
            if len(data) > MAX_ONNX_BYTES:
                too_large.append(name)
            try:
                model = onnx.load_model_from_string(data)
            except Exception as exc:
                parse_failures.append(f"{name}:{str(exc)[:120]}")
                continue
            for node in model.graph.node:
                op = node.op_type.upper()
                op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
                if op in BANNED_OPS:
                    banned.append(f"{name}:{node.op_type}")
                if node.domain and "golf" in node.domain.lower():
                    golf_domain_count += 1
    return {
        "count": len(names),
        "names_ok": len(names) == 400 and names[0] == "task001.onnx" and names[-1] == "task400.onnx",
        "first": names[:3],
        "last": names[-3:],
        "zip_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "parse_failures": parse_failures[:20],
        "parse_failure_count": len(parse_failures),
        "too_large_count": len(too_large),
        "banned_count": len(banned),
        "banned_examples": banned[:20],
        "golf_domain_node_count": golf_domain_count,
        "top_ops": sorted(op_counts.items(), key=lambda item: item[1], reverse=True)[:20],
    }


def read_manifest(path: Path) -> dict[str, object]:
    rows = []
    source_counts: dict[str, int] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                source = row.get("source", "")
                source_counts[source] = source_counts.get(source, 0) + 1
    return {"row_count": len(rows), "source_counts": source_counts}


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_ZIP, OUT_ZIP)
    shutil.copyfile(SOURCE_MANIFEST, OUT_MANIFEST)
    source_result = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    sanity = sanity_zip(OUT_ZIP)
    manifest = read_manifest(OUT_MANIFEST)
    result = {
        "exp": "exp130_public_code_6285_floor",
        "status": "submission_floor_candidate_ready",
        "floor_target_lb": 6285.0,
        "floor_basis": "public CODE beicicc golf-domain reference LB 6645 plus exp_b025 fallback",
        "source_exp": "exp_b037_beicicc_golf_blend",
        "source_result": source_result,
        "sanity": sanity,
        "manifest": manifest,
        "adoption": {
            "submit_as_floor_candidate": True,
            "floor_confirmed_by_this_workspace": False,
            "confirmation_required": "Kaggle LB result for this exact zip",
            "risk": "medium_high: golf domain static local estimate is not official cost; public-code overfit risk remains",
        },
        "outputs": {
            "submission_zip": str(OUT_ZIP),
            "selected_manifest": str(OUT_MANIFEST),
        },
        "elapsed_s": round(time.time() - t0, 3),
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (EXP_DIR / "notes.md").write_text(
        "\n".join(
            [
                "# exp130_public_code_6285_floor",
                "",
                "## Hypothesis",
                "",
                "public CODEのbeicicc golf-domain artifactを提出候補へ昇格すれば、LB 6285を下限化できる可能性がある。",
                "",
                "## Result",
                "",
                f"- source: `{result['source_exp']}`",
                f"- status: `{result['status']}`",
                f"- zip: `{OUT_ZIP}`",
                f"- sha256: `{sanity['sha256']}`",
                f"- names_ok: `{sanity['names_ok']}` / count `{sanity['count']}`",
                f"- parse_failure_count: `{sanity['parse_failure_count']}`",
                f"- too_large_count: `{sanity['too_large_count']}`",
                f"- banned_count: `{sanity['banned_count']}`",
                f"- manifest source_counts: `{manifest['source_counts']}`",
                "",
                "## Interpretation",
                "",
                "これは6285 floorの提出候補であり、workspace内だけではfloor確定ではない。golf domain artifactはpublic reference LB 6645を持つ一方、local official costは静的代用なので、最終判断はこのzipのKaggle LBで行う。",
                "",
                "## Leakage / Overfitting Risk",
                "",
                "public CODE由来であり、public LB overfit riskがある。private performanceを保証しない。採用はLB下限確保用の候補として扱い、rule/compiler本体の改善とは分ける。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
