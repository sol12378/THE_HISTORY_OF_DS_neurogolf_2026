from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_EXTS = {
    ".cfg",
    ".csv",
    ".editorconfig",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

MOJIBAKE_MARKERS = [
    "縺",
    "繧",
    "譛",
    "荳",
    "蜊",
    "鬮",
    "逕",
    "隱",
    "譁",
    "螳",
    "蟄",
    "蜷",
    "繝",
    "逶",
    "邨",
    "蜈",
    "�",
]

SKIP_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "data",
}


def iter_text_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in DEFAULT_EXTS or path.name == ".editorconfig":
            paths.append(path)
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check UTF-8 decoding and common Japanese mojibake markers.")
    parser.add_argument("paths", nargs="*", default=["."], help="Files or directories to check.")
    args = parser.parse_args()

    failures: list[str] = []
    files: list[Path] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        files.extend(iter_text_files(path) if path.is_dir() else [path])

    unique_files = sorted(set(files))
    for path in unique_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            failures.append(f"{path}: utf-8 decode error: {exc}")
            continue
        if path.name == "check_text_encoding.py":
            continue
        markers = [marker for marker in MOJIBAKE_MARKERS if marker in text]
        if markers:
            failures.append(f"{path}: mojibake markers found: {', '.join(markers)}")

    if failures:
        print("Encoding check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Encoding check passed: {len(unique_files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
