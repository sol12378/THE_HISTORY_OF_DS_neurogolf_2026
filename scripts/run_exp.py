from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a NeuroGolf experiment.")
    parser.add_argument("exp_id", help="Experiment id, e.g. exp001_baseline")
    args = parser.parse_args()

    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    result_path = exp_dir / "result.json"
    if not result_path.exists():
        result_path.write_text(json.dumps({"exp_id": args.exp_id, "status": "created"}, indent=2), encoding="utf-8")

    print(f"Prepared {exp_dir}")


if __name__ == "__main__":
    main()
