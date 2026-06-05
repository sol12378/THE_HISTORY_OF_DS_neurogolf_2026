from __future__ import annotations

from pathlib import Path


def main() -> None:
    Path("data/folds").mkdir(parents=True, exist_ok=True)
    print("TODO: implement NeuroGolf validation split generation.")


if __name__ == "__main__":
    main()
