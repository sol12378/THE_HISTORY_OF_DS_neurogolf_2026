import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from experiments.neurogolf_farm import FarmConfig, FarmRunner


def main() -> None:
    exp_dir = WORKSPACE / "experiments" / "exp129_compiler_farm_core"
    runner = FarmRunner(FarmConfig(workspace=WORKSPACE, experiment_dir=exp_dir))
    result = runner.run_smoke()
    FarmRunner.write_notes(exp_dir / "notes.md", result)


if __name__ == "__main__":
    main()
