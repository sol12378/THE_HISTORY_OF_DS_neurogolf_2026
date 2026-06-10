"""Registry for public CODE/notebook candidates and adoption gates."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class PublicCodeSource:
    source_id: str
    experiment: str
    claimed_lb: float | None
    local_estimate: float | None
    kaggle_lb: float | None
    status: str
    adoption_gate: str
    evidence_path: str
    notes: str

    @property
    def submit_floor_ready(self) -> bool:
        return self.status == "submitted_complete" and self.kaggle_lb is not None and self.kaggle_lb >= 6285.0


class PublicCodeRegistry:
    def __init__(self, sources: list[PublicCodeSource] | None = None, current_best_public_lb: float | None = None) -> None:
        self.sources = sources or []
        self.current_best_public_lb = current_best_public_lb

    @classmethod
    def from_experiment_results(cls, root: str | Path) -> "PublicCodeRegistry":
        root_path = Path(root)
        candidates = [
            "exp002_public_blend_6500_fast",
            "exp008_extra_public_sources_strict",
            "exp067_public_notebook_utilization_audit",
            "exp_b022_public_notebook_intelligence",
            "exp_b034_public6200_notebook_local_estimate",
            "exp_b035_new_source_full_arc_blend",
            "exp_b036_beicicc_full_arc_blend",
            "exp_b037_beicicc_golf_blend",
            "exp_b038_comprehensive_blend",
            "exp_b039_best_available_blend",
            "exp130_public_code_6285_floor",
        ]
        sources: list[PublicCodeSource] = []
        for exp_name in candidates:
            exp_dir = root_path / "experiments" / exp_name
            result_path = exp_dir / "result.json"
            notes_path = exp_dir / "notes.md"
            data = _load_json(result_path)
            notes = _read_short(notes_path)
            status = str(data.get("status") or data.get("result") or "intelligence_only")
            local_estimate = _float_or_none(
                data.get("local_estimate")
                or data.get("estimated_local")
                or data.get("final_local")
                or data.get("score")
            )
            adoption = data.get("adoption") if isinstance(data.get("adoption"), dict) else {}
            kaggle_lb = _float_or_none(
                data.get("kaggle_lb")
                or data.get("lb")
                or data.get("public_lb")
                or adoption.get("kaggle_public_score")
            )
            claimed_lb = _float_or_none(data.get("claimed_lb") or data.get("claimed_public_lb"))
            gate = _gate_for(status=status, kaggle_lb=kaggle_lb)
            sources.append(
                PublicCodeSource(
                    source_id=exp_name.replace("exp_", "").replace("exp", "exp_"),
                    experiment=exp_name,
                    claimed_lb=claimed_lb,
                    local_estimate=local_estimate,
                    kaggle_lb=kaggle_lb,
                    status=status,
                    adoption_gate=gate,
                    evidence_path=str(result_path if result_path.exists() else notes_path),
                    notes=notes,
                )
            )
        return cls(sources, current_best_public_lb=_current_best_public_lb(root_path))

    def floor_status(self, target_lb: float = 6285.0) -> dict[str, object]:
        ready = [source for source in self.sources if source.submit_floor_ready]
        max_lb = max(
            [source.kaggle_lb or 0.0 for source in self.sources] + [self.current_best_public_lb or 0.0],
            default=0.0,
        )
        return {
            "target_lb_floor": target_lb,
            "submit_floor_ready": bool(ready),
            "ready_source_count": len(ready),
            "current_best_public_lb": self.current_best_public_lb,
            "max_observed_kaggle_lb": max_lb if max_lb else None,
            "policy": "public CODE is a floor only after full-arc validation plus Kaggle LB evidence; otherwise teacher/intelligence",
        }

    def write_csv(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(asdict(self.sources[0]).keys()) if self.sources else [
                "source_id",
                "experiment",
                "claimed_lb",
                "local_estimate",
                "kaggle_lb",
                "status",
                "adoption_gate",
                "evidence_path",
                "notes",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for source in self.sources:
                writer.writerow(asdict(source))


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_short(path: Path, limit: int = 220) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").replace("\n", " ").strip()
    return text[:limit]


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _current_best_public_lb(root_path: Path) -> float | None:
    summary_path = root_path / "EXP_SUMMARY.md"
    if not summary_path.exists():
        return None
    for line in summary_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("| Best Public LB |"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 4:
                return _float_or_none(cells[3])
    return None


def _gate_for(status: str, kaggle_lb: float | None) -> str:
    if kaggle_lb is not None and kaggle_lb >= 6285.0:
        return "submit_floor_ready"
    if status == "submitted_complete":
        return "lb_calibrated_but_below_6285"
    if "blend" in status or "intelligence" in status:
        return "teacher_only_until_full_arc_and_lb_calibration"
    return "needs_full_arc_validation_and_lb_calibration"
