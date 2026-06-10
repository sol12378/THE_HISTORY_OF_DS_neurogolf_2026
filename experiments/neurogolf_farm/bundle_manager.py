"""Bundle ledger for accepted NeuroGolf candidates.

This module tracks what may enter a submission bundle. It deliberately keeps
submission zip creation outside the core farm so raw artifacts are not copied
or committed by accident.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class BundleCandidate:
    task_id: str
    source_exp: str
    candidate_path: str
    base_cost: float
    candidate_cost: float
    validation_status: str
    local_delta: float
    risk: str
    adoption_status: str

    @property
    def accepted(self) -> bool:
        return (
            self.adoption_status == "accepted"
            and self.validation_status.endswith("_pass_0_fail")
            and self.candidate_cost < self.base_cost
            and self.risk in {"low", "medium"}
        )


class BundleLedger:
    def __init__(self, candidates: list[BundleCandidate] | None = None) -> None:
        self.candidates = candidates or []

    def accepted_candidates(self) -> list[BundleCandidate]:
        return [candidate for candidate in self.candidates if candidate.accepted]

    def total_local_delta(self) -> float:
        return sum(candidate.local_delta for candidate in self.accepted_candidates())

    def write_csv(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "task_id",
            "source_exp",
            "candidate_path",
            "base_cost",
            "candidate_cost",
            "validation_status",
            "local_delta",
            "risk",
            "adoption_status",
        ]
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for candidate in self.candidates:
                writer.writerow(asdict(candidate))
