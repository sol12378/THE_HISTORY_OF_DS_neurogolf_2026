"""Bundle ledger for accepted NeuroGolf candidates.

This module tracks what may enter a submission bundle. It deliberately keeps
submission zip creation outside the core farm so raw artifacts are not copied
or committed by accident.
"""

from __future__ import annotations

import csv
import math
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
    primitive_kind: str = ""

    @property
    def accepted(self) -> bool:
        accuracy_ok = self.validation_status == "full_arc_pass" or self.validation_status.endswith("_pass_0_fail")
        return (
            self.adoption_status == "accepted"
            and accuracy_ok
            and self.candidate_cost < self.base_cost
            and self.risk in {"low", "medium"}
        )

    @property
    def computed_local_delta(self) -> float:
        if self.base_cost <= 0 or self.candidate_cost <= 0 or self.candidate_cost >= self.base_cost:
            return 0.0
        return math.log(self.base_cost / self.candidate_cost)


class BundleLedger:
    def __init__(self, candidates: list[BundleCandidate] | None = None) -> None:
        self.candidates = candidates or []

    @classmethod
    def from_selected_manifest(cls, path: str | Path, *, source_exp: str | None = None) -> "BundleLedger":
        manifest_path = Path(path)
        resolved_source = source_exp or manifest_path.parent.name
        candidates: list[BundleCandidate] = []
        with manifest_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                validation_status = row.get("validation_status", "")
                if validation_status == "full_arc_pass":
                    validation_status = "266_pass_0_fail"
                status = row.get("status", "")
                candidates.append(
                    BundleCandidate(
                        task_id=str(row.get("task_id", "")),
                        source_exp=resolved_source,
                        candidate_path=row.get("candidate_path") or row.get("sha256") or row.get("template_name", ""),
                        base_cost=_float(row.get("baseline_cost") or row.get("base_cost")),
                        candidate_cost=_float(row.get("candidate_cost")),
                        validation_status=validation_status,
                        local_delta=_float(row.get("local_delta")),
                        risk=row.get("risk") or "low",
                        adoption_status="accepted" if status in {"improved", "accepted"} else status,
                        primitive_kind=row.get("primitive_kind", ""),
                    )
                )
        return cls(candidates)

    @classmethod
    def from_selected_manifests(
        cls,
        paths: list[str | Path],
        *,
        exclude_sources: set[str] | None = None,
        exclude_tasks: set[str] | None = None,
    ) -> "BundleLedger":
        excluded = exclude_sources or set()
        excluded_tasks = {str(task_id) for task_id in (exclude_tasks or set())}
        candidates: list[BundleCandidate] = []
        for path in paths:
            ledger = cls.from_selected_manifest(path)
            candidates.extend(
                candidate
                for candidate in ledger.candidates
                if candidate.source_exp not in excluded and candidate.task_id not in excluded_tasks
            )
        return cls(candidates)

    def accepted_candidates(self) -> list[BundleCandidate]:
        return sorted(
            (candidate for candidate in self.candidates if candidate.accepted),
            key=lambda candidate: (-candidate.computed_local_delta, candidate.candidate_cost, candidate.task_id),
        )

    def best_candidates_by_task(self) -> list[BundleCandidate]:
        best: dict[str, BundleCandidate] = {}
        for candidate in self.accepted_candidates():
            if candidate.task_id not in best:
                best[candidate.task_id] = candidate
        return list(best.values())

    def total_local_delta(self) -> float:
        return sum(candidate.computed_local_delta for candidate in self.accepted_candidates())

    def best_total_local_delta(self) -> float:
        return sum(candidate.computed_local_delta for candidate in self.best_candidates_by_task())

    def submission_decision(self, submitted_best_estimate: float) -> dict[str, object]:
        accepted = self.accepted_candidates()
        best_by_task = self.best_candidates_by_task()
        best_delta = self.best_total_local_delta()
        candidate_estimate = submitted_best_estimate + best_delta
        accepted_count_by_primitive: dict[str, int] = {}
        for candidate in accepted:
            primitive_kind = candidate.primitive_kind or "unknown"
            accepted_count_by_primitive[primitive_kind] = accepted_count_by_primitive.get(primitive_kind, 0) + 1
        return {
            "submitted_best_estimate": submitted_best_estimate,
            "accepted_count": len(accepted),
            "best_by_task_count": len(best_by_task),
            "accepted_count_by_primitive": accepted_count_by_primitive,
            "best_total_local_delta": best_delta,
            "candidate_estimate": candidate_estimate,
            "should_submit": candidate_estimate > submitted_best_estimate,
        }

    @classmethod
    def fresh_submission_decision(
        cls,
        paths: list[str | Path],
        *,
        submitted_best_estimate: float,
        exclude_sources: set[str],
        exclude_tasks: set[str] | None = None,
    ) -> dict[str, object]:
        ledger = cls.from_selected_manifests(paths, exclude_sources=exclude_sources, exclude_tasks=exclude_tasks)
        decision = ledger.submission_decision(submitted_best_estimate)
        decision["excluded_sources"] = sorted(exclude_sources)
        decision["excluded_tasks"] = sorted(str(task_id) for task_id in (exclude_tasks or set()))
        decision["input_manifest_count"] = len(paths)
        return decision

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
            "computed_local_delta",
            "risk",
            "adoption_status",
            "primitive_kind",
        ]
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for candidate in self.candidates:
                row = asdict(candidate)
                row["computed_local_delta"] = candidate.computed_local_delta
                writer.writerow(row)


def _float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0
