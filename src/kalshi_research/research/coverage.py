from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, fields

from kalshi_research.research.materializer import ModelFeatureRow


@dataclass(frozen=True, slots=True)
class CoveragePolicy:
    min_rows: int = 100
    min_probability_ready_fraction: float = 0.80
    min_baseline_ready_fraction_of_probability: float = 0.50

    def __post_init__(self) -> None:
        if self.min_rows < 1:
            raise ValueError("min_rows must be at least 1")
        for value in (
            self.min_probability_ready_fraction,
            self.min_baseline_ready_fraction_of_probability,
        ):
            if not 0 <= value <= 1:
                raise ValueError("coverage fractions must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class FeatureCoverageReport:
    total_rows: int
    probability_ready_rows: int
    baseline_ready_rows: int
    probability_ready_fraction: float
    baseline_ready_fraction_of_probability: float
    coinbase_present_fraction: float
    kraken_present_fraction: float
    dual_external_present_fraction: float
    nonfinite_numeric_values: int
    kalshi_book_age_p95_ms: float | None
    brti_age_p95_ms: float | None
    coinbase_age_p95_ms: float | None
    kraken_age_p95_ms: float | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def audit_feature_coverage(
    rows: Iterable[ModelFeatureRow],
    *,
    policy: CoveragePolicy | None = None,
) -> FeatureCoverageReport:
    """Measure whether structurally valid feature data is actually usable."""
    policy = policy or CoveragePolicy()
    materialized = list(rows)
    total = len(materialized)

    probability_ready = sum(row.probability_ready for row in materialized)
    baseline_ready = sum(row.baseline_ready for row in materialized)
    probability_fraction = probability_ready / total if total else 0.0
    baseline_fraction = baseline_ready / probability_ready if probability_ready else 0.0

    coinbase_present = sum(row.coinbase_mid is not None for row in materialized)
    kraken_present = sum(row.kraken_mid is not None for row in materialized)
    dual_present = sum(
        row.coinbase_mid is not None and row.kraken_mid is not None
        for row in materialized
    )

    nonfinite = _count_nonfinite_numeric_values(materialized)
    failures: list[str] = []
    if total < policy.min_rows:
        failures.append(f"row_count_below_minimum:{total}<{policy.min_rows}")
    if probability_fraction < policy.min_probability_ready_fraction:
        failures.append(
            "probability_ready_coverage_below_minimum:"
            f"{probability_fraction:.6f}<{policy.min_probability_ready_fraction:.6f}"
        )
    if baseline_fraction < policy.min_baseline_ready_fraction_of_probability:
        failures.append(
            "baseline_ready_coverage_below_minimum:"
            f"{baseline_fraction:.6f}<"
            f"{policy.min_baseline_ready_fraction_of_probability:.6f}"
        )
    if nonfinite:
        failures.append(f"nonfinite_numeric_values:{nonfinite}")

    return FeatureCoverageReport(
        total_rows=total,
        probability_ready_rows=probability_ready,
        baseline_ready_rows=baseline_ready,
        probability_ready_fraction=probability_fraction,
        baseline_ready_fraction_of_probability=baseline_fraction,
        coinbase_present_fraction=coinbase_present / total if total else 0.0,
        kraken_present_fraction=kraken_present / total if total else 0.0,
        dual_external_present_fraction=dual_present / total if total else 0.0,
        nonfinite_numeric_values=nonfinite,
        kalshi_book_age_p95_ms=_p95(
            row.kalshi_book_age_ms for row in materialized if row.kalshi_book_age_ms is not None
        ),
        brti_age_p95_ms=_p95(
            row.brti_age_ms for row in materialized if row.brti_age_ms is not None
        ),
        coinbase_age_p95_ms=_p95(
            row.coinbase_age_ms for row in materialized if row.coinbase_age_ms is not None
        ),
        kraken_age_p95_ms=_p95(
            row.kraken_age_ms for row in materialized if row.kraken_age_ms is not None
        ),
        failures=tuple(failures),
    )


def _count_nonfinite_numeric_values(rows: list[ModelFeatureRow]) -> int:
    count = 0
    for row in rows:
        for field_info in fields(row):
            value = getattr(row, field_info.name)
            if isinstance(value, float) and not math.isfinite(value):
                count += 1
    return count


def _p95(values: Iterable[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]
