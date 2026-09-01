from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SettlementState:
    target: float
    known_samples: tuple[float, ...]
    total_samples: int = 60

    def __post_init__(self) -> None:
        if self.total_samples <= 0:
            raise ValueError("total_samples must be positive")
        if len(self.known_samples) > self.total_samples:
            raise ValueError("known sample count cannot exceed total_samples")
        if self.target <= 0:
            raise ValueError("target must be positive")

    @property
    def k(self) -> int:
        return len(self.known_samples)

    @property
    def remaining(self) -> int:
        return self.total_samples - self.k

    @property
    def known_sum(self) -> float:
        return math.fsum(self.known_samples)

    @property
    def required_future_average(self) -> float | None:
        if self.remaining == 0:
            return None
        return (self.total_samples * self.target - self.known_sum) / self.remaining

    @property
    def final_average_if_complete(self) -> float | None:
        if self.remaining:
            return None
        return self.known_sum / self.total_samples


def simple_average(samples: Sequence[float]) -> float:
    if not samples:
        raise ValueError("samples cannot be empty")
    return math.fsum(samples) / len(samples)


def resolves_yes(samples: Sequence[float], target: float, rounding_decimals: int = 2) -> bool:
    return round(simple_average(samples), rounding_decimals) >= round(target, rounding_decimals)


def bm_probability_final_average_above_target(
    state: SettlementState,
    current_index: float,
    sigma_per_sqrt_second: float,
    drift_per_second: float = 0.0,
) -> float:
    n = state.remaining
    if n == 0:
        return float((state.final_average_if_complete or 0.0) >= state.target)
    if current_index <= 0:
        raise ValueError("current_index must be positive")
    if sigma_per_sqrt_second < 0:
        raise ValueError("sigma_per_sqrt_second cannot be negative")

    j_sum = n * (n + 1) / 2
    mean_future_sum = n * current_index + drift_per_second * j_sum
    threshold_future_sum = state.total_samples * state.target - state.known_sum
    if sigma_per_sqrt_second == 0:
        return float(mean_future_sum >= threshold_future_sum)

    covariance_sum = n * (n + 1) * (2 * n + 1) / 6
    std_future_sum = sigma_per_sqrt_second * math.sqrt(covariance_sum)
    z = (mean_future_sum - threshold_future_sum) / std_future_sum
    return NormalDist().cdf(z)
