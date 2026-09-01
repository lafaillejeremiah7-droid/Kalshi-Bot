from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Sequence


EPS = 1e-12


def diffusion_probability_yes(spot: float, target: float, sigma_per_sqrt_second: float, seconds: float) -> float:
    """Zero-drift log-diffusion baseline P(S_T >= target)."""
    if spot <= 0 or target <= 0:
        raise ValueError("spot and target must be positive")
    if sigma_per_sqrt_second < 0 or seconds < 0:
        raise ValueError("sigma and seconds cannot be negative")
    if seconds == 0 or sigma_per_sqrt_second <= EPS:
        return float(spot >= target)
    z = math.log(spot / target) / (sigma_per_sqrt_second * math.sqrt(seconds))
    return NormalDist().cdf(z)


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    _validate_probs_outcomes(probabilities, outcomes)
    return math.fsum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(outcomes)


def log_loss(probabilities: Sequence[float], outcomes: Sequence[int], eps: float = 1e-12) -> float:
    _validate_probs_outcomes(probabilities, outcomes)
    losses = []
    for p, y in zip(probabilities, outcomes):
        q = min(max(p, eps), 1 - eps)
        losses.append(-(y * math.log(q) + (1 - y) * math.log(1 - q)))
    return math.fsum(losses) / len(losses)


def expected_calibration_error(probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> float:
    _validate_probs_outcomes(probabilities, outcomes)
    if bins <= 0:
        raise ValueError("bins must be positive")
    bucketed: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for p, y in zip(probabilities, outcomes):
        idx = min(int(p * bins), bins - 1)
        bucketed[idx].append((p, y))
    n = len(probabilities)
    ece = 0.0
    for bucket in bucketed:
        if not bucket:
            continue
        mean_p = math.fsum(p for p, _ in bucket) / len(bucket)
        mean_y = math.fsum(y for _, y in bucket) / len(bucket)
        ece += len(bucket) / n * abs(mean_p - mean_y)
    return ece


def _validate_probs_outcomes(probabilities: Sequence[float], outcomes: Sequence[int]) -> None:
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must be non-empty and equal length")
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("probabilities must be in [0, 1]")
    if any(y not in (0, 1) for y in outcomes):
        raise ValueError("outcomes must be 0/1")


@dataclass(frozen=True, slots=True)
class CostAdjustedEdge:
    fair_probability: float
    executable_price: float
    fees: float = 0.0
    slippage: float = 0.0
    latency_penalty: float = 0.0

    @property
    def raw_edge(self) -> float:
        return self.fair_probability - self.executable_price

    @property
    def net_edge(self) -> float:
        return self.raw_edge - self.fees - self.slippage - self.latency_penalty

    def passes(self, minimum_edge: float) -> bool:
        return self.net_edge >= minimum_edge


def pair_arbitrage_edge(yes_cost: float, no_cost: float, total_costs: float = 0.0) -> float:
    return 1.0 - yes_cost - no_cost - total_costs


def weighted_ensemble(probabilities: Iterable[tuple[float, float]]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for probability, weight in probabilities:
        if not 0 <= probability <= 1 or weight < 0:
            raise ValueError("invalid probability or weight")
        weighted_sum += probability * weight
        total_weight += weight
    if total_weight <= 0:
        raise ValueError("ensemble requires positive total weight")
    return weighted_sum / total_weight
