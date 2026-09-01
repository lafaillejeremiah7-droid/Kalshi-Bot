from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist

from kalshi_research.math.probability import (
    brier_score,
    diffusion_probability_yes,
    expected_calibration_error,
    log_loss,
)
from kalshi_research.research.materializer import ModelFeatureRow
from kalshi_research.research.walkforward import expanding_walkforward


class ExperimentError(RuntimeError):
    """Raised when an experiment would violate a predeclared research contract."""


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    decision_horizons_s: tuple[int, ...] = (30, 60, 120, 300, 600)
    lead_lags_s: tuple[int, ...] = (0, 1, 2, 3, 5, 10)
    grid_step_s: int = 1
    max_asof_age_s: float = 2.5
    min_leadlag_pairs: int = 300
    bootstrap_samples: int = 1000
    bootstrap_block_size: int = 10
    random_seed: int = 17
    min_train_markets: int = 100
    validation_markets: int = 20
    test_markets: int = 20
    step_markets: int = 20

    def __post_init__(self) -> None:
        if not self.decision_horizons_s or not self.lead_lags_s:
            raise ValueError("horizons and lead lags cannot be empty")
        if tuple(sorted(set(self.decision_horizons_s))) != self.decision_horizons_s:
            raise ValueError("decision_horizons_s must be unique and strictly increasing")
        if tuple(sorted(set(self.lead_lags_s))) != self.lead_lags_s:
            raise ValueError("lead_lags_s must be unique and strictly increasing")
        if any(value <= 0 for value in self.decision_horizons_s):
            raise ValueError("decision horizons must be positive")
        if any(value < 0 for value in self.lead_lags_s):
            raise ValueError("lead lags cannot be negative")
        if self.grid_step_s <= 0 or self.max_asof_age_s <= 0:
            raise ValueError("grid_step_s and max_asof_age_s must be positive")
        if self.min_leadlag_pairs < 3:
            raise ValueError("min_leadlag_pairs must be at least 3")
        if self.bootstrap_samples <= 0 or self.bootstrap_block_size <= 0:
            raise ValueError("bootstrap settings must be positive")
        if min(
            self.min_train_markets,
            self.validation_markets,
            self.test_markets,
            self.step_markets,
        ) <= 0:
            raise ValueError("walk-forward market counts must be positive")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ProbabilityScore:
    count: int
    brier: float
    log_loss: float
    calibration_error: float


@dataclass(frozen=True, slots=True)
class HorizonProbabilityBenchmark:
    horizon_s: int
    market_ids: tuple[str, ...]
    candidate: ProbabilityScore
    market_implied: ProbabilityScore

    @property
    def brier_improvement(self) -> float:
        return self.market_implied.brier - self.candidate.brier

    @property
    def log_loss_improvement(self) -> float:
        return self.market_implied.log_loss - self.candidate.log_loss


@dataclass(frozen=True, slots=True)
class FoldProbabilityBenchmark:
    fold_index: int
    horizon_s: int
    train_market_ids: tuple[str, ...]
    validation_market_ids: tuple[str, ...]
    test_market_ids: tuple[str, ...]
    benchmark: HorizonProbabilityBenchmark


@dataclass(frozen=True, slots=True)
class TimedPrice:
    recv_ts_ns: int
    price: float

    def __post_init__(self) -> None:
        if self.recv_ts_ns < 0:
            raise ValueError("recv_ts_ns cannot be negative")
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError("price must be finite and positive")


@dataclass(frozen=True, slots=True)
class LeadLagResult:
    lag_s: int
    pairs: int
    correlation: float | None
    raw_p_value: float | None
    bonferroni_p_value: float | None
    bootstrap_ci_low: float | None
    bootstrap_ci_high: float | None
    eligible: bool


@dataclass(frozen=True, slots=True)
class LeadLagReport:
    plan_digest: str
    grid_points: int
    results: tuple[LeadLagResult, ...]

    @property
    def best_eligible(self) -> LeadLagResult | None:
        candidates = [
            result
            for result in self.results
            if result.eligible and result.correlation is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda result: abs(result.correlation or 0.0))


def select_horizon_rows(
    rows: Iterable[ModelFeatureRow],
    horizon_s: int,
    *,
    allowed_market_ids: set[str] | None = None,
) -> dict[str, ModelFeatureRow]:
    """Select one row per market immediately before a predeclared horizon.

    A row with fewer seconds remaining than the requested horizon is later
    information and is never eligible. Among eligible rows, the one with the
    smallest seconds-to-close is closest to the horizon from the safe side.
    """
    if horizon_s <= 0:
        raise ValueError("horizon_s must be positive")

    selected: dict[str, ModelFeatureRow] = {}
    for row in rows:
        if allowed_market_ids is not None and row.market_ticker not in allowed_market_ids:
            continue
        if not row.baseline_ready or row.seconds_to_close is None:
            continue
        if row.seconds_to_close < horizon_s:
            continue

        previous = selected.get(row.market_ticker)
        if previous is None:
            selected[row.market_ticker] = row
            continue

        previous_seconds = previous.seconds_to_close
        if previous_seconds is None:
            raise ExperimentError("baseline_ready row unexpectedly lacks seconds_to_close")
        if row.seconds_to_close < previous_seconds or (
            row.seconds_to_close == previous_seconds
            and row.decision_recv_ts_ns > previous.decision_recv_ts_ns
        ):
            selected[row.market_ticker] = row
    return selected


def benchmark_probability_horizon(
    rows: Iterable[ModelFeatureRow],
    outcomes_by_market: Mapping[str, int],
    horizon_s: int,
    *,
    allowed_market_ids: set[str] | None = None,
) -> HorizonProbabilityBenchmark:
    selected = select_horizon_rows(
        rows,
        horizon_s,
        allowed_market_ids=allowed_market_ids,
    )
    market_ids: list[str] = []
    candidate_probs: list[float] = []
    market_probs: list[float] = []
    outcomes: list[int] = []

    for market_id in sorted(selected):
        if market_id not in outcomes_by_market:
            continue
        outcome = outcomes_by_market[market_id]
        if outcome not in (0, 1):
            raise ValueError(f"outcome for {market_id} must be 0 or 1")

        row = selected[market_id]
        required = (
            row.brti,
            row.target_price,
            row.brti_vol_per_sqrt_second,
            row.seconds_to_close,
            row.kalshi_yes_mid,
        )
        if any(value is None for value in required):
            raise ExperimentError(
                f"baseline_ready row for {market_id} is missing required probability fields"
            )

        candidate = diffusion_probability_yes(
            float(row.brti),
            float(row.target_price),
            float(row.brti_vol_per_sqrt_second),
            float(row.seconds_to_close),
        )
        market_probability = float(row.kalshi_yes_mid)
        if not 0 <= market_probability <= 1:
            raise ExperimentError(f"Kalshi implied probability out of range for {market_id}")

        market_ids.append(market_id)
        candidate_probs.append(candidate)
        market_probs.append(market_probability)
        outcomes.append(outcome)

    if not outcomes:
        raise ExperimentError(f"no evaluable markets at horizon {horizon_s}s")

    return HorizonProbabilityBenchmark(
        horizon_s=horizon_s,
        market_ids=tuple(market_ids),
        candidate=_probability_score(candidate_probs, outcomes),
        market_implied=_probability_score(market_probs, outcomes),
    )


def benchmark_probability_walkforward(
    rows: Sequence[ModelFeatureRow],
    outcomes_by_market: Mapping[str, int],
    ordered_market_ids: Sequence[str],
    plan: ExperimentPlan,
) -> tuple[FoldProbabilityBenchmark, ...]:
    folds = expanding_walkforward(
        ordered_market_ids,
        min_train=plan.min_train_markets,
        validation_size=plan.validation_markets,
        test_size=plan.test_markets,
        step=plan.step_markets,
    )
    results: list[FoldProbabilityBenchmark] = []
    for fold_index, fold in enumerate(folds):
        test_ids = set(fold.test)
        for horizon_s in plan.decision_horizons_s:
            try:
                benchmark = benchmark_probability_horizon(
                    rows,
                    outcomes_by_market,
                    horizon_s,
                    allowed_market_ids=test_ids,
                )
            except ExperimentError as exc:
                if str(exc).startswith("no evaluable markets"):
                    continue
                raise
            results.append(
                FoldProbabilityBenchmark(
                    fold_index=fold_index,
                    horizon_s=horizon_s,
                    train_market_ids=fold.train,
                    validation_market_ids=fold.validation,
                    test_market_ids=fold.test,
                    benchmark=benchmark,
                )
            )
    return tuple(results)


def scan_receive_time_lead_lag(
    leader: Sequence[TimedPrice],
    follower: Sequence[TimedPrice],
    plan: ExperimentPlan,
) -> LeadLagReport:
    """Test whether leader returns precede follower returns using receive time only.

    Positive lag means a leader return ending at time t is paired with a
    follower return ending at t + lag. Fisher p-values are approximate because
    high-frequency returns can be serially dependent; block-bootstrap intervals
    are reported alongside them.
    """
    _validate_timed_prices(leader, "leader")
    _validate_timed_prices(follower, "follower")

    step_ns = plan.grid_step_s * 1_000_000_000
    max_age_ns = int(plan.max_asof_age_s * 1_000_000_000)
    grid = _asof_grid(leader, follower, step_ns=step_ns, max_age_ns=max_age_ns)
    leader_returns, follower_returns = _grid_log_returns(grid, step_ns)

    tests = len(plan.lead_lags_s)
    results: list[LeadLagResult] = []
    for lag_s in plan.lead_lags_s:
        lag_ns = lag_s * 1_000_000_000
        pairs = [
            (leader_return, follower_returns[timestamp + lag_ns])
            for timestamp, leader_return in sorted(leader_returns.items())
            if timestamp + lag_ns in follower_returns
        ]
        eligible = len(pairs) >= plan.min_leadlag_pairs
        if len(pairs) < 3:
            results.append(
                LeadLagResult(
                    lag_s=lag_s,
                    pairs=len(pairs),
                    correlation=None,
                    raw_p_value=None,
                    bonferroni_p_value=None,
                    bootstrap_ci_low=None,
                    bootstrap_ci_high=None,
                    eligible=False,
                )
            )
            continue

        correlation = _pearson_pairs(pairs)
        raw_p = _fisher_two_sided_p(correlation, len(pairs))
        adjusted_p = min(1.0, raw_p * tests)
        ci_low, ci_high = _moving_block_bootstrap_ci(
            pairs,
            samples=plan.bootstrap_samples,
            block_size=plan.bootstrap_block_size,
            seed=plan.random_seed + lag_s * 1009,
        )
        results.append(
            LeadLagResult(
                lag_s=lag_s,
                pairs=len(pairs),
                correlation=correlation,
                raw_p_value=raw_p,
                bonferroni_p_value=adjusted_p,
                bootstrap_ci_low=ci_low,
                bootstrap_ci_high=ci_high,
                eligible=eligible,
            )
        )

    return LeadLagReport(
        plan_digest=plan.digest,
        grid_points=len(grid),
        results=tuple(results),
    )


def _probability_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> ProbabilityScore:
    return ProbabilityScore(
        count=len(outcomes),
        brier=brier_score(probabilities, outcomes),
        log_loss=log_loss(probabilities, outcomes),
        calibration_error=expected_calibration_error(probabilities, outcomes),
    )


def _validate_timed_prices(points: Sequence[TimedPrice], name: str) -> None:
    if len(points) < 2:
        raise ValueError(f"{name} requires at least two observations")
    previous: int | None = None
    for point in points:
        if previous is not None and point.recv_ts_ns <= previous:
            raise ValueError(f"{name} receive timestamps must be strictly increasing")
        previous = point.recv_ts_ns


def _asof_grid(
    leader: Sequence[TimedPrice],
    follower: Sequence[TimedPrice],
    *,
    step_ns: int,
    max_age_ns: int,
) -> dict[int, tuple[float, float]]:
    start = max(leader[0].recv_ts_ns, follower[0].recv_ts_ns)
    end = min(leader[-1].recv_ts_ns, follower[-1].recv_ts_ns)
    first_grid = ((start + step_ns - 1) // step_ns) * step_ns
    last_grid = (end // step_ns) * step_ns
    if first_grid > last_grid:
        return {}

    leader_index = follower_index = 0
    latest_leader: TimedPrice | None = None
    latest_follower: TimedPrice | None = None
    grid: dict[int, tuple[float, float]] = {}

    for timestamp in range(first_grid, last_grid + 1, step_ns):
        while (
            leader_index < len(leader)
            and leader[leader_index].recv_ts_ns <= timestamp
        ):
            latest_leader = leader[leader_index]
            leader_index += 1
        while (
            follower_index < len(follower)
            and follower[follower_index].recv_ts_ns <= timestamp
        ):
            latest_follower = follower[follower_index]
            follower_index += 1

        if latest_leader is None or latest_follower is None:
            continue
        if timestamp - latest_leader.recv_ts_ns > max_age_ns:
            continue
        if timestamp - latest_follower.recv_ts_ns > max_age_ns:
            continue
        grid[timestamp] = (latest_leader.price, latest_follower.price)
    return grid


def _grid_log_returns(
    grid: Mapping[int, tuple[float, float]],
    step_ns: int,
) -> tuple[dict[int, float], dict[int, float]]:
    leader_returns: dict[int, float] = {}
    follower_returns: dict[int, float] = {}
    for timestamp in sorted(grid):
        previous_timestamp = timestamp - step_ns
        if previous_timestamp not in grid:
            continue
        previous_leader, previous_follower = grid[previous_timestamp]
        current_leader, current_follower = grid[timestamp]
        leader_returns[timestamp] = math.log(current_leader / previous_leader)
        follower_returns[timestamp] = math.log(current_follower / previous_follower)
    return leader_returns, follower_returns


def _pearson_pairs(pairs: Sequence[tuple[float, float]]) -> float:
    x_mean = math.fsum(x for x, _ in pairs) / len(pairs)
    y_mean = math.fsum(y for _, y in pairs) / len(pairs)
    x_centered = [x - x_mean for x, _ in pairs]
    y_centered = [y - y_mean for _, y in pairs]
    denominator = math.sqrt(
        math.fsum(value * value for value in x_centered)
        * math.fsum(value * value for value in y_centered)
    )
    if denominator <= 1e-18:
        return 0.0
    return math.fsum(
        x_value * y_value
        for x_value, y_value in zip(x_centered, y_centered)
    ) / denominator


def _fisher_two_sided_p(correlation: float, pairs: int) -> float:
    if pairs <= 3:
        return 1.0
    clipped = min(max(correlation, -1 + 1e-12), 1 - 1e-12)
    z = math.atanh(clipped) * math.sqrt(pairs - 3)
    return 2 * (1 - NormalDist().cdf(abs(z)))


def _moving_block_bootstrap_ci(
    pairs: Sequence[tuple[float, float]],
    *,
    samples: int,
    block_size: int,
    seed: int,
) -> tuple[float, float]:
    n = len(pairs)
    block = min(block_size, n)
    rng = random.Random(seed)
    correlations: list[float] = []
    max_start = n - block

    for _ in range(samples):
        resampled: list[tuple[float, float]] = []
        while len(resampled) < n:
            start = rng.randint(0, max_start) if max_start > 0 else 0
            resampled.extend(pairs[start : start + block])
        correlations.append(_pearson_pairs(resampled[:n]))

    correlations.sort()
    low_index = max(0, math.floor(0.025 * (samples - 1)))
    high_index = min(samples - 1, math.ceil(0.975 * (samples - 1)))
    return correlations[low_index], correlations[high_index]
