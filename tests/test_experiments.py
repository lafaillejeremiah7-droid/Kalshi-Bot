import math

import pytest

from kalshi_research.research.experiments import (
    ExperimentPlan,
    TimedPrice,
    _asof_grid,
    benchmark_probability_horizon,
    benchmark_probability_walkforward,
    scan_receive_time_lead_lag,
    select_horizon_rows,
)
from kalshi_research.research.materializer import ModelFeatureRow


NS = 1_000_000_000


def feature_row(
    market: str,
    seconds: float,
    decision: int,
    *,
    brti: float = 101.0,
    target: float = 100.0,
    sigma: float = 0.01,
    kalshi_mid: float = 0.55,
    baseline_ready: bool = True,
) -> ModelFeatureRow:
    return ModelFeatureRow(
        decision_recv_ts_ns=decision,
        market_ticker=market,
        probability_ready=baseline_ready,
        baseline_ready=baseline_ready,
        seconds_to_close=seconds,
        target_price=target,
        brti=brti,
        brti_log_distance_to_target=math.log(brti / target),
        brti_vol_per_sqrt_second=sigma,
        normalized_distance_to_target=(
            math.log(brti / target) / (sigma * math.sqrt(seconds))
            if baseline_ready and seconds > 0
            else None
        ),
        kalshi_yes_bid=kalshi_mid - 0.01,
        kalshi_yes_ask=kalshi_mid + 0.01,
        kalshi_yes_mid=kalshi_mid,
        kalshi_spread=0.02,
        kalshi_book_imbalance=0.0,
        external_consensus_mid=brti,
        brti_vs_external_bps=0.0,
        coinbase_mid=brti,
        kraken_mid=brti,
        final_minute_sample_count=None,
        final_minute_progress=None,
        final_minute_average=None,
        required_remaining_brti_average=None,
        kalshi_book_age_ms=10.0,
        brti_age_ms=10.0,
        coinbase_age_ms=10.0,
        kraken_age_ms=10.0,
    )


def test_experiment_plan_digest_is_deterministic_and_sensitive_to_plan():
    first = ExperimentPlan(bootstrap_samples=25)
    second = ExperimentPlan(bootstrap_samples=25)
    changed = ExperimentPlan(bootstrap_samples=26)

    assert first.digest == second.digest
    assert first.digest != changed.digest


def test_horizon_selector_uses_nearest_row_from_safe_side_only():
    rows = [
        feature_row("m1", 70, 1),
        feature_row("m1", 65, 2),
        feature_row("m1", 59, 3),
    ]
    selected = select_horizon_rows(rows, 60)

    assert selected["m1"].seconds_to_close == 65
    assert selected["m1"].decision_recv_ts_ns == 2


def test_horizon_selector_returns_one_row_per_market():
    rows = [
        feature_row("m1", 70, 1),
        feature_row("m1", 65, 2),
        feature_row("m2", 61, 3),
        feature_row("m2", 60, 4),
    ]
    selected = select_horizon_rows(rows, 60)

    assert set(selected) == {"m1", "m2"}
    assert selected["m1"].seconds_to_close == 65
    assert selected["m2"].seconds_to_close == 60


def test_probability_benchmark_compares_diffusion_to_same_horizon_market_price():
    rows = [
        feature_row("yes-market", 60, 1, brti=102, kalshi_mid=0.60),
        feature_row("no-market", 60, 2, brti=98, kalshi_mid=0.40),
    ]
    report = benchmark_probability_horizon(
        rows,
        {"yes-market": 1, "no-market": 0},
        60,
    )

    assert report.candidate.count == 2
    assert report.market_implied.count == 2
    assert set(report.market_ids) == {"yes-market", "no-market"}
    assert 0 <= report.candidate.brier <= 1
    assert 0 <= report.market_implied.brier <= 1


def test_probability_benchmark_skips_missing_labels_and_nonready_rows():
    rows = [
        feature_row("usable", 60, 1),
        feature_row("unlabeled", 60, 2),
        feature_row("not-ready", 60, 3, baseline_ready=False),
    ]
    report = benchmark_probability_horizon(rows, {"usable": 1, "not-ready": 0}, 60)

    assert report.market_ids == ("usable",)
    assert report.candidate.count == 1


def test_walkforward_probability_benchmark_evaluates_only_whole_test_markets():
    market_ids = [f"m{i}" for i in range(5)]
    rows = [
        feature_row(market_id, 60, index + 1, brti=101 if index % 2 else 99)
        for index, market_id in enumerate(market_ids)
    ]
    outcomes = {market_id: index % 2 for index, market_id in enumerate(market_ids)}
    plan = ExperimentPlan(
        decision_horizons_s=(60,),
        lead_lags_s=(0,),
        min_leadlag_pairs=3,
        bootstrap_samples=10,
        min_train_markets=2,
        validation_markets=1,
        test_markets=1,
        step_markets=1,
    )

    reports = benchmark_probability_walkforward(rows, outcomes, market_ids, plan)

    assert len(reports) == 2
    for report in reports:
        assert set(report.benchmark.market_ids).issubset(set(report.test_market_ids))
        assert not set(report.train_market_ids) & set(report.test_market_ids)
        assert not set(report.validation_market_ids) & set(report.test_market_ids)


def test_asof_grid_never_uses_future_received_sample():
    leader = [
        TimedPrice(0, 100.0),
        TimedPrice(int(1.1 * NS), 200.0),
        TimedPrice(2 * NS, 201.0),
    ]
    follower = [
        TimedPrice(0, 100.0),
        TimedPrice(1 * NS, 100.0),
        TimedPrice(2 * NS, 100.0),
    ]

    grid = _asof_grid(leader, follower, step_ns=NS, max_age_ns=2 * NS)

    assert grid[1 * NS][0] == 100.0
    assert grid[2 * NS][0] == 201.0


def delayed_series(length: int = 40) -> tuple[list[TimedPrice], list[TimedPrice]]:
    returns = [
        0.0010,
        -0.0017,
        0.0004,
        0.0022,
        -0.0008,
        0.0013,
        -0.0021,
        0.0009,
        0.0018,
        -0.0003,
    ]
    leader_prices = [100.0]
    for index in range(length):
        leader_prices.append(
            leader_prices[-1] * math.exp(returns[index % len(returns)] + index * 1e-7)
        )

    leader = [TimedPrice(index * NS, price) for index, price in enumerate(leader_prices)]
    follower_prices = [leader_prices[0]] + leader_prices[:-1]
    follower = [TimedPrice(index * NS, price) for index, price in enumerate(follower_prices)]
    return leader, follower


def small_leadlag_plan() -> ExperimentPlan:
    return ExperimentPlan(
        decision_horizons_s=(60,),
        lead_lags_s=(0, 1, 2),
        grid_step_s=1,
        max_asof_age_s=1.5,
        min_leadlag_pairs=10,
        bootstrap_samples=50,
        bootstrap_block_size=4,
        random_seed=123,
        min_train_markets=2,
        validation_markets=1,
        test_markets=1,
        step_markets=1,
    )


def test_receive_time_leadlag_finds_synthetic_one_second_lead():
    leader, follower = delayed_series()
    report = scan_receive_time_lead_lag(leader, follower, small_leadlag_plan())

    assert report.best_eligible is not None
    assert report.best_eligible.lag_s == 1
    assert report.best_eligible.correlation == pytest.approx(1.0, abs=1e-9)


def test_bonferroni_adjustment_is_never_smaller_than_raw_p_value():
    leader, follower = delayed_series()
    report = scan_receive_time_lead_lag(leader, follower, small_leadlag_plan())

    for result in report.results:
        if result.raw_p_value is not None:
            assert result.bonferroni_p_value is not None
            assert result.bonferroni_p_value >= result.raw_p_value


def test_block_bootstrap_is_deterministic_for_fixed_plan_seed():
    leader, follower = delayed_series()
    plan = small_leadlag_plan()

    first = scan_receive_time_lead_lag(leader, follower, plan)
    second = scan_receive_time_lead_lag(leader, follower, plan)

    assert first == second


def test_leadlag_marks_insufficient_pair_count_ineligible():
    leader, follower = delayed_series(length=8)
    plan = ExperimentPlan(
        decision_horizons_s=(60,),
        lead_lags_s=(0, 1),
        min_leadlag_pairs=20,
        bootstrap_samples=10,
        bootstrap_block_size=2,
        min_train_markets=2,
        validation_markets=1,
        test_markets=1,
        step_markets=1,
    )
    report = scan_receive_time_lead_lag(leader, follower, plan)

    assert all(not result.eligible for result in report.results)
