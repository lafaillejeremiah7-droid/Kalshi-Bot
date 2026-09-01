from dataclasses import replace

from kalshi_research.research.coverage import CoveragePolicy, audit_feature_coverage
from kalshi_research.research.materializer import ModelFeatureRow


def row(
    *,
    probability_ready: bool = True,
    baseline_ready: bool = True,
    coinbase_mid: float | None = 100.0,
    kraken_mid: float | None = 100.1,
    normalized_distance: float | None = 0.2,
    age_ms: float = 10.0,
) -> ModelFeatureRow:
    return ModelFeatureRow(
        decision_recv_ts_ns=1,
        market_ticker="KXBTC15M-TEST",
        probability_ready=probability_ready,
        baseline_ready=baseline_ready,
        seconds_to_close=10.0,
        target_price=100.0,
        brti=100.2,
        brti_log_distance_to_target=0.001,
        brti_vol_per_sqrt_second=0.01,
        normalized_distance_to_target=normalized_distance,
        kalshi_yes_bid=0.48,
        kalshi_yes_ask=0.50,
        kalshi_yes_mid=0.49,
        kalshi_spread=0.02,
        kalshi_book_imbalance=0.1,
        external_consensus_mid=100.05,
        brti_vs_external_bps=1.0,
        coinbase_mid=coinbase_mid,
        kraken_mid=kraken_mid,
        final_minute_sample_count=20,
        final_minute_progress=1 / 3,
        final_minute_average=100.1,
        required_remaining_brti_average=99.95,
        kalshi_book_age_ms=age_ms,
        brti_age_ms=age_ms,
        coinbase_age_ms=age_ms if coinbase_mid is not None else None,
        kraken_age_ms=age_ms if kraken_mid is not None else None,
    )


def test_healthy_coverage_passes():
    rows = [row() for _ in range(90)] + [
        row(probability_ready=False, baseline_ready=False) for _ in range(10)
    ]
    report = audit_feature_coverage(rows)

    assert report.passed
    assert report.probability_ready_fraction == 0.9
    assert report.baseline_ready_fraction_of_probability == 1.0
    assert report.dual_external_present_fraction == 1.0


def test_low_probability_ready_coverage_fails():
    rows = [row() for _ in range(70)] + [
        row(probability_ready=False, baseline_ready=False) for _ in range(30)
    ]
    report = audit_feature_coverage(rows)

    assert not report.passed
    assert any(
        failure.startswith("probability_ready_coverage_below_minimum")
        for failure in report.failures
    )


def test_baseline_coverage_is_measured_only_within_probability_ready_rows():
    rows = [row(baseline_ready=True) for _ in range(40)]
    rows += [row(baseline_ready=False) for _ in range(50)]
    rows += [row(probability_ready=False, baseline_ready=False) for _ in range(10)]
    report = audit_feature_coverage(rows)

    assert report.probability_ready_fraction == 0.9
    assert report.baseline_ready_fraction_of_probability == 40 / 90
    assert not report.passed
    assert any(
        failure.startswith("baseline_ready_coverage_below_minimum")
        for failure in report.failures
    )


def test_nonfinite_feature_fails_closed():
    rows = [row() for _ in range(100)]
    rows[-1] = replace(rows[-1], normalized_distance_to_target=float("inf"))
    report = audit_feature_coverage(rows)

    assert not report.passed
    assert report.nonfinite_numeric_values == 1
    assert "nonfinite_numeric_values:1" in report.failures


def test_minimum_sample_size_is_enforced():
    report = audit_feature_coverage(
        [row() for _ in range(10)],
        policy=CoveragePolicy(min_rows=20),
    )

    assert not report.passed
    assert "row_count_below_minimum:10<20" in report.failures


def test_p95_source_age_is_reported():
    rows = [row(age_ms=float(i)) for i in range(1, 101)]
    report = audit_feature_coverage(rows)

    assert report.kalshi_book_age_p95_ms == 95.0
    assert report.brti_age_p95_ms == 95.0
    assert report.coinbase_age_p95_ms == 95.0
    assert report.kraken_age_p95_ms == 95.0
