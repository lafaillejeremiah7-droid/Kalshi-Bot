import math
from decimal import Decimal

import pytest

from kalshi_research.research.materializer import FeatureMaterializer
from kalshi_research.research.synchronizer import SourceState, SynchronizedFrame


T0 = 1_800_000_000_000_000_000


def state(recv_ts_ns: int | None, decision_ts_ns: int, fresh: bool = True) -> SourceState:
    if recv_ts_ns is None:
        return SourceState(recv_ts_ns=None, age_ns=None, fresh=False)
    return SourceState(
        recv_ts_ns=recv_ts_ns,
        age_ns=decision_ts_ns - recv_ts_ns,
        fresh=fresh,
    )


def frame(
    *,
    decision_ts_ns: int,
    brti_value: str,
    brti_recv_ts_ns: int,
    target: str = "100",
    coinbase_mid: str | None = "100",
    coinbase_fresh: bool = True,
    kraken_mid: str | None = None,
    kraken_fresh: bool = False,
    final_average: str | None = None,
    sample_count: int | None = None,
    brti_fresh: bool = True,
    seconds_to_close: float = 10.0,
) -> SynchronizedFrame:
    return SynchronizedFrame(
        decision_recv_ts_ns=decision_ts_ns,
        market_ticker="KXBTC15M-TEST",
        target_price=Decimal(target),
        close_ts_ns=decision_ts_ns + int(seconds_to_close * 1_000_000_000),
        seconds_to_close=seconds_to_close,
        yes_bid=Decimal("0.45"),
        yes_bid_size=Decimal("8"),
        yes_ask=Decimal("0.49"),
        yes_ask_size=Decimal("2"),
        yes_spread=Decimal("0.04"),
        brti=Decimal(brti_value),
        brti_final_minute_average=(
            Decimal(final_average) if final_average is not None else None
        ),
        brti_final_minute_sample_count=sample_count,
        coinbase_mid=Decimal(coinbase_mid) if coinbase_mid is not None else None,
        coinbase_bid_size=Decimal("3") if coinbase_mid is not None else None,
        coinbase_ask_size=Decimal("4") if coinbase_mid is not None else None,
        kraken_mid=Decimal(kraken_mid) if kraken_mid is not None else None,
        kraken_bid_size=Decimal("5") if kraken_mid is not None else None,
        kraken_ask_size=Decimal("6") if kraken_mid is not None else None,
        market_state=state(decision_ts_ns, decision_ts_ns),
        kalshi_book_state=state(decision_ts_ns, decision_ts_ns),
        brti_state=state(brti_recv_ts_ns, decision_ts_ns, brti_fresh),
        coinbase_state=state(
            decision_ts_ns,
            decision_ts_ns,
            coinbase_fresh and coinbase_mid is not None,
        ),
        kraken_state=state(
            decision_ts_ns,
            decision_ts_ns,
            kraken_fresh and kraken_mid is not None,
        ),
    )


def test_first_brti_observation_has_no_volatility_baseline():
    materializer = FeatureMaterializer()
    row = materializer.materialize(
        frame(decision_ts_ns=T0, brti_value="100", brti_recv_ts_ns=T0)
    )

    assert row.brti_vol_per_sqrt_second is None
    assert row.normalized_distance_to_target is None
    assert not row.baseline_ready


def test_event_timed_volatility_and_normalized_distance_are_mathematically_consistent():
    materializer = FeatureMaterializer()
    materializer.materialize(
        frame(decision_ts_ns=T0, brti_value="100", brti_recv_ts_ns=T0)
    )
    row = materializer.materialize(
        frame(
            decision_ts_ns=T0 + 1_000_000_000,
            brti_value="101",
            brti_recv_ts_ns=T0 + 1_000_000_000,
            coinbase_mid="100.5",
        )
    )

    expected_sigma = math.log(101 / 100)
    assert row.brti_vol_per_sqrt_second == pytest.approx(expected_sigma)
    assert row.brti_log_distance_to_target == pytest.approx(expected_sigma)
    assert row.normalized_distance_to_target == pytest.approx(1 / math.sqrt(10))
    assert row.baseline_ready


def test_expiry_does_not_emit_infinite_normalized_distance():
    materializer = FeatureMaterializer()
    materializer.materialize(
        frame(decision_ts_ns=T0, brti_value="100", brti_recv_ts_ns=T0)
    )
    row = materializer.materialize(
        frame(
            decision_ts_ns=T0 + 1_000_000_000,
            brti_value="101",
            brti_recv_ts_ns=T0 + 1_000_000_000,
            seconds_to_close=0.0,
        )
    )

    assert row.brti_vol_per_sqrt_second is not None
    assert row.normalized_distance_to_target is None
    assert not row.baseline_ready


def test_final_minute_required_remaining_average_uses_known_rolling_sum():
    materializer = FeatureMaterializer()
    row = materializer.materialize(
        frame(
            decision_ts_ns=T0,
            brti_value="99",
            brti_recv_ts_ns=T0,
            final_average="99",
            sample_count=30,
        )
    )

    assert row.final_minute_progress == pytest.approx(0.5)
    assert row.required_remaining_brti_average == pytest.approx(101.0)


def test_stale_external_source_is_excluded_from_consensus():
    materializer = FeatureMaterializer()
    row = materializer.materialize(
        frame(
            decision_ts_ns=T0,
            brti_value="100",
            brti_recv_ts_ns=T0,
            coinbase_mid="90",
            coinbase_fresh=False,
            kraken_mid="102",
            kraken_fresh=True,
        )
    )

    assert row.coinbase_mid is None
    assert row.kraken_mid == pytest.approx(102.0)
    assert row.external_consensus_mid == pytest.approx(102.0)


def test_duplicate_brti_receive_timestamp_is_not_double_counted():
    materializer = FeatureMaterializer()
    materializer.materialize(
        frame(decision_ts_ns=T0, brti_value="100", brti_recv_ts_ns=T0)
    )
    row = materializer.materialize(
        frame(
            decision_ts_ns=T0 + 500_000_000,
            brti_value="100",
            brti_recv_ts_ns=T0,
        )
    )

    assert row.brti_vol_per_sqrt_second is None


def test_stale_brti_features_fail_closed():
    materializer = FeatureMaterializer()
    row = materializer.materialize(
        frame(
            decision_ts_ns=T0,
            brti_value="100",
            brti_recv_ts_ns=T0,
            final_average="100",
            sample_count=20,
            brti_fresh=False,
        )
    )

    assert row.brti is None
    assert row.final_minute_average is None
    assert row.final_minute_sample_count is None
    assert row.required_remaining_brti_average is None
    assert not row.baseline_ready


def test_materializer_rejects_backwards_decision_time():
    materializer = FeatureMaterializer()
    materializer.materialize(
        frame(decision_ts_ns=T0 + 1, brti_value="100", brti_recv_ts_ns=T0 + 1)
    )
    with pytest.raises(ValueError):
        materializer.materialize(
            frame(decision_ts_ns=T0, brti_value="100", brti_recv_ts_ns=T0)
        )
