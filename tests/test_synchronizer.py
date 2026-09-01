from decimal import Decimal

import pytest

from kalshi_research.domain.events import (
    IndexTickEvent,
    MarketEvent,
    OrderbookDeltaEvent,
    OrderbookSnapshotEvent,
    PriceLevel,
    Source,
    SpotTickEvent,
)
from kalshi_research.research.synchronizer import (
    FreshnessPolicy,
    ReceiveTimeSynchronizer,
    SynchronizationError,
)


T0 = 1_800_000_000_000_000_000
TICKER = "KXBTC15M-TEST"


def market_event(recv_ts_ns: int = T0) -> MarketEvent:
    return MarketEvent(
        event_ts_ns=recv_ts_ns,
        recv_ts_ns=recv_ts_ns,
        market_ticker=TICKER,
        event_ticker="KXBTC15M",
        series_ticker="KXBTC15M",
        target_price=Decimal("60000"),
        open_ts_ns=T0 - 900_000_000_000,
        close_ts_ns=T0 + 60_000_000_000,
        status="open",
    )


def snapshot(recv_ts_ns: int = T0 + 10_000_000, seq: int = 10) -> OrderbookSnapshotEvent:
    return OrderbookSnapshotEvent(
        event_ts_ns=recv_ts_ns,
        recv_ts_ns=recv_ts_ns,
        market_ticker=TICKER,
        seq=seq,
        yes_bids=(PriceLevel(price=Decimal("0.47"), size=Decimal("8")),),
        no_bids=(PriceLevel(price=Decimal("0.50"), size=Decimal("5")),),
    )


def brti(recv_ts_ns: int = T0 + 20_000_000) -> IndexTickEvent:
    return IndexTickEvent(
        event_ts_ns=recv_ts_ns,
        recv_ts_ns=recv_ts_ns,
        value=Decimal("60012.5"),
        final_minute_average=Decimal("60008.25"),
        final_minute_sample_count=17,
    )


def spot(source: Source, recv_ts_ns: int, bid: str, ask: str) -> SpotTickEvent:
    return SpotTickEvent(
        source=source,
        event_ts_ns=recv_ts_ns,
        recv_ts_ns=recv_ts_ns,
        venue=str(source),
        bid=Decimal(bid),
        ask=Decimal(ask),
        bid_size=Decimal("2"),
        ask_size=Decimal("3"),
    )


def test_freshness_policy_rejects_nonpositive_thresholds():
    with pytest.raises(ValueError):
        FreshnessPolicy(brti_ns=0)


def test_frame_requires_received_data():
    sync = ReceiveTimeSynchronizer()
    with pytest.raises(SynchronizationError):
        sync.frame(TICKER)


def test_receive_time_must_be_monotonic():
    sync = ReceiveTimeSynchronizer()
    sync.ingest(market_event(T0 + 2))
    with pytest.raises(SynchronizationError):
        sync.ingest(market_event(T0 + 1))


def test_materializes_feature_ready_frame_without_lookahead():
    sync = ReceiveTimeSynchronizer()
    sync.ingest(market_event())
    sync.ingest(snapshot())
    sync.ingest(brti())
    sync.ingest(spot(Source.COINBASE, T0 + 30_000_000, "60010", "60014"))

    frame = sync.frame(TICKER)

    assert frame.target_price == Decimal("60000")
    assert frame.yes_bid == Decimal("0.47")
    assert frame.yes_bid_size == Decimal("8")
    assert frame.yes_ask == Decimal("0.50")
    assert frame.yes_ask_size == Decimal("5")
    assert frame.yes_spread == Decimal("0.03")
    assert frame.brti == Decimal("60012.5")
    assert frame.brti_final_minute_average == Decimal("60008.25")
    assert frame.brti_final_minute_sample_count == 17
    assert frame.coinbase_mid == Decimal("60012")
    assert frame.probability_ready


def test_backdated_frame_is_rejected_after_later_data_was_ingested():
    sync = ReceiveTimeSynchronizer()
    sync.ingest(market_event())
    sync.ingest(brti(T0 + 100))

    with pytest.raises(SynchronizationError):
        sync.frame(TICKER, decision_recv_ts_ns=T0 + 50)


def test_future_decision_marks_sources_stale_instead_of_forward_filling_freshness():
    policy = FreshnessPolicy(
        kalshi_book_ns=100,
        brti_ns=100,
        coinbase_ns=100,
        kraken_ns=100,
        market_metadata_ns=1_000,
    )
    sync = ReceiveTimeSynchronizer(policy=policy)
    sync.ingest(market_event())
    sync.ingest(snapshot(T0 + 10))
    sync.ingest(brti(T0 + 20))
    sync.ingest(spot(Source.COINBASE, T0 + 30, "60010", "60014"))

    frame = sync.frame(TICKER, decision_recv_ts_ns=T0 + 500)

    assert frame.brti_state.age_ns == 480
    assert not frame.brti_state.fresh
    assert not frame.coinbase_state.fresh
    assert not frame.kalshi_book_state.fresh
    assert not frame.probability_ready


def test_orderbook_delta_updates_only_after_valid_sequence():
    sync = ReceiveTimeSynchronizer()
    sync.ingest(snapshot(seq=10))
    sync.ingest(
        OrderbookDeltaEvent(
            event_ts_ns=T0 + 20_000_000,
            recv_ts_ns=T0 + 20_000_000,
            market_ticker=TICKER,
            seq=11,
            side="yes",
            price=Decimal("0.49"),
            delta=Decimal("4"),
        )
    )

    frame = sync.frame(TICKER)
    assert frame.yes_bid == Decimal("0.49")
    assert frame.yes_bid_size == Decimal("4")


def test_external_source_is_not_required_from_both_venues():
    sync = ReceiveTimeSynchronizer()
    sync.ingest(market_event())
    sync.ingest(snapshot())
    sync.ingest(brti())
    sync.ingest(spot(Source.KRAKEN, T0 + 30_000_000, "60011", "60013"))

    frame = sync.frame(TICKER)
    assert frame.kraken_mid == Decimal("60012")
    assert frame.coinbase_mid is None
    assert frame.any_external_fresh
    assert frame.probability_ready
