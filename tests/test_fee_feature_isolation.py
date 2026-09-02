from __future__ import annotations

from decimal import Decimal

from kalshi_research.domain.events import (
    FeeScheduleEvent,
    IndexTickEvent,
    MarketEvent,
    OrderbookSnapshotEvent,
    PriceLevel,
    Source,
    SpotTickEvent,
)
from kalshi_research.research.dataset import FeatureReplayPipeline, feature_rows_digest


NS = 1_000_000_000
BASE = 1_900_000_000 * NS
TICKER = "KXBTC15M-FEE-ISOLATION"


def _base_events():
    open_ts = BASE
    close_ts = BASE + 900 * NS
    return (
        MarketEvent(
            event_ts_ns=open_ts,
            recv_ts_ns=open_ts,
            market_ticker=TICKER,
            event_ticker="KXBTC15M-FEE",
            series_ticker="KXBTC15M",
            target_price=Decimal("100"),
            open_ts_ns=open_ts,
            close_ts_ns=close_ts,
            status="open",
        ),
        IndexTickEvent(
            event_ts_ns=open_ts + NS,
            recv_ts_ns=open_ts + NS,
            value=Decimal("100.10"),
        ),
        SpotTickEvent(
            source=Source.COINBASE,
            event_ts_ns=open_ts + 2 * NS,
            recv_ts_ns=open_ts + 2 * NS,
            venue="coinbase",
            bid=Decimal("100.00"),
            ask=Decimal("100.20"),
        ),
        OrderbookSnapshotEvent(
            event_ts_ns=open_ts + 3 * NS,
            recv_ts_ns=open_ts + 3 * NS,
            market_ticker=TICKER,
            sid=1,
            seq=1,
            yes_bids=(PriceLevel(price=Decimal("0.49"), size=Decimal("10")),),
            no_bids=(PriceLevel(price=Decimal("0.49"), size=Decimal("10")),),
        ),
        IndexTickEvent(
            event_ts_ns=open_ts + 4 * NS,
            recv_ts_ns=open_ts + 4 * NS,
            value=Decimal("100.20"),
        ),
    )


def _rows(events):
    return list(FeatureReplayPipeline(market_ticker=TICKER).run(events))


def test_fee_metadata_cannot_change_predictive_feature_rows_or_digest():
    base = _base_events()
    fee_event = FeeScheduleEvent(
        source=Source.KALSHI,
        event_ts_ns=BASE + 2_500_000_000,
        recv_ts_ns=BASE + 2_500_000_000,
        series_ticker="KXBTC15M",
        fee_change_id="future-fee",
        fee_type="quadratic",
        fee_multiplier=Decimal("9"),
        effective_ts_ns=BASE + 600 * NS,
        historical=False,
    )
    with_fee = tuple(sorted((*base, fee_event), key=lambda event: event.recv_ts_ns))

    baseline_rows = _rows(base)
    changed_rows = _rows(with_fee)

    assert baseline_rows == changed_rows
    assert feature_rows_digest(baseline_rows) == feature_rows_digest(changed_rows)
