from __future__ import annotations

from decimal import Decimal

import pytest

from kalshi_research.domain.events import FeeScheduleEvent, Source
from kalshi_research.math.execution import (
    FeeAccumulator,
    quadratic_trade_fee,
    settlement_rounding,
    walk_asks_decimal,
)
from kalshi_research.research.fees import (
    FeeScheduleError,
    FeeScheduleTimeline,
    normalize_series_fee_events,
)
from kalshi_research.storage.sqlite_store import SqliteEventStore


NS = 1_000_000_000


def _fee_event(
    *,
    recv: int,
    effective: int,
    multiplier: str = "1",
    historical: bool = False,
    change_id: str | None = None,
) -> FeeScheduleEvent:
    return FeeScheduleEvent(
        source=Source.KALSHI,
        event_ts_ns=recv,
        recv_ts_ns=recv,
        series_ticker="KXBTC15M",
        fee_change_id=change_id,
        fee_type="quadratic",
        fee_multiplier=Decimal(multiplier),
        effective_ts_ns=effective,
        historical=historical,
    )


def test_quadratic_fee_uses_multiplier_and_centicent_rounding():
    assert quadratic_trade_fee(
        price=Decimal("0.5"),
        quantity=Decimal("1"),
        fee_multiplier=Decimal("1"),
        liquidity_role="taker",
        fee_type="quadratic",
    ) == Decimal("0.0175")
    assert quadratic_trade_fee(
        price=Decimal("0.055"),
        quantity=Decimal("1"),
        fee_multiplier=Decimal("1"),
        liquidity_role="taker",
        fee_type="quadratic",
    ) == Decimal("0.0037")
    assert quadratic_trade_fee(
        price=Decimal("0.5"),
        quantity=Decimal("1"),
        fee_multiplier=Decimal("2"),
        liquidity_role="taker",
        fee_type="quadratic",
    ) == Decimal("0.0350")


def test_quadratic_market_has_zero_maker_trade_fee():
    assert quadratic_trade_fee(
        price=Decimal("0.5"),
        quantity=Decimal("10"),
        fee_multiplier=Decimal("1"),
        liquidity_role="maker",
        fee_type="quadratic",
    ) == Decimal("0")
    assert quadratic_trade_fee(
        price=Decimal("0.5"),
        quantity=Decimal("10"),
        fee_multiplier=Decimal("1"),
        liquidity_role="maker",
        fee_type="quadratic_with_maker_fees",
    ) == Decimal("0.0438")


def test_fee_accumulator_matches_documented_subpenny_example():
    accumulator = FeeAccumulator()
    fills = [
        accumulator.apply_fill(revenue=Decimal("-0.0550"), trade_fee=Decimal("0.0085"))
        for _ in range(3)
    ]

    assert [fill.rounding_fee for fill in fills] == [Decimal("0.0065")] * 3
    assert [fill.rebate for fill in fills] == [Decimal("0"), Decimal("0.01"), Decimal("0")]
    assert [fill.net_fee for fill in fills] == [
        Decimal("0.0150"),
        Decimal("0.0050"),
        Decimal("0.0150"),
    ]
    assert fills[-1].accumulator_after == Decimal("0.0095")


def test_fee_accumulator_matches_documented_fractional_example():
    accumulator = FeeAccumulator()
    fills = [
        accumulator.apply_fill(revenue=Decimal("-0.1500"), trade_fee=Decimal("0.0041"))
        for _ in range(3)
    ]
    assert [fill.rounding_fee for fill in fills] == [Decimal("0.0059")] * 3
    assert [fill.net_fee for fill in fills] == [
        Decimal("0.0100"),
        Decimal("0.0000"),
        Decimal("0.0100"),
    ]


def test_decimal_depth_walk_preserves_partial_fills():
    result = walk_asks_decimal(
        [
            (Decimal("0.51"), Decimal("0.40")),
            (Decimal("0.52"), Decimal("0.35")),
        ],
        Decimal("1.00"),
    )
    assert result.filled_qty == Decimal("0.75")
    assert result.unfilled_qty == Decimal("0.25")
    assert result.notional == Decimal("0.3860")
    assert result.levels_consumed == (
        (Decimal("0.51"), Decimal("0.40")),
        (Decimal("0.52"), Decimal("0.35")),
    )


def test_binary_settlement_rounding_is_cent_aligned():
    rounded = settlement_rounding(Decimal("10.60"), Decimal("1"))
    assert rounded.raw_payout == Decimal("10.60")
    assert rounded.posted_payout == Decimal("10.60")
    assert rounded.settlement_fee == Decimal("0.00")

    scalar_example = settlement_rounding(Decimal("10.60"), Decimal("0.5970"))
    assert scalar_example.raw_payout == Decimal("6.328200")
    assert scalar_example.posted_payout == Decimal("6.32")
    assert scalar_example.settlement_fee == Decimal("0.008200")


def test_fee_timeline_respects_effective_time_and_knowledge_cutoff():
    current = _fee_event(recv=100 * NS, effective=100 * NS, multiplier="1")
    scheduled = _fee_event(
        recv=100 * NS,
        effective=200 * NS,
        multiplier="2",
        change_id="future",
    )
    backfilled = _fee_event(
        recv=300 * NS,
        effective=120 * NS,
        multiplier="1.5",
        historical=True,
        change_id="historical",
    )
    timeline = FeeScheduleTimeline.from_events(
        [current, scheduled, backfilled],
        series_ticker="KXBTC15M",
    )

    assert timeline.at(150 * NS, knowledge_cutoff_ns=150 * NS).fee_multiplier == Decimal("1.5")
    assert timeline.at(
        150 * NS,
        knowledge_cutoff_ns=150 * NS,
        allow_posthoc_history=False,
    ).fee_multiplier == Decimal("1")
    assert timeline.at(250 * NS, knowledge_cutoff_ns=250 * NS).fee_multiplier == Decimal("2")


def test_fee_normalization_and_storage_roundtrip(tmp_path):
    observed = 200 * NS
    events = normalize_series_fee_events(
        series={
            "ticker": "KXBTC15M",
            "fee_type": "quadratic",
            "fee_multiplier": 1,
        },
        fee_changes=[
            {
                "id": "old",
                "series_ticker": "KXBTC15M",
                "fee_type": "quadratic",
                "fee_multiplier": 2,
                "scheduled_ts": "1970-01-01T00:02:00Z",
            }
        ],
        observed_ts_ns=observed,
    )
    assert len(events) == 2
    assert events[0].historical is True
    assert events[-1].historical is False

    with SqliteEventStore(tmp_path / "research.sqlite3") as store:
        assert store.append_many(events) == 2
        replayed = tuple(store.iter_events(order_by="recv_ts"))
    assert replayed == events


def test_fee_timeline_fails_closed_when_no_prior_schedule():
    timeline = FeeScheduleTimeline.from_events(
        [_fee_event(recv=200 * NS, effective=200 * NS)],
        series_ticker="KXBTC15M",
    )
    with pytest.raises(FeeScheduleError, match="no known effective fee schedule"):
        timeline.at(100 * NS)
