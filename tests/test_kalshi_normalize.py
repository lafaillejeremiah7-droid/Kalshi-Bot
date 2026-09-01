from decimal import Decimal

import pytest

from kalshi_research.domain.events import IndexTickEvent, OrderbookDeltaEvent, OrderbookSnapshotEvent
from kalshi_research.feeds.kalshi_normalize import (
    KalshiSchemaError,
    SubscriptionSequenceGuard,
    UnknownKalshiMessage,
    normalize_kalshi_message,
)
from kalshi_research.feeds.kalshi_ws import SequenceGap


RECV_NS = 1_710_000_002_000_000_000


def test_normalizes_fixed_point_orderbook_snapshot():
    event = normalize_kalshi_message(
        {
            "type": "orderbook_snapshot",
            "sid": 2,
            "seq": 10,
            "msg": {
                "market_ticker": "KXBTC15M-TEST",
                "market_id": "9b0f6b43-5b68-4f9f-9f02-9a2d1b8ac1a1",
                "yes_dollars_fp": [["0.4100", "12.50"]],
                "no_dollars_fp": [["0.5500", "7.00"]],
            },
        },
        recv_ts_ns=RECV_NS,
    )
    assert isinstance(event, OrderbookSnapshotEvent)
    assert event.sid == 2
    assert event.seq == 10
    assert event.yes_bids[0].price == Decimal("0.4100")
    assert event.yes_bids[0].size == Decimal("12.50")


def test_normalizes_orderbook_delta_with_exchange_timestamp():
    event = normalize_kalshi_message(
        {
            "type": "orderbook_delta",
            "sid": 2,
            "seq": 11,
            "msg": {
                "market_ticker": "KXBTC15M-TEST",
                "market_id": "9b0f6b43-5b68-4f9f-9f02-9a2d1b8ac1a1",
                "price_dollars": "0.4200",
                "delta_fp": "-2.50",
                "side": "yes",
                "ts_ms": 1_710_000_001_000,
            },
        },
        recv_ts_ns=RECV_NS,
    )
    assert isinstance(event, OrderbookDeltaEvent)
    assert event.event_ts_ns == 1_710_000_001_000_000_000
    assert event.delta == Decimal("-2.50")


def test_normalizes_trade_fields():
    event = normalize_kalshi_message(
        {
            "type": "trade",
            "sid": 5,
            "msg": {
                "trade_id": "d91bc706-ee49-470d-82d8-11418bda6fed",
                "market_ticker": "KXBTC15M-TEST",
                "yes_price_dollars": "0.360",
                "no_price_dollars": "0.640",
                "count_fp": "136.00",
                "taker_side": "no",
                "taker_book_side": "ask",
                "is_block_trade": False,
                "ts": 1_710_000_001,
                "ts_ms": 1_710_000_001_000,
            },
        },
        recv_ts_ns=RECV_NS,
    )
    assert event is not None
    assert event.price == Decimal("0.360")
    assert event.no_price == Decimal("0.640")
    assert event.size == Decimal("136.00")
    assert event.taker_book_side == "ask"


def test_normalizes_brti_final_minute_state():
    event = normalize_kalshi_message(
        {
            "type": "cfbenchmarks_value",
            "sid": 8,
            "seq": 42,
            "msg": {
                "index_id": "BRTI",
                "received_at": 1_710_000_001_100,
                "data": "{\"type\":\"value\",\"id\":\"BRTI\",\"time\":1710000001000,\"value\":\"68000.12\"}",
                "avg_60s_data": {
                    "value": "67998.12000000",
                    "window_size": 59,
                    "window_start_ts_ms": 1_709_999_941_000,
                    "window_end_ts_exclusive": 1_710_000_001_000,
                },
                "last_60s_windowed_average_15min": {
                    "value": "68000.23000000",
                    "window_size": 14,
                    "window_start_ts_ms": 1_709_999_980_000,
                    "window_end_ts_exclusive": 1_710_000_001_000,
                },
            },
        },
        recv_ts_ns=RECV_NS,
    )
    assert isinstance(event, IndexTickEvent)
    assert event.value == Decimal("68000.12")
    assert event.kalshi_received_ts_ns == 1_710_000_001_100_000_000
    assert event.final_minute_average == Decimal("68000.23000000")
    assert event.final_minute_sample_count == 14


def test_control_message_is_not_a_research_event():
    assert (
        normalize_kalshi_message(
            {"type": "subscribed", "id": 1, "msg": {"channel": "trade"}},
            recv_ts_ns=RECV_NS,
        )
        is None
    )


def test_unknown_message_fails_closed():
    with pytest.raises(UnknownKalshiMessage):
        normalize_kalshi_message(
            {"type": "future_magic_channel", "msg": {}}, recv_ts_ns=RECV_NS
        )


def test_known_message_with_unknown_field_is_quarantine_worthy():
    with pytest.raises(KalshiSchemaError):
        normalize_kalshi_message(
            {
                "type": "orderbook_snapshot",
                "sid": 2,
                "seq": 10,
                "unexpected": "schema drift",
                "msg": {
                    "market_ticker": "KXBTC15M-TEST",
                    "market_id": "9b0f6b43-5b68-4f9f-9f02-9a2d1b8ac1a1",
                },
            },
            recv_ts_ns=RECV_NS,
        )


def test_sequence_guard_is_keyed_by_subscription_and_requires_snapshot():
    guard = SubscriptionSequenceGuard()
    snap_a = OrderbookSnapshotEvent(
        source="kalshi",
        event_ts_ns=1,
        recv_ts_ns=1,
        market_ticker="KXBTC15M-A",
        sid=10,
        seq=100,
    )
    snap_b = OrderbookSnapshotEvent(
        source="kalshi",
        event_ts_ns=1,
        recv_ts_ns=1,
        market_ticker="KXBTC15M-B",
        sid=11,
        seq=7,
    )
    guard.observe(snap_a)
    guard.observe(snap_b)
    guard.observe(
        OrderbookDeltaEvent(
            source="kalshi",
            event_ts_ns=2,
            recv_ts_ns=2,
            market_ticker="KXBTC15M-A",
            sid=10,
            seq=101,
            side="yes",
            price=Decimal("0.4"),
            delta=Decimal("1"),
        )
    )
    guard.observe(
        OrderbookDeltaEvent(
            source="kalshi",
            event_ts_ns=2,
            recv_ts_ns=2,
            market_ticker="KXBTC15M-B",
            sid=11,
            seq=8,
            side="no",
            price=Decimal("0.5"),
            delta=Decimal("1"),
        )
    )

    with pytest.raises(SequenceGap):
        guard.observe(
            OrderbookDeltaEvent(
                source="kalshi",
                event_ts_ns=3,
                recv_ts_ns=3,
                market_ticker="KXBTC15M-A",
                sid=10,
                seq=103,
                side="yes",
                price=Decimal("0.4"),
                delta=Decimal("1"),
            )
        )

    fresh = SubscriptionSequenceGuard()
    with pytest.raises(SequenceGap):
        fresh.observe(
            OrderbookDeltaEvent(
                source="kalshi",
                event_ts_ns=2,
                recv_ts_ns=2,
                market_ticker="KXBTC15M-A",
                sid=99,
                seq=2,
                side="yes",
                price=Decimal("0.4"),
                delta=Decimal("1"),
            )
        )
