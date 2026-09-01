from decimal import Decimal

import pytest

from kalshi_research.domain.events import (
    OrderbookDeltaEvent,
    OrderbookSnapshotEvent,
    PriceLevel,
    Source,
)
from kalshi_research.replay.engine import ReplayEngine
from kalshi_research.storage.sqlite_store import SqliteEventStore


def _snapshot(*, event_ts_ns: int, recv_ts_ns: int) -> OrderbookSnapshotEvent:
    return OrderbookSnapshotEvent(
        source=Source.KALSHI,
        event_ts_ns=event_ts_ns,
        recv_ts_ns=recv_ts_ns,
        market_ticker="KXBTC15M-X",
        seq=1,
        yes_bids=(PriceLevel(price=Decimal("0.40"), size=Decimal("10")),),
        no_bids=(PriceLevel(price=Decimal("0.55"), size=Decimal("10")),),
    )


def test_store_deduplicates_and_replays(tmp_path):
    snapshot = _snapshot(event_ts_ns=1_000_000_000, recv_ts_ns=1_001_000_000)
    delta = OrderbookDeltaEvent(
        source=Source.KALSHI,
        event_ts_ns=2_000_000_000,
        recv_ts_ns=2_001_000_000,
        market_ticker="KXBTC15M-X",
        seq=2,
        side="yes",
        price=Decimal("0.41"),
        delta=Decimal("5"),
    )
    with SqliteEventStore(tmp_path / "r.db") as store:
        assert store.append(snapshot)
        assert not store.append(snapshot)
        assert store.append(delta)
        state = ReplayEngine().run(store.iter_events("KXBTC15M-X"))
    assert state.processed_events == 2
    assert state.books["KXBTC15M-X"].best_yes_bid == Decimal("0.41")


def test_store_receive_order_prevents_future_source_time_from_peeking(tmp_path):
    first_received = _snapshot(event_ts_ns=9_000, recv_ts_ns=100)
    second_received = OrderbookDeltaEvent(
        source=Source.KALSHI,
        event_ts_ns=1_000,
        recv_ts_ns=200,
        market_ticker="KXBTC15M-X",
        seq=2,
        side="yes",
        price=Decimal("0.41"),
        delta=Decimal("1"),
    )
    with SqliteEventStore(tmp_path / "r.db") as store:
        store.append(first_received)
        store.append(second_received)
        ordered = list(store.iter_events("KXBTC15M-X"))
    assert [event.recv_ts_ns for event in ordered] == [100, 200]
    assert [event.event_ts_ns for event in ordered] == [9_000, 1_000]
    assert ReplayEngine().run(ordered).processed_events == 2


def test_replay_rejects_decreasing_receive_time():
    first = _snapshot(event_ts_ns=1_000, recv_ts_ns=200)
    second = OrderbookDeltaEvent(
        source=Source.KALSHI,
        event_ts_ns=2_000,
        recv_ts_ns=100,
        market_ticker="KXBTC15M-X",
        seq=2,
        side="yes",
        price=Decimal("0.41"),
        delta=Decimal("1"),
    )
    with pytest.raises(ValueError, match="recv_ts_ns"):
        ReplayEngine().run([first, second])
