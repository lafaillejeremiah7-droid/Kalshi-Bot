from decimal import Decimal

from kalshi_research.domain.events import (
    OrderbookDeltaEvent,
    OrderbookSnapshotEvent,
    PriceLevel,
    Source,
)
from kalshi_research.replay.engine import ReplayEngine
from kalshi_research.storage.sqlite_store import SqliteEventStore


def test_store_deduplicates_and_replays(tmp_path):
    snapshot = OrderbookSnapshotEvent(
        source=Source.KALSHI,
        event_ts_ns=1_000_000_000,
        recv_ts_ns=1_001_000_000,
        market_ticker="KXBTC15M-X",
        seq=1,
        yes_bids=(PriceLevel(price=Decimal("0.40"), size=Decimal("10")),),
        no_bids=(PriceLevel(price=Decimal("0.55"), size=Decimal("10")),),
    )
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
