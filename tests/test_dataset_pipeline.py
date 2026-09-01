from decimal import Decimal

from kalshi_research.domain.events import (
    IndexTickEvent,
    MarketEvent,
    OrderbookSnapshotEvent,
    PriceLevel,
    Source,
    SpotTickEvent,
)
from kalshi_research.research.dataset import (
    FeatureReplayPipeline,
    feature_rows_digest,
    feature_rows_from_store,
)
from kalshi_research.research.synchronizer import SynchronizationError
from kalshi_research.storage.sqlite_store import SqliteEventStore


T0 = 1_800_000_000_000_000_000
TICKER = "KXBTC15M-TEST"


def events():
    return [
        MarketEvent(
            event_ts_ns=T0,
            recv_ts_ns=T0,
            market_ticker=TICKER,
            event_ticker="KXBTC15M",
            series_ticker="KXBTC15M",
            target_price=Decimal("100"),
            open_ts_ns=T0 - 900_000_000_000,
            close_ts_ns=T0 + 60_000_000_000,
            status="open",
        ),
        OrderbookSnapshotEvent(
            event_ts_ns=T0 + 10_000_000,
            recv_ts_ns=T0 + 10_000_000,
            market_ticker=TICKER,
            seq=1,
            yes_bids=(PriceLevel(price=Decimal("0.45"), size=Decimal("5")),),
            no_bids=(PriceLevel(price=Decimal("0.50"), size=Decimal("4")),),
        ),
        IndexTickEvent(
            event_ts_ns=T0 + 20_000_000,
            recv_ts_ns=T0 + 20_000_000,
            value=Decimal("100"),
        ),
        SpotTickEvent(
            source=Source.COINBASE,
            event_ts_ns=T0 + 30_000_000,
            recv_ts_ns=T0 + 30_000_000,
            venue="coinbase",
            bid=Decimal("99.9"),
            ask=Decimal("100.1"),
        ),
        IndexTickEvent(
            event_ts_ns=T0 + 1_020_000_000,
            recv_ts_ns=T0 + 1_020_000_000,
            value=Decimal("101"),
        ),
    ]


def test_store_replay_keeps_untickered_external_events(tmp_path):
    path = tmp_path / "events.sqlite"
    with SqliteEventStore(path) as store:
        store.append_many(events())
        rows = feature_rows_from_store(store, TICKER)

    assert rows
    assert rows[-1].coinbase_mid == 100.0
    assert rows[-1].external_consensus_mid == 100.0
    assert rows[-1].brti == 101.0


def test_dataset_digest_is_deterministic(tmp_path):
    path = tmp_path / "events.sqlite"
    with SqliteEventStore(path) as store:
        store.append_many(events())
        first = feature_rows_from_store(store, TICKER)
        second = feature_rows_from_store(store, TICKER)

    assert feature_rows_digest(first) == feature_rows_digest(second)


def test_probability_ready_filter_only_emits_usable_source_state(tmp_path):
    path = tmp_path / "events.sqlite"
    with SqliteEventStore(path) as store:
        store.append_many(events())
        rows = feature_rows_from_store(store, TICKER, probability_ready_only=True)

    assert rows
    assert all(row.probability_ready for row in rows)


def test_pipeline_rejects_receive_time_disorder():
    ordered = events()
    disordered = [ordered[1], ordered[0]]
    pipeline = FeatureReplayPipeline(market_ticker=TICKER)

    try:
        list(pipeline.run(disordered))
    except SynchronizationError:
        pass
    else:
        raise AssertionError("receive-time disorder must fail closed")
