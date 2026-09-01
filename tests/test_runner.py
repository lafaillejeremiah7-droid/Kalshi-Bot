import json
from decimal import Decimal

import pytest

from kalshi_research.domain.events import (
    IndexTickEvent,
    MarketEvent,
    OrderbookSnapshotEvent,
    PriceLevel,
    SettlementEvent,
    Source,
    SpotTickEvent,
)
from kalshi_research.research.coverage import CoveragePolicy
from kalshi_research.research.experiments import ExperimentPlan
from kalshi_research.research.runner import (
    ResearchRunError,
    research_report_digest,
    research_report_json,
    run_research_events,
    run_research_store,
)
from kalshi_research.storage.sqlite_store import SqliteEventStore


NS = 1_000_000_000
BASE = 1_900_000_000 * NS
TARGET = Decimal("100")


def small_plan() -> ExperimentPlan:
    return ExperimentPlan(
        decision_horizons_s=(60,),
        lead_lags_s=(0, 1),
        grid_step_s=1,
        max_asof_age_s=1.5,
        min_leadlag_pairs=3,
        bootstrap_samples=20,
        bootstrap_block_size=2,
        random_seed=31,
        min_train_markets=2,
        validation_markets=1,
        test_markets=1,
        step_markets=1,
    )


def small_coverage() -> CoveragePolicy:
    return CoveragePolicy(
        min_rows=1,
        min_probability_ready_fraction=0.10,
        min_baseline_ready_fraction_of_probability=0.50,
    )


def synthetic_events(
    outcomes: tuple[int, ...] = (1, 0, 1, 0, 1),
    *,
    feature_directions: tuple[int, ...] | None = None,
):
    feature_directions = feature_directions or outcomes
    if len(feature_directions) != len(outcomes):
        raise ValueError("feature_directions and outcomes must have equal length")

    events = []
    for index, (outcome, direction) in enumerate(zip(outcomes, feature_directions)):
        ticker = f"KXBTC15M-TEST-{index}"
        open_ts = BASE + index * 120 * NS
        close_ts = open_ts + 100 * NS
        metadata_recv = open_ts - NS
        events.append(
            MarketEvent(
                event_ts_ns=metadata_recv,
                recv_ts_ns=metadata_recv,
                market_ticker=ticker,
                event_ticker="KXBTC15M-TEST",
                series_ticker="KXBTC15M",
                target_price=TARGET,
                open_ts_ns=open_ts,
                close_ts_ns=close_ts,
                status="open",
            )
        )

        first_brti = Decimal("100.8") if direction else Decimal("99.2")
        second_brti = Decimal("101.0") if direction else Decimal("99.0")
        brti_one_recv = close_ts - 61_500_000_000
        brti_two_recv = close_ts - 60_900_000_000
        events.extend(
            [
                IndexTickEvent(
                    event_ts_ns=brti_one_recv,
                    recv_ts_ns=brti_one_recv,
                    value=first_brti,
                ),
                IndexTickEvent(
                    event_ts_ns=brti_two_recv,
                    recv_ts_ns=brti_two_recv,
                    value=second_brti,
                ),
            ]
        )

        coinbase_recv = close_ts - 60_700_000_000
        kraken_recv = close_ts - 60_600_000_000
        events.extend(
            [
                SpotTickEvent(
                    source=Source.COINBASE,
                    event_ts_ns=coinbase_recv,
                    recv_ts_ns=coinbase_recv,
                    venue="coinbase",
                    bid=second_brti - Decimal("0.10"),
                    ask=second_brti + Decimal("0.10"),
                ),
                SpotTickEvent(
                    source=Source.KRAKEN,
                    event_ts_ns=kraken_recv,
                    recv_ts_ns=kraken_recv,
                    venue="kraken",
                    bid=second_brti - Decimal("0.12"),
                    ask=second_brti + Decimal("0.12"),
                ),
            ]
        )

        book_recv = close_ts - 60_500_000_000
        events.append(
            OrderbookSnapshotEvent(
                event_ts_ns=book_recv,
                recv_ts_ns=book_recv,
                market_ticker=ticker,
                sid=index + 1,
                seq=1,
                yes_bids=(
                    PriceLevel(price=Decimal("0.49"), size=Decimal("10")),
                ),
                no_bids=(
                    PriceLevel(price=Decimal("0.49"), size=Decimal("10")),
                ),
            )
        )

        settlement_recv = close_ts + NS
        final_value = Decimal("101") if outcome else Decimal("99")
        events.append(
            SettlementEvent(
                event_ts_ns=settlement_recv,
                recv_ts_ns=settlement_recv,
                market_ticker=ticker,
                target_price=TARGET,
                final_value=final_value,
                result="yes" if outcome else "no",
            )
        )

    return tuple(sorted(events, key=lambda event: (event.recv_ts_ns, event.event_ts_ns)))


def run(events):
    return run_research_events(
        events,
        plan=small_plan(),
        coverage_policy=small_coverage(),
    )


def test_runner_is_deterministic_research_only_and_serializable():
    events = synthetic_events()

    first = run(events)
    second = run(events)

    assert first == second
    assert first.mode == "research_only"
    assert first.order_placement is False
    assert first.market_count == 5
    assert len(first.probability_benchmarks) == 2
    assert all(market.coverage.passed for market in first.markets)
    assert all(market.feature_rows >= 5 for market in first.markets)
    assert research_report_digest(first) == research_report_digest(second)

    payload = json.loads(research_report_json(first))
    assert payload["order_placement"] is False
    assert payload["series_ticker"] == "KXBTC15M"
    assert len(payload["markets"]) == 5


def test_settlement_label_changes_do_not_change_preclose_feature_digests():
    directions = (1, 0, 1, 0, 1)
    first = run(synthetic_events(directions, feature_directions=directions))
    changed_outcomes = (0, 0, 1, 0, 1)
    second = run(
        synthetic_events(
            changed_outcomes,
            feature_directions=directions,
        )
    )

    first_digests = {market.market_ticker: market.feature_digest for market in first.markets}
    second_digests = {market.market_ticker: market.feature_digest for market in second.markets}
    assert first_digests == second_digests
    assert first.events_digest != second.events_digest


def test_preopen_global_history_does_not_change_market_feature_digests():
    events = synthetic_events()
    baseline = run(events)
    preopen_recv = BASE - 500_000_000
    with_preopen = tuple(
        sorted(
            (
                *events,
                IndexTickEvent(
                    event_ts_ns=preopen_recv,
                    recv_ts_ns=preopen_recv,
                    value=Decimal("50000"),
                ),
            ),
            key=lambda event: (event.recv_ts_ns, event.event_ts_ns),
        )
    )

    changed = run(with_preopen)

    baseline_digests = tuple(market.feature_digest for market in baseline.markets)
    changed_digests = tuple(market.feature_digest for market in changed.markets)
    assert baseline_digests == changed_digests
    assert baseline.events_digest != changed.events_digest


def test_runner_rejects_conflicting_settlement_labels():
    events = list(synthetic_events())
    first_settlement = next(
        event for event in events if isinstance(event, SettlementEvent) and event.market_ticker.endswith("-0")
    )
    last_recv = max(event.recv_ts_ns for event in events)
    events.append(
        SettlementEvent(
            event_ts_ns=last_recv + NS,
            recv_ts_ns=last_recv + NS,
            market_ticker=first_settlement.market_ticker,
            target_price=TARGET,
            final_value=Decimal("99"),
            result="no",
        )
    )
    events.sort(key=lambda event: (event.recv_ts_ns, event.event_ts_ns))

    with pytest.raises(ResearchRunError, match="conflicting settlement labels"):
        run(tuple(events))


def test_runner_rejects_settlement_received_before_close():
    events = list(synthetic_events())
    market = next(
        event for event in events if isinstance(event, MarketEvent) and event.market_ticker.endswith("-0")
    )
    original = next(
        event for event in events if isinstance(event, SettlementEvent) and event.market_ticker == market.market_ticker
    )
    events.remove(original)
    premature_recv = market.close_ts_ns - NS
    events.append(
        SettlementEvent(
            event_ts_ns=premature_recv,
            recv_ts_ns=premature_recv,
            market_ticker=market.market_ticker,
            target_price=TARGET,
            final_value=original.final_value,
            result=original.result,
        )
    )
    events.sort(key=lambda event: (event.recv_ts_ns, event.event_ts_ns))

    with pytest.raises(ResearchRunError, match="settlement received before market close"):
        run(tuple(events))


def test_structural_audit_failure_stops_experiments_before_replay():
    events = list(synthetic_events())
    events[0], events[1] = events[1], events[0]

    with pytest.raises(ResearchRunError, match="structural audit failed"):
        run(tuple(events))


def test_store_runner_matches_in_memory_runner(tmp_path):
    events = synthetic_events()
    expected = run(events)
    path = tmp_path / "research.sqlite3"

    with SqliteEventStore(path) as store:
        assert store.append_many(events) == len(events)
        actual = run_research_store(
            store,
            plan=small_plan(),
            coverage_policy=small_coverage(),
        )

    assert actual == expected
    assert research_report_digest(actual) == research_report_digest(expected)
