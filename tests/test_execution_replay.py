from __future__ import annotations

from decimal import Decimal

from kalshi_research.domain.events import (
    FeeScheduleEvent,
    MarketEvent,
    OrderbookDeltaEvent,
    OrderbookSnapshotEvent,
    PriceLevel,
    SettlementEvent,
    Source,
    TradeEvent,
)
from kalshi_research.research.execution_replay import (
    ExecutionPolicy,
    ResearchOrderIntent,
    replay_orders,
    replay_stress_grid,
)


NS = 1_000_000_000
BASE = 1_900_000_000 * NS


def _fee(recv: int = BASE) -> FeeScheduleEvent:
    return FeeScheduleEvent(
        source=Source.KALSHI,
        event_ts_ns=recv,
        recv_ts_ns=recv,
        series_ticker="KXBTC15M",
        fee_type="quadratic",
        fee_multiplier=Decimal("1"),
        effective_ts_ns=recv,
    )


def _market(
    ticker: str,
    *,
    open_offset_s: int = 0,
    close_offset_s: int = 100,
) -> MarketEvent:
    open_ts = BASE + open_offset_s * NS
    close_ts = BASE + close_offset_s * NS
    return MarketEvent(
        event_ts_ns=open_ts,
        recv_ts_ns=open_ts,
        market_ticker=ticker,
        event_ticker="KXBTC15M-TEST",
        series_ticker="KXBTC15M",
        target_price=Decimal("100"),
        open_ts_ns=open_ts,
        close_ts_ns=close_ts,
        status="open",
    )


def _settlement(ticker: str, *, yes: bool, close_offset_s: int = 100) -> SettlementEvent:
    recv = BASE + (close_offset_s + 1) * NS
    return SettlementEvent(
        event_ts_ns=recv,
        recv_ts_ns=recv,
        market_ticker=ticker,
        target_price=Decimal("100"),
        final_value=Decimal("101") if yes else Decimal("99"),
        result="yes" if yes else "no",
    )


def _snapshot(
    ticker: str,
    *,
    recv_offset_s: float = 10.0,
    yes=(("0.49", "10"),),
    no=(("0.49", "10"),),
    seq: int = 1,
) -> OrderbookSnapshotEvent:
    recv = BASE + int(recv_offset_s * NS)
    return OrderbookSnapshotEvent(
        event_ts_ns=recv,
        recv_ts_ns=recv,
        market_ticker=ticker,
        sid=1,
        seq=seq,
        yes_bids=tuple(PriceLevel(price=Decimal(p), size=Decimal(q)) for p, q in yes),
        no_bids=tuple(PriceLevel(price=Decimal(p), size=Decimal(q)) for p, q in no),
    )


def _base_events(ticker: str = "KXBTC15M-R1", *, yes: bool = True):
    return tuple(
        sorted(
            (
                _fee(),
                _market(ticker),
                _snapshot(ticker),
                _settlement(ticker, yes=yes),
            ),
            key=lambda event: (event.recv_ts_ns, event.event_ts_ns),
        )
    )


def test_taker_yes_walks_complementary_no_depth_and_partial_fills():
    ticker = "KXBTC15M-TAKER"
    events = (
        _fee(),
        _market(ticker),
        _snapshot(
            ticker,
            no=(("0.50", "0.40"), ("0.48", "0.35")),
        ),
        _settlement(ticker, yes=True),
    )
    intent = ResearchOrderIntent(
        intent_id="taker",
        market_ticker=ticker,
        decision_recv_ts_ns=BASE + 11 * NS,
        outcome_side="yes",
        quantity=Decimal("1"),
    )
    report = replay_orders(events, [intent], policy=ExecutionPolicy(latency_ns=0))
    order = report.orders[0]

    assert order.status == "partial"
    assert order.filled_quantity == Decimal("0.75")
    assert order.unfilled_quantity == Decimal("0.25")
    assert order.top_price_at_arrival == Decimal("0.50")
    assert order.average_fill_price == Decimal("0.5093333333333333333333333333")
    assert order.payout == Decimal("0.75")
    assert order.net_pnl < order.gross_pnl


def test_latency_uses_only_book_updates_received_by_arrival():
    ticker = "KXBTC15M-LATENCY"
    delta_recv = BASE + 12 * NS
    events = (
        _fee(),
        _market(ticker),
        _snapshot(ticker, recv_offset_s=10, no=(("0.49", "1"),)),
        OrderbookDeltaEvent(
            event_ts_ns=delta_recv,
            recv_ts_ns=delta_recv,
            market_ticker=ticker,
            sid=1,
            seq=2,
            side="no",
            price=Decimal("0.49"),
            delta=Decimal("-1"),
        ),
        OrderbookDeltaEvent(
            event_ts_ns=delta_recv,
            recv_ts_ns=delta_recv,
            market_ticker=ticker,
            sid=1,
            seq=3,
            side="no",
            price=Decimal("0.45"),
            delta=Decimal("1"),
        ),
        _settlement(ticker, yes=True),
    )
    intent = ResearchOrderIntent(
        intent_id="latency",
        market_ticker=ticker,
        decision_recv_ts_ns=BASE + 11 * NS,
        outcome_side="yes",
        quantity=Decimal("1"),
    )

    fast = replay_orders(events, [intent], policy=ExecutionPolicy(latency_ns=0)).orders[0]
    slow = replay_orders(events, [intent], policy=ExecutionPolicy(latency_ns=2 * NS)).orders[0]

    assert fast.average_fill_price == Decimal("0.51")
    assert slow.average_fill_price == Decimal("0.55")
    assert slow.net_pnl < fast.net_pnl


def test_sequence_gap_and_stale_book_fail_closed():
    ticker = "KXBTC15M-GAP"
    gap_recv = BASE + 11 * NS
    events = (
        _fee(),
        _market(ticker),
        _snapshot(ticker, recv_offset_s=10),
        OrderbookDeltaEvent(
            event_ts_ns=gap_recv,
            recv_ts_ns=gap_recv,
            market_ticker=ticker,
            sid=1,
            seq=3,
            side="no",
            price=Decimal("0.49"),
            delta=Decimal("1"),
        ),
        _settlement(ticker, yes=True),
    )
    intent = ResearchOrderIntent(
        intent_id="gap",
        market_ticker=ticker,
        decision_recv_ts_ns=BASE + 12 * NS,
        outcome_side="yes",
        quantity=Decimal("1"),
    )
    gap = replay_orders(events, [intent], policy=ExecutionPolicy(latency_ns=0)).orders[0]
    assert gap.status == "rejected"
    assert gap.reason == "orderbook_sequence_gap"

    stale_events = _base_events("KXBTC15M-STALE")
    stale_intent = ResearchOrderIntent(
        intent_id="stale",
        market_ticker="KXBTC15M-STALE",
        decision_recv_ts_ns=BASE + 20 * NS,
        outcome_side="yes",
        quantity=Decimal("1"),
    )
    stale = replay_orders(
        stale_events,
        [stale_intent],
        policy=ExecutionPolicy(latency_ns=0, max_book_age_ns=NS),
    ).orders[0]
    assert stale.status == "rejected"
    assert stale.reason == "stale_orderbook"


def test_conservative_maker_fill_requires_trade_volume_beyond_queue_ahead():
    ticker = "KXBTC15M-MAKER"
    trade1_recv = BASE + 12 * NS
    trade2_recv = BASE + 13 * NS
    events = (
        _fee(),
        _market(ticker),
        _snapshot(ticker, yes=(("0.49", "2"),), no=(("0.50", "2"),)),
        TradeEvent(
            source=Source.KALSHI,
            event_ts_ns=trade1_recv,
            recv_ts_ns=trade1_recv,
            market_ticker=ticker,
            trade_id="one",
            price=Decimal("0.49"),
            no_price=Decimal("0.51"),
            size=Decimal("1"),
            taker_side="no",
        ),
        TradeEvent(
            source=Source.KALSHI,
            event_ts_ns=trade2_recv,
            recv_ts_ns=trade2_recv,
            market_ticker=ticker,
            trade_id="two",
            price=Decimal("0.49"),
            no_price=Decimal("0.51"),
            size=Decimal("2"),
            taker_side="no",
        ),
        _settlement(ticker, yes=True),
    )
    intent = ResearchOrderIntent(
        intent_id="maker",
        market_ticker=ticker,
        decision_recv_ts_ns=BASE + 11 * NS,
        outcome_side="yes",
        quantity=Decimal("1"),
        liquidity_role="maker",
        limit_price=Decimal("0.49"),
    )
    order = replay_orders(events, [intent], policy=ExecutionPolicy(latency_ns=0)).orders[0]

    assert order.status == "filled"
    assert order.filled_quantity == Decimal("1")
    assert len(order.fills) == 1
    assert order.fills[0].recv_ts_ns == trade2_recv
    assert order.trade_fees == Decimal("0")


def test_maker_does_not_credit_cancellations_unless_policy_explicitly_does():
    ticker = "KXBTC15M-MAKER-CANCEL"
    trade_recv = BASE + 12 * NS
    events = (
        _fee(),
        _market(ticker),
        _snapshot(ticker, yes=(("0.49", "2"),), no=(("0.50", "2"),)),
        TradeEvent(
            source=Source.KALSHI,
            event_ts_ns=trade_recv,
            recv_ts_ns=trade_recv,
            market_ticker=ticker,
            trade_id="one",
            price=Decimal("0.49"),
            no_price=Decimal("0.51"),
            size=Decimal("1.5"),
            taker_side="no",
        ),
        _settlement(ticker, yes=True),
    )
    intent = ResearchOrderIntent(
        intent_id="maker-cancel",
        market_ticker=ticker,
        decision_recv_ts_ns=BASE + 11 * NS,
        outcome_side="yes",
        quantity=Decimal("1"),
        liquidity_role="maker",
        limit_price=Decimal("0.49"),
    )
    conservative = replay_orders(events, [intent], policy=ExecutionPolicy(latency_ns=0)).orders[0]
    optimistic = replay_orders(
        events,
        [intent],
        policy=ExecutionPolicy(latency_ns=0, cancellation_credit_fraction=Decimal("0.5")),
    ).orders[0]

    assert conservative.status == "rejected"
    assert conservative.reason == "maker_queue_not_reached"
    assert optimistic.status == "partial"
    assert optimistic.filled_quantity == Decimal("0.5")


def test_winning_and_losing_contracts_settle_to_actual_binary_outcome():
    winner = _base_events("KXBTC15M-WIN", yes=True)
    loser = _base_events("KXBTC15M-LOSE", yes=False)
    events = tuple(sorted((*winner, *loser), key=lambda event: (event.recv_ts_ns, event.event_ts_ns)))
    intents = [
        ResearchOrderIntent(
            intent_id="win",
            market_ticker="KXBTC15M-WIN",
            decision_recv_ts_ns=BASE + 11 * NS,
            outcome_side="yes",
            quantity=Decimal("1"),
        ),
        ResearchOrderIntent(
            intent_id="lose",
            market_ticker="KXBTC15M-LOSE",
            decision_recv_ts_ns=BASE + 11 * NS,
            outcome_side="yes",
            quantity=Decimal("1"),
        ),
    ]
    report = replay_orders(events, intents, policy=ExecutionPolicy(latency_ns=0))
    by_id = {order.intent_id: order for order in report.orders}

    assert by_id["win"].payout == Decimal("1")
    assert by_id["lose"].payout == Decimal("0")
    assert by_id["win"].net_pnl > 0
    assert by_id["lose"].net_pnl < 0


def test_fee_and_latency_stress_cannot_improve_same_taker_fill_economics():
    ticker = "KXBTC15M-STRESS"
    events = _base_events(ticker, yes=True)
    intent = ResearchOrderIntent(
        intent_id="stress",
        market_ticker=ticker,
        decision_recv_ts_ns=BASE + 11 * NS,
        outcome_side="yes",
        quantity=Decimal("10"),
    )
    reports = replay_stress_grid(
        events,
        [intent],
        base_policy=ExecutionPolicy(latency_ns=0, max_book_age_ns=10 * NS),
    )

    assert reports[-1].net_pnl <= reports[0].net_pnl
    assert reports[-1].trade_fees >= reports[0].trade_fees


def test_overlapping_positions_enforce_bankroll_and_track_locked_capital():
    first = _base_events("KXBTC15M-CAP-1", yes=True)
    second = _base_events("KXBTC15M-CAP-2", yes=True)
    events = tuple(sorted((*first, *second), key=lambda event: (event.recv_ts_ns, event.event_ts_ns)))
    intents = [
        ResearchOrderIntent(
            intent_id="one",
            market_ticker="KXBTC15M-CAP-1",
            decision_recv_ts_ns=BASE + 11 * NS,
            outcome_side="yes",
            quantity=Decimal("1"),
        ),
        ResearchOrderIntent(
            intent_id="two",
            market_ticker="KXBTC15M-CAP-2",
            decision_recv_ts_ns=BASE + 12 * NS,
            outcome_side="yes",
            quantity=Decimal("1"),
        ),
    ]
    report = replay_orders(
        events,
        intents,
        policy=ExecutionPolicy(latency_ns=0, bankroll=Decimal("0.70"), max_book_age_ns=5 * NS),
    )

    assert report.orders[0].status == "filled"
    assert report.orders[1].status == "rejected"
    assert report.orders[1].reason == "bankroll_limit"
    assert report.max_capital_locked == report.orders[0].capital_locked


def test_missing_fee_history_rejects_execution_instead_of_assuming_zero_cost():
    ticker = "KXBTC15M-NOFEE"
    events = (_market(ticker), _snapshot(ticker), _settlement(ticker, yes=True))
    intent = ResearchOrderIntent(
        intent_id="nofee",
        market_ticker=ticker,
        decision_recv_ts_ns=BASE + 11 * NS,
        outcome_side="yes",
        quantity=Decimal("1"),
    )

    try:
        replay_orders(events, [intent], policy=ExecutionPolicy(latency_ns=0))
    except Exception as exc:
        assert "fee schedule" in str(exc)
    else:
        raise AssertionError("missing fee schedule must fail closed")
