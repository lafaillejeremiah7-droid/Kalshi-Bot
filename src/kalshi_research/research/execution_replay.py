from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Literal, Sequence

from kalshi_research.domain.events import (
    FeeScheduleEvent,
    MarketEvent,
    OrderbookDeltaEvent,
    OrderbookSnapshotEvent,
    ResearchEvent,
    SettlementEvent,
    TradeEvent,
)
from kalshi_research.math.execution import (
    FeeAccumulator,
    quadratic_trade_fee,
    settlement_rounding,
    walk_asks_decimal,
)
from kalshi_research.research.fees import FeeScheduleError, FeeScheduleTimeline


class ExecutionReplayError(RuntimeError):
    """Raised when execution economics cannot be evaluated without guessing."""


@dataclass(frozen=True, slots=True)
class ResearchOrderIntent:
    intent_id: str
    market_ticker: str
    decision_recv_ts_ns: int
    outcome_side: Literal["yes", "no"]
    quantity: Decimal
    liquidity_role: Literal["taker", "maker"] = "taker"
    limit_price: Decimal | None = None
    expires_ts_ns: int | None = None
    attribution: str = "directional"

    def __post_init__(self) -> None:
        if not self.intent_id:
            raise ValueError("intent_id is required")
        if not self.market_ticker:
            raise ValueError("market_ticker is required")
        if self.decision_recv_ts_ns <= 0:
            raise ValueError("decision receive timestamp must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.liquidity_role == "maker":
            if self.limit_price is None:
                raise ValueError("maker intents require a limit price")
            if not Decimal("0") <= self.limit_price <= Decimal("1"):
                raise ValueError("maker limit price must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    latency_ns: int = 100_000_000
    max_book_age_ns: int = 2_000_000_000
    fee_stress_multiplier: Decimal = Decimal("1")
    cancellation_credit_fraction: Decimal = Decimal("0")
    bankroll: Decimal | None = None

    def __post_init__(self) -> None:
        if self.latency_ns < 0:
            raise ValueError("latency_ns cannot be negative")
        if self.max_book_age_ns < 0:
            raise ValueError("max_book_age_ns cannot be negative")
        if self.fee_stress_multiplier <= 0:
            raise ValueError("fee_stress_multiplier must be positive")
        if not Decimal("0") <= self.cancellation_credit_fraction <= Decimal("1"):
            raise ValueError("cancellation credit must be within [0, 1]")
        if self.bankroll is not None and self.bankroll <= 0:
            raise ValueError("bankroll must be positive")


@dataclass(frozen=True, slots=True)
class SimulatedFill:
    recv_ts_ns: int
    price: Decimal
    quantity: Decimal
    liquidity_role: str
    trade_fee: Decimal
    rounding_fee: Decimal
    rebate: Decimal
    net_fee: Decimal


@dataclass(frozen=True, slots=True)
class ExecutedOrder:
    intent_id: str
    market_ticker: str
    attribution: str
    outcome_side: str
    liquidity_role: str
    decision_recv_ts_ns: int
    arrival_ts_ns: int
    settlement_recv_ts_ns: int | None
    requested_quantity: Decimal
    filled_quantity: Decimal
    unfilled_quantity: Decimal
    status: Literal["filled", "partial", "rejected"]
    reason: str | None
    fills: tuple[SimulatedFill, ...]
    entry_notional: Decimal
    trade_fees: Decimal
    rounding_fees: Decimal
    rebates: Decimal
    net_entry_fees: Decimal
    settlement_fee: Decimal
    payout: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    top_price_at_arrival: Decimal | None
    average_fill_price: Decimal | None
    slippage: Decimal
    capital_locked: Decimal


@dataclass(frozen=True, slots=True)
class ExecutionReplayReport:
    policy: ExecutionPolicy
    orders: tuple[ExecutedOrder, ...]
    gross_pnl: Decimal
    net_pnl: Decimal
    trade_fees: Decimal
    rounding_fees: Decimal
    rebates: Decimal
    settlement_fees: Decimal
    slippage: Decimal
    max_drawdown: Decimal
    max_capital_locked: Decimal
    filled_orders: int
    partial_orders: int
    rejected_orders: int
    rejection_reasons: tuple[tuple[str, int], ...]
    pnl_by_attribution: tuple[tuple[str, Decimal], ...]


@dataclass(slots=True)
class _BookState:
    yes: dict[Decimal, Decimal]
    no: dict[Decimal, Decimal]
    last_seq: int | None = None
    last_recv_ts_ns: int | None = None
    broken: bool = False

    @classmethod
    def empty(cls) -> "_BookState":
        return cls(yes={}, no={})

    def snapshot(self, event: OrderbookSnapshotEvent) -> None:
        self.yes = {level.price: level.size for level in event.yes_bids if level.size > 0}
        self.no = {level.price: level.size for level in event.no_bids if level.size > 0}
        self.last_seq = event.seq
        self.last_recv_ts_ns = event.recv_ts_ns
        self.broken = False

    def delta(self, event: OrderbookDeltaEvent) -> None:
        if self.last_seq is None or self.broken:
            self.broken = True
            return
        if event.seq != self.last_seq + 1:
            self.broken = True
            return
        levels = self.yes if event.side == "yes" else self.no
        new_size = levels.get(event.price, Decimal("0")) + event.delta
        if new_size < 0:
            self.broken = True
            return
        if new_size == 0:
            levels.pop(event.price, None)
        else:
            levels[event.price] = new_size
        self.last_seq = event.seq
        self.last_recv_ts_ns = event.recv_ts_ns


def replay_orders(
    events: Iterable[ResearchEvent],
    intents: Iterable[ResearchOrderIntent],
    *,
    policy: ExecutionPolicy | None = None,
    series_ticker: str = "KXBTC15M",
) -> ExecutionReplayReport:
    """Replay research-only order intents against receive-time-safe market data.

    Taker buys walk only visible complementary-side depth present by arrival time.
    Maker buys receive zero cancellation credit by default and fill only after
    observed opposing taker volume exceeds visible queue ahead. Every fill uses the
    exchange fee schedule effective at fill time. Positions are held to settlement;
    there is no synthetic candle-close exit and no order-placement authority.
    """
    policy = policy or ExecutionPolicy()
    materialized = tuple(events)
    sorted_events = tuple(sorted(materialized, key=lambda event: (event.recv_ts_ns, event.event_ts_ns)))
    materialized_intents = tuple(sorted(intents, key=lambda intent: (intent.decision_recv_ts_ns, intent.intent_id)))
    if len({intent.intent_id for intent in materialized_intents}) != len(materialized_intents):
        raise ExecutionReplayError("intent ids must be unique")

    markets = _markets(sorted_events, series_ticker)
    settlements = _settlements(sorted_events, markets)
    fee_events = tuple(event for event in sorted_events if isinstance(event, FeeScheduleEvent))
    if materialized_intents and not fee_events:
        raise ExecutionReplayError("execution replay requires canonical fee schedule events")
    timeline = (
        FeeScheduleTimeline.from_events(fee_events, series_ticker=series_ticker)
        if fee_events
        else None
    )

    executed: list[ExecutedOrder] = []
    active_locks: list[tuple[int, Decimal]] = []
    for intent in materialized_intents:
        market = markets.get(intent.market_ticker)
        settlement = settlements.get(intent.market_ticker)
        arrival = intent.decision_recv_ts_ns + policy.latency_ns
        active_locks = [(end, amount) for end, amount in active_locks if end > arrival]
        currently_locked = sum((amount for _, amount in active_locks), Decimal("0"))

        result = _replay_one(
            sorted_events,
            intent,
            market=market,
            settlement=settlement,
            timeline=timeline,
            policy=policy,
        )
        if (
            result.status != "rejected"
            and policy.bankroll is not None
            and currently_locked + result.capital_locked > policy.bankroll
        ):
            result = _rejected(
                intent,
                arrival,
                reason="bankroll_limit",
                settlement_recv_ts_ns=(settlement.recv_ts_ns if settlement else None),
            )
        elif result.status != "rejected" and result.capital_locked > 0 and settlement is not None:
            active_locks.append((settlement.recv_ts_ns, result.capital_locked))
        executed.append(result)

    return _summarize(policy, tuple(executed))


def default_stress_policies(base: ExecutionPolicy | None = None) -> tuple[ExecutionPolicy, ...]:
    base = base or ExecutionPolicy()
    return (
        base,
        ExecutionPolicy(
            latency_ns=max(base.latency_ns * 2, base.latency_ns + 100_000_000),
            max_book_age_ns=base.max_book_age_ns,
            fee_stress_multiplier=base.fee_stress_multiplier,
            cancellation_credit_fraction=base.cancellation_credit_fraction,
            bankroll=base.bankroll,
        ),
        ExecutionPolicy(
            latency_ns=max(base.latency_ns * 5, base.latency_ns + 500_000_000),
            max_book_age_ns=base.max_book_age_ns,
            fee_stress_multiplier=base.fee_stress_multiplier * Decimal("1.5"),
            cancellation_credit_fraction=Decimal("0"),
            bankroll=base.bankroll,
        ),
    )


def replay_stress_grid(
    events: Iterable[ResearchEvent],
    intents: Iterable[ResearchOrderIntent],
    *,
    base_policy: ExecutionPolicy | None = None,
    series_ticker: str = "KXBTC15M",
) -> tuple[ExecutionReplayReport, ...]:
    materialized_events = tuple(events)
    materialized_intents = tuple(intents)
    return tuple(
        replay_orders(
            materialized_events,
            materialized_intents,
            policy=policy,
            series_ticker=series_ticker,
        )
        for policy in default_stress_policies(base_policy)
    )


def _markets(events: Sequence[ResearchEvent], series_ticker: str) -> dict[str, MarketEvent]:
    markets: dict[str, MarketEvent] = {}
    for event in events:
        if not isinstance(event, MarketEvent) or event.series_ticker != series_ticker:
            continue
        previous = markets.get(event.market_ticker)
        if previous is not None and (
            previous.target_price != event.target_price
            or previous.open_ts_ns != event.open_ts_ns
            or previous.close_ts_ns != event.close_ts_ns
        ):
            raise ExecutionReplayError(f"conflicting market metadata:{event.market_ticker}")
        markets[event.market_ticker] = event
    return markets


def _settlements(
    events: Sequence[ResearchEvent],
    markets: dict[str, MarketEvent],
) -> dict[str, SettlementEvent]:
    settlements: dict[str, SettlementEvent] = {}
    for event in events:
        if not isinstance(event, SettlementEvent) or event.market_ticker not in markets:
            continue
        market = markets[event.market_ticker]
        if event.recv_ts_ns < market.close_ts_ns:
            raise ExecutionReplayError(f"premature settlement:{event.market_ticker}")
        previous = settlements.get(event.market_ticker)
        if previous is not None and (
            previous.result != event.result or previous.final_value != event.final_value
        ):
            raise ExecutionReplayError(f"conflicting settlement:{event.market_ticker}")
        settlements[event.market_ticker] = event
    return settlements


def _replay_one(
    events: Sequence[ResearchEvent],
    intent: ResearchOrderIntent,
    *,
    market: MarketEvent | None,
    settlement: SettlementEvent | None,
    timeline: FeeScheduleTimeline | None,
    policy: ExecutionPolicy,
) -> ExecutedOrder:
    arrival = intent.decision_recv_ts_ns + policy.latency_ns
    if market is None:
        return _rejected(intent, arrival, reason="missing_market_metadata")
    if settlement is None:
        return _rejected(intent, arrival, reason="missing_settlement")
    if arrival < market.open_ts_ns:
        return _rejected(intent, arrival, reason="market_not_open", settlement_recv_ts_ns=settlement.recv_ts_ns)
    if arrival >= market.close_ts_ns:
        return _rejected(intent, arrival, reason="arrived_at_or_after_close", settlement_recv_ts_ns=settlement.recv_ts_ns)
    if timeline is None:
        return _rejected(intent, arrival, reason="missing_fee_schedule", settlement_recv_ts_ns=settlement.recv_ts_ns)

    book, book_reason = _book_at(events, intent.market_ticker, arrival)
    if book_reason is not None:
        return _rejected(intent, arrival, reason=book_reason, settlement_recv_ts_ns=settlement.recv_ts_ns)
    if book.last_recv_ts_ns is None or arrival - book.last_recv_ts_ns > policy.max_book_age_ns:
        return _rejected(intent, arrival, reason="stale_orderbook", settlement_recv_ts_ns=settlement.recv_ts_ns)

    try:
        if intent.liquidity_role == "taker":
            fills, top_price, unfilled = _taker_fills(book, intent, arrival, timeline, policy)
        else:
            fills, top_price, unfilled = _maker_fills(
                events,
                book,
                intent,
                arrival,
                market.close_ts_ns,
                timeline,
                policy,
            )
    except FeeScheduleError:
        return _rejected(intent, arrival, reason="fee_schedule_unavailable", settlement_recv_ts_ns=settlement.recv_ts_ns)
    except ValueError as exc:
        return _rejected(intent, arrival, reason=f"execution_model_error:{exc}", settlement_recv_ts_ns=settlement.recv_ts_ns)

    filled = sum((fill.quantity for fill in fills), Decimal("0"))
    if filled <= 0:
        reason = "no_executable_depth" if intent.liquidity_role == "taker" else "maker_queue_not_reached"
        return _rejected(intent, arrival, reason=reason, settlement_recv_ts_ns=settlement.recv_ts_ns)

    entry_notional = sum((fill.price * fill.quantity for fill in fills), Decimal("0"))
    trade_fees = sum((fill.trade_fee for fill in fills), Decimal("0"))
    rounding_fees = sum((fill.rounding_fee for fill in fills), Decimal("0"))
    rebates = sum((fill.rebate for fill in fills), Decimal("0"))
    net_entry_fees = sum((fill.net_fee for fill in fills), Decimal("0"))
    winning = settlement.result == intent.outcome_side
    settlement_result = settlement_rounding(filled, Decimal("1") if winning else Decimal("0"))
    payout = settlement_result.posted_payout
    gross_pnl = payout - entry_notional
    net_pnl = gross_pnl - net_entry_fees
    average = entry_notional / filled
    slippage = Decimal("0") if top_price is None else (average - top_price) * filled
    capital_locked = entry_notional + net_entry_fees
    status: Literal["filled", "partial", "rejected"] = "filled" if unfilled == 0 else "partial"
    return ExecutedOrder(
        intent_id=intent.intent_id,
        market_ticker=intent.market_ticker,
        attribution=intent.attribution,
        outcome_side=intent.outcome_side,
        liquidity_role=intent.liquidity_role,
        decision_recv_ts_ns=intent.decision_recv_ts_ns,
        arrival_ts_ns=arrival,
        settlement_recv_ts_ns=settlement.recv_ts_ns,
        requested_quantity=intent.quantity,
        filled_quantity=filled,
        unfilled_quantity=unfilled,
        status=status,
        reason=None,
        fills=fills,
        entry_notional=entry_notional,
        trade_fees=trade_fees,
        rounding_fees=rounding_fees,
        rebates=rebates,
        net_entry_fees=net_entry_fees,
        settlement_fee=settlement_result.settlement_fee,
        payout=payout,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        top_price_at_arrival=top_price,
        average_fill_price=average,
        slippage=slippage,
        capital_locked=capital_locked,
    )


def _book_at(
    events: Sequence[ResearchEvent],
    ticker: str,
    arrival_ts_ns: int,
) -> tuple[_BookState, str | None]:
    state = _BookState.empty()
    for event in events:
        if event.recv_ts_ns > arrival_ts_ns:
            break
        if event.market_ticker != ticker:
            continue
        if isinstance(event, OrderbookSnapshotEvent):
            state.snapshot(event)
        elif isinstance(event, OrderbookDeltaEvent):
            state.delta(event)
    if state.last_seq is None:
        return state, "missing_orderbook_snapshot"
    if state.broken:
        return state, "orderbook_sequence_gap"
    return state, None


def _asks(book: _BookState, outcome_side: str) -> tuple[tuple[Decimal, Decimal], ...]:
    opposite = book.no if outcome_side == "yes" else book.yes
    levels = [(Decimal("1") - price, quantity) for price, quantity in opposite.items() if quantity > 0]
    return tuple(sorted(levels, key=lambda item: item[0]))


def _taker_fills(
    book: _BookState,
    intent: ResearchOrderIntent,
    arrival: int,
    timeline: FeeScheduleTimeline,
    policy: ExecutionPolicy,
) -> tuple[tuple[SimulatedFill, ...], Decimal | None, Decimal]:
    asks = _asks(book, intent.outcome_side)
    top_price = asks[0][0] if asks else None
    walked = walk_asks_decimal(asks, intent.quantity) if asks else None
    if walked is None or walked.filled_qty <= 0:
        return (), top_price, intent.quantity
    accumulator = FeeAccumulator()
    fills: list[SimulatedFill] = []
    schedule = timeline.at(arrival, knowledge_cutoff_ns=arrival)
    for price, quantity in walked.levels_consumed:
        fee = quadratic_trade_fee(
            price=price,
            quantity=quantity,
            fee_multiplier=schedule.fee_multiplier * policy.fee_stress_multiplier,
            liquidity_role="taker",
            fee_type=schedule.fee_type,
        )
        breakdown = accumulator.apply_fill(revenue=-(price * quantity), trade_fee=fee)
        fills.append(
            SimulatedFill(
                recv_ts_ns=arrival,
                price=price,
                quantity=quantity,
                liquidity_role="taker",
                trade_fee=breakdown.trade_fee,
                rounding_fee=breakdown.rounding_fee,
                rebate=breakdown.rebate,
                net_fee=breakdown.net_fee,
            )
        )
    return tuple(fills), top_price, walked.unfilled_qty


def _maker_fills(
    events: Sequence[ResearchEvent],
    book: _BookState,
    intent: ResearchOrderIntent,
    arrival: int,
    close_ts_ns: int,
    timeline: FeeScheduleTimeline,
    policy: ExecutionPolicy,
) -> tuple[tuple[SimulatedFill, ...], Decimal | None, Decimal]:
    assert intent.limit_price is not None
    own_book = book.yes if intent.outcome_side == "yes" else book.no
    best_bid = max(own_book, default=None)
    if best_bid is not None and intent.limit_price > Decimal("1") - (
        max(book.no if intent.outcome_side == "yes" else book.yes, default=Decimal("0"))
    ):
        raise ValueError("maker limit would cross the spread")
    queue_ahead = own_book.get(intent.limit_price, Decimal("0"))
    cancellation_credit = queue_ahead * policy.cancellation_credit_fraction
    effective_queue = max(Decimal("0"), queue_ahead - cancellation_credit)
    expiry = min(intent.expires_ts_ns or close_ts_ns, close_ts_ns)
    if expiry <= arrival:
        return (), best_bid, intent.quantity

    opposing_taker = "no" if intent.outcome_side == "yes" else "yes"
    traded_through = Decimal("0")
    filled = Decimal("0")
    accumulator = FeeAccumulator()
    fills: list[SimulatedFill] = []
    for event in events:
        if event.recv_ts_ns <= arrival:
            continue
        if event.recv_ts_ns > expiry:
            break
        if not isinstance(event, TradeEvent) or event.market_ticker != intent.market_ticker:
            continue
        if event.is_block_trade:
            continue
        if event.taker_side != opposing_taker:
            continue
        trade_price = event.price if intent.outcome_side == "yes" else event.no_price
        if trade_price is None or trade_price > intent.limit_price:
            continue
        prior_available = max(Decimal("0"), traded_through - effective_queue)
        traded_through += event.size
        now_available = max(Decimal("0"), traded_through - effective_queue)
        incremental = min(intent.quantity - filled, now_available - prior_available)
        if incremental <= 0:
            continue
        schedule = timeline.at(event.recv_ts_ns, knowledge_cutoff_ns=event.recv_ts_ns)
        fee = quadratic_trade_fee(
            price=intent.limit_price,
            quantity=incremental,
            fee_multiplier=schedule.fee_multiplier * policy.fee_stress_multiplier,
            liquidity_role="maker",
            fee_type=schedule.fee_type,
        )
        breakdown = accumulator.apply_fill(
            revenue=-(intent.limit_price * incremental),
            trade_fee=fee,
        )
        fills.append(
            SimulatedFill(
                recv_ts_ns=event.recv_ts_ns,
                price=intent.limit_price,
                quantity=incremental,
                liquidity_role="maker",
                trade_fee=breakdown.trade_fee,
                rounding_fee=breakdown.rounding_fee,
                rebate=breakdown.rebate,
                net_fee=breakdown.net_fee,
            )
        )
        filled += incremental
        if filled >= intent.quantity:
            break
    return tuple(fills), best_bid, intent.quantity - filled


def _rejected(
    intent: ResearchOrderIntent,
    arrival: int,
    *,
    reason: str,
    settlement_recv_ts_ns: int | None = None,
) -> ExecutedOrder:
    return ExecutedOrder(
        intent_id=intent.intent_id,
        market_ticker=intent.market_ticker,
        attribution=intent.attribution,
        outcome_side=intent.outcome_side,
        liquidity_role=intent.liquidity_role,
        decision_recv_ts_ns=intent.decision_recv_ts_ns,
        arrival_ts_ns=arrival,
        settlement_recv_ts_ns=settlement_recv_ts_ns,
        requested_quantity=intent.quantity,
        filled_quantity=Decimal("0"),
        unfilled_quantity=intent.quantity,
        status="rejected",
        reason=reason,
        fills=(),
        entry_notional=Decimal("0"),
        trade_fees=Decimal("0"),
        rounding_fees=Decimal("0"),
        rebates=Decimal("0"),
        net_entry_fees=Decimal("0"),
        settlement_fee=Decimal("0"),
        payout=Decimal("0"),
        gross_pnl=Decimal("0"),
        net_pnl=Decimal("0"),
        top_price_at_arrival=None,
        average_fill_price=None,
        slippage=Decimal("0"),
        capital_locked=Decimal("0"),
    )


def _summarize(policy: ExecutionPolicy, orders: tuple[ExecutedOrder, ...]) -> ExecutionReplayReport:
    gross = sum((order.gross_pnl for order in orders), Decimal("0"))
    net = sum((order.net_pnl for order in orders), Decimal("0"))
    trade_fees = sum((order.trade_fees for order in orders), Decimal("0"))
    rounding = sum((order.rounding_fees for order in orders), Decimal("0"))
    rebates = sum((order.rebates for order in orders), Decimal("0"))
    settlement_fees = sum((order.settlement_fee for order in orders), Decimal("0"))
    slippage = sum((order.slippage for order in orders), Decimal("0"))

    locks: list[tuple[int, int, Decimal]] = []
    for order in orders:
        if order.status == "rejected" or order.settlement_recv_ts_ns is None:
            continue
        locks.append((order.arrival_ts_ns, 1, order.capital_locked))
        locks.append((order.settlement_recv_ts_ns, -1, order.capital_locked))
    locked = Decimal("0")
    max_locked = Decimal("0")
    for _, direction, amount in sorted(locks, key=lambda item: (item[0], item[1])):
        locked += amount if direction == 1 else -amount
        max_locked = max(max_locked, locked)

    cumulative = Decimal("0")
    peak = Decimal("0")
    max_drawdown = Decimal("0")
    settled_orders = sorted(
        (order for order in orders if order.status != "rejected" and order.settlement_recv_ts_ns is not None),
        key=lambda order: (order.settlement_recv_ts_ns, order.intent_id),
    )
    for order in settled_orders:
        cumulative += order.net_pnl
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)

    reasons = Counter(order.reason for order in orders if order.reason is not None)
    by_attr: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for order in orders:
        by_attr[order.attribution] += order.net_pnl
    return ExecutionReplayReport(
        policy=policy,
        orders=orders,
        gross_pnl=gross,
        net_pnl=net,
        trade_fees=trade_fees,
        rounding_fees=rounding,
        rebates=rebates,
        settlement_fees=settlement_fees,
        slippage=slippage,
        max_drawdown=max_drawdown,
        max_capital_locked=max_locked,
        filled_orders=sum(order.status == "filled" for order in orders),
        partial_orders=sum(order.status == "partial" for order in orders),
        rejected_orders=sum(order.status == "rejected" for order in orders),
        rejection_reasons=tuple(sorted((str(reason), count) for reason, count in reasons.items())),
        pnl_by_attribution=tuple(sorted(by_attr.items())),
    )
