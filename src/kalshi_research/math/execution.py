from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Sequence


CENT = Decimal("0.01")
CENTICENT = Decimal("0.0001")
GENERAL_TAKER_BASE_COEFFICIENT = Decimal("0.07")
GENERAL_MAKER_BASE_COEFFICIENT = Decimal("0.0175")


@dataclass(frozen=True, slots=True)
class FillEstimate:
    requested_qty: float
    filled_qty: float
    notional: float
    average_price: float | None
    unfilled_qty: float


@dataclass(frozen=True, slots=True)
class DecimalFillEstimate:
    requested_qty: Decimal
    filled_qty: Decimal
    notional: Decimal
    average_price: Decimal | None
    unfilled_qty: Decimal
    levels_consumed: tuple[tuple[Decimal, Decimal], ...]


@dataclass(frozen=True, slots=True)
class FeeBreakdown:
    trade_fee: Decimal
    rounding_fee: Decimal
    rebate: Decimal
    net_fee: Decimal
    balance_change: Decimal
    posted_balance_change: Decimal
    accumulator_after: Decimal


@dataclass(frozen=True, slots=True)
class SettlementRounding:
    raw_payout: Decimal
    posted_payout: Decimal
    settlement_fee: Decimal


def walk_asks(asks: Sequence[tuple[float, float]], quantity: float) -> FillEstimate:
    """Walk visible asks for a taker buy; levels are (price, available_qty)."""
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    remaining = quantity
    notional = 0.0
    filled = 0.0
    for price, size in sorted(asks, key=lambda x: x[0]):
        if not 0 <= price <= 1 or size < 0:
            raise ValueError("invalid binary order-book level")
        take = min(remaining, size)
        notional += take * price
        filled += take
        remaining -= take
        if remaining <= 1e-12:
            break
    return FillEstimate(
        requested_qty=quantity,
        filled_qty=filled,
        notional=notional,
        average_price=(notional / filled if filled else None),
        unfilled_qty=max(0.0, quantity - filled),
    )


def walk_asks_decimal(
    asks: Sequence[tuple[Decimal, Decimal]],
    quantity: Decimal,
) -> DecimalFillEstimate:
    """Walk visible fixed-point asks without binary floating-point error."""
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    remaining = quantity
    notional = Decimal("0")
    consumed: list[tuple[Decimal, Decimal]] = []
    for price, size in sorted(asks, key=lambda item: item[0]):
        if price < 0 or price > 1 or size < 0:
            raise ValueError("invalid binary order-book level")
        if remaining <= 0:
            break
        take = min(remaining, size)
        if take <= 0:
            continue
        consumed.append((price, take))
        notional += price * take
        remaining -= take
    filled = quantity - remaining
    return DecimalFillEstimate(
        requested_qty=quantity,
        filled_qty=filled,
        notional=notional,
        average_price=(notional / filled if filled > 0 else None),
        unfilled_qty=remaining,
        levels_consumed=tuple(consumed),
    )


def _ceil_increment(value: Decimal, increment: Decimal) -> Decimal:
    if value < 0 or increment <= 0:
        raise ValueError("fee rounding inputs must be nonnegative with positive increment")
    if value == 0:
        return Decimal("0")
    units = (value / increment).to_integral_value(rounding=ROUND_CEILING)
    return units * increment


def _floor_increment(value: Decimal, increment: Decimal) -> Decimal:
    if increment <= 0:
        raise ValueError("rounding increment must be positive")
    units = (value / increment).to_integral_value(rounding=ROUND_FLOOR)
    return units * increment


def quadratic_trade_fee(
    *,
    price: Decimal,
    quantity: Decimal,
    fee_multiplier: Decimal,
    liquidity_role: str,
    fee_type: str,
    taker_base_coefficient: Decimal = GENERAL_TAKER_BASE_COEFFICIENT,
    maker_base_coefficient: Decimal = GENERAL_MAKER_BASE_COEFFICIENT,
) -> Decimal:
    """Return the exchange trade-fee component, rounded to a centicent.

    The public series API supplies `fee_type` and `fee_multiplier`. The general
    regulatory fee schedule supplies the base quadratic coefficients. The
    coefficients are explicit arguments so research can version/stress them rather
    than silently assume that today's schedule applies forever.
    """
    if price < 0 or price > 1:
        raise ValueError("price must be between 0 and 1")
    if quantity <= 0 or fee_multiplier <= 0:
        raise ValueError("quantity and fee multiplier must be positive")
    if liquidity_role not in {"taker", "maker"}:
        raise ValueError("liquidity_role must be taker or maker")
    if fee_type not in {"quadratic", "quadratic_with_maker_fees", "flat"}:
        raise ValueError(f"unsupported fee type:{fee_type}")
    if fee_type == "flat":
        raise ValueError("flat fee schedules require an explicit flat-fee model")

    if liquidity_role == "taker":
        coefficient = taker_base_coefficient * fee_multiplier
    elif fee_type == "quadratic_with_maker_fees":
        coefficient = maker_base_coefficient * fee_multiplier
    else:
        coefficient = Decimal("0")

    raw = coefficient * quantity * price * (Decimal("1") - price)
    return _ceil_increment(raw, CENTICENT)


class FeeAccumulator:
    """Per-order Kalshi fixed-point fee-rounding accumulator.

    It mirrors the documented balance mechanics: trade fee is already centicent
    rounded; `revenue - trade_fee` is floored to cents; the sub-cent difference is
    accumulated; a one-cent rebate is issued only when both the accumulator and
    current fill fee can support it, so net fee never becomes negative.
    """

    def __init__(self) -> None:
        self.rounding_overpayment = Decimal("0")

    def apply_fill(self, *, revenue: Decimal, trade_fee: Decimal) -> FeeBreakdown:
        if trade_fee < 0:
            raise ValueError("trade_fee cannot be negative")
        balance_change = revenue - trade_fee
        posted = _floor_increment(balance_change, CENT)
        rounding_fee = balance_change - posted
        if rounding_fee < 0 or rounding_fee >= CENT:
            raise AssertionError("cent alignment produced invalid rounding fee")

        self.rounding_overpayment += rounding_fee
        rebate = Decimal("0")
        current_pre_rebate = trade_fee + rounding_fee
        if self.rounding_overpayment >= CENT and current_pre_rebate >= CENT:
            rebate = CENT
            self.rounding_overpayment -= CENT

        net_fee = current_pre_rebate - rebate
        if net_fee < 0:
            raise AssertionError("fee accumulator produced a negative net fee")
        return FeeBreakdown(
            trade_fee=trade_fee,
            rounding_fee=rounding_fee,
            rebate=rebate,
            net_fee=net_fee,
            balance_change=balance_change,
            posted_balance_change=posted,
            accumulator_after=self.rounding_overpayment,
        )


def settlement_rounding(quantity: Decimal, payout_per_contract: Decimal) -> SettlementRounding:
    """Round positive settlement payouts down to whole cents as documented."""
    if quantity < 0 or payout_per_contract < 0:
        raise ValueError("settlement inputs cannot be negative")
    raw = quantity * payout_per_contract
    posted = _floor_increment(raw, CENT)
    fee = raw - posted
    return SettlementRounding(raw_payout=raw, posted_payout=posted, settlement_fee=fee)


def conservative_maker_fill_qty(
    queue_ahead: float,
    traded_at_or_through_price: float,
    cancellations_ahead_credit: float = 0.0,
    order_qty: float = 1.0,
) -> float:
    """Conservative queue model for research.

    Only volume beyond estimated queue ahead can fill us. Cancellation credit is
    explicit and defaults to zero so a backtest cannot silently assume favorable
    queue advancement.
    """
    if min(queue_ahead, traded_at_or_through_price, cancellations_ahead_credit, order_qty) < 0:
        raise ValueError("queue inputs cannot be negative")
    effective_ahead = max(0.0, queue_ahead - cancellations_ahead_credit)
    available_for_us = max(0.0, traded_at_or_through_price - effective_ahead)
    return min(order_qty, available_for_us)


def adverse_selection_markout(fill_price: float, future_fair_price: float, side: str) -> float:
    if side not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    return future_fair_price - fill_price if side == "buy" else fill_price - future_fair_price
