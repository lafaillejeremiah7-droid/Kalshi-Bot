from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class FillEstimate:
    requested_qty: float
    filled_qty: float
    notional: float
    average_price: float | None
    unfilled_qty: float


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
