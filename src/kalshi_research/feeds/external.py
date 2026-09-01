from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ExternalQuote:
    venue: str
    event_ts_ns: int
    recv_ts_ns: int
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal("2")


class ExternalMarketFeed(Protocol):
    """Protocol implemented later by Coinbase/Kraken/futures collectors."""

    async def quotes(self):  # pragma: no cover - protocol definition
        ...
