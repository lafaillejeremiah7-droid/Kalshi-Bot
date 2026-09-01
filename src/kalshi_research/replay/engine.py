from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from kalshi_research.domain.events import OrderbookDeltaEvent, OrderbookSnapshotEvent, ResearchEvent
from kalshi_research.feeds.kalshi_ws import BinaryOrderBook


@dataclass(slots=True)
class ReplayState:
    books: dict[str, BinaryOrderBook]
    processed_events: int = 0


class ReplayEngine:
    """Deterministic event-time replay. No wall-clock sleeps, no future peeking."""

    def __init__(self) -> None:
        self.state = ReplayState(books={})

    def run(self, events: Iterable[ResearchEvent], on_event: Callable[[ResearchEvent, ReplayState], None] | None = None) -> ReplayState:
        for event in events:
            self._apply(event)
            self.state.processed_events += 1
            if on_event:
                on_event(event, self.state)
        return self.state

    def _apply(self, event: ResearchEvent) -> None:
        if isinstance(event, OrderbookSnapshotEvent):
            book = self.state.books.setdefault(event.market_ticker, BinaryOrderBook(event.market_ticker))
            book.apply_snapshot(event.seq, [[str(level.price), str(level.size)] for level in event.yes_bids], [[str(level.price), str(level.size)] for level in event.no_bids])
        elif isinstance(event, OrderbookDeltaEvent):
            book = self.state.books.setdefault(event.market_ticker, BinaryOrderBook(event.market_ticker))
            book.apply_delta(event.seq, event.side, str(event.price), str(event.delta))
