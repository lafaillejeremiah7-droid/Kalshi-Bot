from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from kalshi_research.domain.events import OrderbookDeltaEvent, OrderbookSnapshotEvent, ResearchEvent
from kalshi_research.feeds.kalshi_ws import BinaryOrderBook


@dataclass(slots=True)
class ReplayState:
    books: dict[str, BinaryOrderBook]
    processed_events: int = 0
    last_recv_ts_ns: int | None = None


class ReplayEngine:
    """Deterministic receive-time replay with no future peeking.

    Callers must supply events in nondecreasing ``recv_ts_ns`` order. This is
    deliberately stricter than event-time replay: a source timestamp that the
    collector had not received yet cannot become visible to a simulated model.
    """

    def __init__(self) -> None:
        self.state = ReplayState(books={})

    def run(
        self,
        events: Iterable[ResearchEvent],
        on_event: Callable[[ResearchEvent, ReplayState], None] | None = None,
    ) -> ReplayState:
        for event in events:
            if (
                self.state.last_recv_ts_ns is not None
                and event.recv_ts_ns < self.state.last_recv_ts_ns
            ):
                raise ValueError(
                    "replay events must be ordered by nondecreasing recv_ts_ns; "
                    "event-time ordering can introduce look-ahead"
                )
            self._apply(event)
            self.state.processed_events += 1
            self.state.last_recv_ts_ns = event.recv_ts_ns
            if on_event:
                on_event(event, self.state)
        return self.state

    def _apply(self, event: ResearchEvent) -> None:
        if isinstance(event, OrderbookSnapshotEvent):
            book = self.state.books.setdefault(
                event.market_ticker, BinaryOrderBook(event.market_ticker)
            )
            book.apply_snapshot(
                event.seq,
                [[str(level.price), str(level.size)] for level in event.yes_bids],
                [[str(level.price), str(level.size)] for level in event.no_bids],
            )
        elif isinstance(event, OrderbookDeltaEvent):
            book = self.state.books.setdefault(
                event.market_ticker, BinaryOrderBook(event.market_ticker)
            )
            book.apply_delta(event.seq, event.side, str(event.price), str(event.delta))
