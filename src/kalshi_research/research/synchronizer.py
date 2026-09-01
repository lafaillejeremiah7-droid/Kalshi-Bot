from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from kalshi_research.domain.events import (
    IndexTickEvent,
    MarketEvent,
    OrderbookDeltaEvent,
    OrderbookSnapshotEvent,
    ResearchEvent,
    Source,
    SpotTickEvent,
)
from kalshi_research.feeds.kalshi_ws import BinaryOrderBook


class SynchronizationError(RuntimeError):
    """Raised when receive-time invariants would be violated."""


@dataclass(frozen=True, slots=True)
class FreshnessPolicy:
    kalshi_book_ns: int = 750_000_000
    brti_ns: int = 2_500_000_000
    coinbase_ns: int = 1_500_000_000
    kraken_ns: int = 1_500_000_000
    market_metadata_ns: int = 3_600_000_000_000

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class SourceState:
    recv_ts_ns: int | None
    age_ns: int | None
    fresh: bool


@dataclass(frozen=True, slots=True)
class SynchronizedFrame:
    decision_recv_ts_ns: int
    market_ticker: str
    target_price: Decimal | None
    close_ts_ns: int | None
    seconds_to_close: float | None

    yes_bid: Decimal | None
    yes_bid_size: Decimal | None
    yes_ask: Decimal | None
    yes_ask_size: Decimal | None
    yes_spread: Decimal | None

    brti: Decimal | None
    brti_final_minute_average: Decimal | None
    brti_final_minute_sample_count: int | None

    coinbase_mid: Decimal | None
    coinbase_bid_size: Decimal | None
    coinbase_ask_size: Decimal | None
    kraken_mid: Decimal | None
    kraken_bid_size: Decimal | None
    kraken_ask_size: Decimal | None

    market_state: SourceState
    kalshi_book_state: SourceState
    brti_state: SourceState
    coinbase_state: SourceState
    kraken_state: SourceState

    @property
    def any_external_fresh(self) -> bool:
        return self.coinbase_state.fresh or self.kraken_state.fresh

    @property
    def core_fresh(self) -> bool:
        return self.kalshi_book_state.fresh and self.brti_state.fresh

    @property
    def probability_ready(self) -> bool:
        return self.core_fresh and self.any_external_fresh and self.target_price is not None


@dataclass(slots=True)
class _MarketBookState:
    book: BinaryOrderBook
    recv_ts_ns: int


@dataclass(slots=True)
class ReceiveTimeSynchronizer:
    policy: FreshnessPolicy = field(default_factory=FreshnessPolicy)
    _clock_ns: int | None = None
    _markets: dict[str, MarketEvent] = field(default_factory=dict)
    _market_recv_ts: dict[str, int] = field(default_factory=dict)
    _books: dict[str, _MarketBookState] = field(default_factory=dict)
    _brti: IndexTickEvent | None = None
    _spots: dict[Source, SpotTickEvent] = field(default_factory=dict)

    @property
    def clock_ns(self) -> int | None:
        return self._clock_ns

    def ingest(self, event: ResearchEvent) -> None:
        if self._clock_ns is not None and event.recv_ts_ns < self._clock_ns:
            raise SynchronizationError(
                "events must be ingested in nondecreasing receive-time order"
            )
        self._clock_ns = event.recv_ts_ns

        if isinstance(event, MarketEvent):
            self._markets[event.market_ticker] = event
            self._market_recv_ts[event.market_ticker] = event.recv_ts_ns
            return

        if isinstance(event, OrderbookSnapshotEvent):
            book = BinaryOrderBook(ticker=event.market_ticker)
            book.yes = {level.price: level.size for level in event.yes_bids if level.size > 0}
            book.no = {level.price: level.size for level in event.no_bids if level.size > 0}
            book.last_seq = event.seq
            self._books[event.market_ticker] = _MarketBookState(book, event.recv_ts_ns)
            return

        if isinstance(event, OrderbookDeltaEvent):
            state = self._books.get(event.market_ticker)
            if state is None:
                raise SynchronizationError("Kalshi delta received before snapshot")
            state.book.apply_delta(
                event.seq,
                event.side,
                str(event.price),
                str(event.delta),
            )
            state.recv_ts_ns = event.recv_ts_ns
            return

        if isinstance(event, IndexTickEvent):
            self._brti = event
            return

        if isinstance(event, SpotTickEvent):
            if event.source not in (Source.COINBASE, Source.KRAKEN):
                return
            self._spots[event.source] = event

    def frame(
        self,
        market_ticker: str,
        *,
        decision_recv_ts_ns: int | None = None,
    ) -> SynchronizedFrame:
        if self._clock_ns is None:
            raise SynchronizationError("cannot materialize a frame before ingesting data")

        decision = self._clock_ns if decision_recv_ts_ns is None else decision_recv_ts_ns
        if decision < self._clock_ns:
            raise SynchronizationError(
                "decision time predates already-ingested state; rebuild from receive-time replay"
            )

        market = self._markets.get(market_ticker)
        book_state = self._books.get(market_ticker)
        coinbase = self._spots.get(Source.COINBASE)
        kraken = self._spots.get(Source.KRAKEN)

        yes_bid = yes_bid_size = yes_ask = yes_ask_size = yes_spread = None
        if book_state is not None:
            book = book_state.book
            yes_bid = book.best_yes_bid
            no_bid = book.best_no_bid
            yes_ask = book.implied_yes_ask
            if yes_bid is not None:
                yes_bid_size = book.yes.get(yes_bid)
            if no_bid is not None:
                yes_ask_size = book.no.get(no_bid)
            if yes_bid is not None and yes_ask is not None:
                yes_spread = yes_ask - yes_bid

        target_price = market.target_price if market is not None else None
        close_ts_ns = market.close_ts_ns if market is not None else None
        seconds_to_close = None
        if close_ts_ns is not None:
            seconds_to_close = max(0.0, (close_ts_ns - decision) / 1_000_000_000)

        brti = self._brti

        return SynchronizedFrame(
            decision_recv_ts_ns=decision,
            market_ticker=market_ticker,
            target_price=target_price,
            close_ts_ns=close_ts_ns,
            seconds_to_close=seconds_to_close,
            yes_bid=yes_bid,
            yes_bid_size=yes_bid_size,
            yes_ask=yes_ask,
            yes_ask_size=yes_ask_size,
            yes_spread=yes_spread,
            brti=brti.value if brti is not None else None,
            brti_final_minute_average=(
                brti.final_minute_average if brti is not None else None
            ),
            brti_final_minute_sample_count=(
                brti.final_minute_sample_count if brti is not None else None
            ),
            coinbase_mid=_mid(coinbase),
            coinbase_bid_size=coinbase.bid_size if coinbase is not None else None,
            coinbase_ask_size=coinbase.ask_size if coinbase is not None else None,
            kraken_mid=_mid(kraken),
            kraken_bid_size=kraken.bid_size if kraken is not None else None,
            kraken_ask_size=kraken.ask_size if kraken is not None else None,
            market_state=_source_state(
                decision,
                self._market_recv_ts.get(market_ticker),
                self.policy.market_metadata_ns,
            ),
            kalshi_book_state=_source_state(
                decision,
                book_state.recv_ts_ns if book_state is not None else None,
                self.policy.kalshi_book_ns,
            ),
            brti_state=_source_state(
                decision,
                brti.recv_ts_ns if brti is not None else None,
                self.policy.brti_ns,
            ),
            coinbase_state=_source_state(
                decision,
                coinbase.recv_ts_ns if coinbase is not None else None,
                self.policy.coinbase_ns,
            ),
            kraken_state=_source_state(
                decision,
                kraken.recv_ts_ns if kraken is not None else None,
                self.policy.kraken_ns,
            ),
        )


def _mid(event: SpotTickEvent | None) -> Decimal | None:
    if event is None or event.bid is None or event.ask is None:
        return None
    return (event.bid + event.ask) / Decimal("2")


def _source_state(decision_ns: int, recv_ns: int | None, max_age_ns: int) -> SourceState:
    if recv_ns is None:
        return SourceState(recv_ts_ns=None, age_ns=None, fresh=False)
    age = decision_ns - recv_ns
    if age < 0:
        raise SynchronizationError("source receive timestamp is in the future")
    return SourceState(recv_ts_ns=recv_ns, age_ns=age, fresh=age <= max_age_ns)
