from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

import websockets


COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"
KRAKEN_WS_URL = "wss://ws.kraken.com/v2"


class ExternalFeedError(ValueError):
    """External market-data frame is malformed or violates a book invariant."""


@dataclass(frozen=True, slots=True)
class ExternalQuote:
    venue: str
    event_ts_ns: int
    recv_ts_ns: int
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    symbol: str = "BTC-USD"
    last: Decimal | None = None
    source_sequence: int | None = None
    checksum: int | None = None

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal("2")

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid


class ExternalMarketFeed(Protocol):
    async def quotes(self) -> AsyncIterator[ExternalQuote]:  # pragma: no cover - protocol
        ...


_RFC3339_RE = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?"
    r"(?P<zone>Z|[+-]\d{2}:\d{2})$"
)


def rfc3339_to_ns(value: str) -> int:
    """Parse RFC3339 without discarding sub-microsecond exchange timestamps."""
    if not isinstance(value, str):
        raise ExternalFeedError("timestamp must be a string")
    match = _RFC3339_RE.fullmatch(value)
    if match is None:
        raise ExternalFeedError(f"invalid RFC3339 timestamp: {value!r}")
    zone = "+00:00" if match.group("zone") == "Z" else match.group("zone")
    try:
        base = datetime.fromisoformat(match.group("base") + zone)
    except ValueError as exc:
        raise ExternalFeedError(f"invalid RFC3339 timestamp: {value!r}") from exc
    epoch_seconds = int(base.timestamp())
    fraction = match.group("fraction") or ""
    fraction_ns = int(fraction.ljust(9, "0")) if fraction else 0
    return epoch_seconds * 1_000_000_000 + fraction_ns


def _decode(raw: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw, parse_float=Decimal)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ExternalFeedError("external websocket frame is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ExternalFeedError("external websocket frame must be a JSON object")
    return value


def _decimal(value: Any, field: str, *, positive: bool = False, nonnegative: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ExternalFeedError(f"{field} must be numeric")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ExternalFeedError(f"{field} is not a valid decimal") from exc
    if not parsed.is_finite():
        raise ExternalFeedError(f"{field} must be finite")
    if positive and parsed <= 0:
        raise ExternalFeedError(f"{field} must be positive")
    if nonnegative and parsed < 0:
        raise ExternalFeedError(f"{field} must be nonnegative")
    return parsed


def _quote(
    *,
    venue: str,
    symbol: str,
    event_ts_ns: int,
    recv_ts_ns: int,
    bid: Any,
    ask: Any,
    bid_size: Any,
    ask_size: Any,
    last: Any | None = None,
    source_sequence: int | None = None,
    checksum: int | None = None,
) -> ExternalQuote:
    bid_d = _decimal(bid, f"{venue} bid", positive=True)
    ask_d = _decimal(ask, f"{venue} ask", positive=True)
    bid_size_d = _decimal(bid_size, f"{venue} bid size", positive=True)
    ask_size_d = _decimal(ask_size, f"{venue} ask size", positive=True)
    if bid_d >= ask_d:
        raise ExternalFeedError(f"{venue} book is crossed or locked: {bid_d} >= {ask_d}")
    last_d = _decimal(last, f"{venue} last", positive=True) if last is not None else None
    return ExternalQuote(
        venue=venue,
        event_ts_ns=event_ts_ns,
        recv_ts_ns=recv_ts_ns,
        bid=bid_d,
        ask=ask_d,
        bid_size=bid_size_d,
        ask_size=ask_size_d,
        symbol=symbol,
        last=last_d,
        source_sequence=source_sequence,
        checksum=checksum,
    )


def coinbase_subscription_messages() -> tuple[dict[str, Any], ...]:
    return (
        {"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "ticker"},
        {"type": "subscribe", "channel": "heartbeats"},
    )


def normalize_coinbase_ticker(
    raw: str | bytes | dict[str, Any], *, recv_ts_ns: int
) -> tuple[ExternalQuote, ...]:
    payload = _decode(raw)
    channel = payload.get("channel")
    if channel in {"heartbeats", "subscriptions"}:
        return ()
    if channel != "ticker":
        raise ExternalFeedError(f"unsupported Coinbase channel: {channel!r}")

    timestamp = payload.get("timestamp")
    sequence = payload.get("sequence_num")
    events = payload.get("events")
    if not isinstance(timestamp, str):
        raise ExternalFeedError("Coinbase ticker timestamp is missing")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise ExternalFeedError("Coinbase sequence_num must be an integer")
    if not isinstance(events, list):
        raise ExternalFeedError("Coinbase ticker events must be a list")
    event_ts_ns = rfc3339_to_ns(timestamp)

    quotes: list[ExternalQuote] = []
    for event in events:
        if not isinstance(event, dict):
            raise ExternalFeedError("Coinbase ticker event must be an object")
        tickers = event.get("tickers")
        if not isinstance(tickers, list):
            raise ExternalFeedError("Coinbase ticker event is missing tickers")
        for ticker in tickers:
            if not isinstance(ticker, dict):
                raise ExternalFeedError("Coinbase ticker entry must be an object")
            product_id = ticker.get("product_id")
            if product_id != "BTC-USD":
                continue
            required = (
                "price",
                "best_bid",
                "best_bid_quantity",
                "best_ask",
                "best_ask_quantity",
            )
            if any(not isinstance(ticker.get(field), str) for field in required):
                raise ExternalFeedError("Coinbase BTC-USD ticker is missing decimal string fields")
            quotes.append(
                _quote(
                    venue="coinbase",
                    symbol="BTC-USD",
                    event_ts_ns=event_ts_ns,
                    recv_ts_ns=recv_ts_ns,
                    bid=ticker["best_bid"],
                    ask=ticker["best_ask"],
                    bid_size=ticker["best_bid_quantity"],
                    ask_size=ticker["best_ask_quantity"],
                    last=ticker["price"],
                    source_sequence=sequence,
                )
            )
    return tuple(quotes)


class CoinbaseTickerFeed:
    """Public Coinbase Advanced Trade BTC-USD ticker feed."""

    def __init__(self, url: str = COINBASE_WS_URL) -> None:
        self.url = url

    async def quotes(self) -> AsyncIterator[ExternalQuote]:
        async with websockets.connect(
            self.url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_queue=4096,
        ) as websocket:
            for message in coinbase_subscription_messages():
                await websocket.send(json.dumps(message, separators=(",", ":")))
            async for raw in websocket:
                recv_ts_ns = time.time_ns()
                for quote in normalize_coinbase_ticker(raw, recv_ts_ns=recv_ts_ns):
                    yield quote


def kraken_subscription_message() -> dict[str, Any]:
    return {
        "method": "subscribe",
        "params": {
            "channel": "book",
            "symbol": ["BTC/USD"],
            "depth": 10,
            "snapshot": True,
        },
    }


class KrakenBook:
    """Stateful Kraken Spot WebSocket v2 BTC/USD level-2 book.

    Kraken includes a CRC32 checksum for the top ten levels. The checksum is
    retained on emitted quotes for audit/replay, but this class intentionally
    does not claim verification until the exchange-specific canonical string
    algorithm is implemented and independently tested.
    """

    def __init__(self) -> None:
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self.ready = False

    def apply(
        self, raw: str | bytes | dict[str, Any], *, recv_ts_ns: int
    ) -> tuple[ExternalQuote, ...]:
        payload = _decode(raw)
        channel = payload.get("channel")
        if channel in {"heartbeat", "status"}:
            return ()
        if payload.get("method") == "subscribe":
            if payload.get("success") is False:
                raise ExternalFeedError(f"Kraken subscription failed: {payload.get('error')!r}")
            return ()
        if channel != "book":
            raise ExternalFeedError(f"unsupported Kraken channel: {channel!r}")
        message_type = payload.get("type")
        if message_type not in {"snapshot", "update"}:
            raise ExternalFeedError(f"unsupported Kraken book message type: {message_type!r}")
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise ExternalFeedError("Kraken book message must contain data")

        quotes: list[ExternalQuote] = []
        for book in data:
            if not isinstance(book, dict):
                raise ExternalFeedError("Kraken book data entry must be an object")
            if book.get("symbol") != "BTC/USD":
                continue
            timestamp = book.get("timestamp")
            if not isinstance(timestamp, str):
                raise ExternalFeedError("Kraken book timestamp is missing")
            if message_type == "snapshot":
                self.bids.clear()
                self.asks.clear()
                self._apply_levels(self.bids, book.get("bids", []), "bid")
                self._apply_levels(self.asks, book.get("asks", []), "ask")
                self.ready = True
            else:
                if not self.ready:
                    raise ExternalFeedError("Kraken book update received before snapshot")
                self._apply_levels(self.bids, book.get("bids", []), "bid")
                self._apply_levels(self.asks, book.get("asks", []), "ask")

            if not self.bids or not self.asks:
                raise ExternalFeedError("Kraken book has no two-sided top of book")
            best_bid = max(self.bids)
            best_ask = min(self.asks)
            checksum = book.get("checksum")
            if isinstance(checksum, bool) or (checksum is not None and not isinstance(checksum, int)):
                raise ExternalFeedError("Kraken checksum must be an integer when present")
            quotes.append(
                _quote(
                    venue="kraken",
                    symbol="BTC/USD",
                    event_ts_ns=rfc3339_to_ns(timestamp),
                    recv_ts_ns=recv_ts_ns,
                    bid=best_bid,
                    ask=best_ask,
                    bid_size=self.bids[best_bid],
                    ask_size=self.asks[best_ask],
                    checksum=checksum,
                )
            )
        return tuple(quotes)

    @staticmethod
    def _apply_levels(
        side: dict[Decimal, Decimal], levels: Any, side_name: str
    ) -> None:
        if levels is None:
            return
        if not isinstance(levels, list):
            raise ExternalFeedError(f"Kraken {side_name} levels must be a list")
        for level in levels:
            if not isinstance(level, dict):
                raise ExternalFeedError(f"Kraken {side_name} level must be an object")
            if "price" not in level or "qty" not in level:
                raise ExternalFeedError(f"Kraken {side_name} level is missing price/qty")
            price = _decimal(level["price"], f"Kraken {side_name} price", positive=True)
            qty = _decimal(level["qty"], f"Kraken {side_name} qty", nonnegative=True)
            if qty == 0:
                side.pop(price, None)
            else:
                side[price] = qty


class KrakenBookFeed:
    """Public Kraken Spot WebSocket v2 BTC/USD level-2 feed."""

    def __init__(self, url: str = KRAKEN_WS_URL) -> None:
        self.url = url
        self.book = KrakenBook()

    async def quotes(self) -> AsyncIterator[ExternalQuote]:
        async with websockets.connect(
            self.url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_queue=4096,
        ) as websocket:
            await websocket.send(json.dumps(kraken_subscription_message(), separators=(",", ":")))
            async for raw in websocket:
                recv_ts_ns = time.time_ns()
                for quote in self.book.apply(raw, recv_ts_ns=recv_ts_ns):
                    yield quote
