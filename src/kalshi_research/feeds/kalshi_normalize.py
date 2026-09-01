from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationError

from kalshi_research.domain.events import (
    IndexTickEvent,
    OrderbookDeltaEvent,
    OrderbookSnapshotEvent,
    PriceLevel,
    ResearchEvent,
    Source,
    TradeEvent,
)
from kalshi_research.feeds.kalshi_ws import SequenceGap


class KalshiSchemaError(ValueError):
    """Known Kalshi message type did not match the expected research schema."""


class UnknownKalshiMessage(KalshiSchemaError):
    """Message type is outside the explicitly supported research schema."""


class _StrictModel(BaseModel):
    # Primitive fields remain strict through Strict* types, while JSON arrays may
    # still validate into tuple-shaped price levels.
    model_config = ConfigDict(extra="forbid")


class _OrderbookSnapshotMsg(_StrictModel):
    market_ticker: StrictStr
    market_id: StrictStr
    yes_dollars_fp: list[tuple[StrictStr, StrictStr]] = Field(default_factory=list)
    no_dollars_fp: list[tuple[StrictStr, StrictStr]] = Field(default_factory=list)


class _OrderbookSnapshot(_StrictModel):
    type: Literal["orderbook_snapshot"]
    sid: StrictInt = Field(ge=1)
    seq: StrictInt = Field(ge=1)
    msg: _OrderbookSnapshotMsg


class _OrderbookDeltaMsg(_StrictModel):
    market_ticker: StrictStr
    market_id: StrictStr
    price_dollars: StrictStr
    delta_fp: StrictStr
    side: Literal["yes", "no"]
    client_order_id: StrictStr | None = None
    subaccount: StrictInt | None = None
    ts: StrictStr | None = None
    ts_ms: StrictInt | None = None


class _OrderbookDelta(_StrictModel):
    type: Literal["orderbook_delta"]
    sid: StrictInt = Field(ge=1)
    seq: StrictInt = Field(ge=1)
    msg: _OrderbookDeltaMsg


class _TradeMsg(_StrictModel):
    trade_id: StrictStr
    market_ticker: StrictStr
    yes_price_dollars: StrictStr
    no_price_dollars: StrictStr
    count_fp: StrictStr
    taker_side: Literal["yes", "no"]
    taker_outcome_side: Literal["yes", "no"] | None = None
    taker_book_side: Literal["bid", "ask"] | None = None
    is_block_trade: StrictBool | None = None
    ts: StrictInt | None = None
    ts_ms: StrictInt


class _Trade(_StrictModel):
    type: Literal["trade"]
    sid: StrictInt = Field(ge=1)
    msg: _TradeMsg


class _AverageData(_StrictModel):
    value: StrictStr
    window_size: StrictInt = Field(ge=0)
    window_start_ts_ms: StrictInt
    window_end_ts_exclusive: StrictInt


class _CFBenchmarksMsg(_StrictModel):
    index_id: StrictStr
    received_at: StrictInt
    data: StrictStr
    avg_60s_data: _AverageData
    last_60s_windowed_average_15min: _AverageData | None = None


class _CFBenchmarksValue(_StrictModel):
    type: Literal["cfbenchmarks_value"]
    sid: StrictInt = Field(ge=1)
    seq: StrictInt = Field(ge=1)
    msg: _CFBenchmarksMsg


_CONTROL_TYPES = {
    "subscribed",
    "unsubscribed",
    "ok",
    "error",
    "cfbenchmarks_value_indexlist",
}


def _decimal(value: str, field: str, *, positive: bool = False) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise KalshiSchemaError(f"{field} is not a valid decimal") from exc
    if not parsed.is_finite():
        raise KalshiSchemaError(f"{field} must be finite")
    if positive and parsed <= 0:
        raise KalshiSchemaError(f"{field} must be positive")
    return parsed


def _price(value: str, field: str) -> Decimal:
    parsed = _decimal(value, field)
    if not Decimal("0") <= parsed <= Decimal("1"):
        raise KalshiSchemaError(f"{field} must be within [0, 1]")
    return parsed


def _level(pair: tuple[str, str]) -> PriceLevel:
    price, size = pair
    qty = _decimal(size, "orderbook level size")
    if qty < 0:
        raise KalshiSchemaError("orderbook level size cannot be negative")
    return PriceLevel(price=_price(price, "orderbook level price"), size=qty)


def _decode(raw: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise KalshiSchemaError("websocket frame is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise KalshiSchemaError("websocket frame must decode to an object")
    return payload


def _source_value(frame: str) -> tuple[int, Decimal]:
    try:
        data = json.loads(frame)
    except json.JSONDecodeError as exc:
        raise KalshiSchemaError("CF Benchmarks data field is not valid JSON") from exc
    if not isinstance(data, dict):
        raise KalshiSchemaError("CF Benchmarks data field must decode to an object")
    ts = data.get("time")
    value = data.get("value")
    if isinstance(ts, bool) or not isinstance(ts, int):
        raise KalshiSchemaError("CF Benchmarks upstream time must be integer milliseconds")
    if not isinstance(value, str):
        raise KalshiSchemaError("CF Benchmarks upstream value must be a decimal string")
    return ts, _decimal(value, "CF Benchmarks value", positive=True)


def normalize_kalshi_message(
    raw: str | bytes | dict[str, Any], *, recv_ts_ns: int
) -> ResearchEvent | None:
    """Normalize one Kalshi frame; control messages return None.

    Known data messages are validated against explicit schemas. Unknown or
    malformed frames raise so the capture runner can quarantine them.
    """
    payload = _decode(raw)
    msg_type = payload.get("type")
    if not isinstance(msg_type, str):
        raise KalshiSchemaError("Kalshi websocket message has no string type")
    if msg_type in _CONTROL_TYPES:
        return None

    try:
        if msg_type == "orderbook_snapshot":
            parsed = _OrderbookSnapshot.model_validate(payload)
            return OrderbookSnapshotEvent(
                source=Source.KALSHI,
                event_ts_ns=recv_ts_ns,
                recv_ts_ns=recv_ts_ns,
                market_ticker=parsed.msg.market_ticker,
                market_id=parsed.msg.market_id,
                sid=parsed.sid,
                seq=parsed.seq,
                yes_bids=tuple(_level(level) for level in parsed.msg.yes_dollars_fp),
                no_bids=tuple(_level(level) for level in parsed.msg.no_dollars_fp),
            )

        if msg_type == "orderbook_delta":
            parsed = _OrderbookDelta.model_validate(payload)
            event_ts_ns = parsed.msg.ts_ms * 1_000_000 if parsed.msg.ts_ms else recv_ts_ns
            return OrderbookDeltaEvent(
                source=Source.KALSHI,
                event_ts_ns=event_ts_ns,
                recv_ts_ns=recv_ts_ns,
                market_ticker=parsed.msg.market_ticker,
                market_id=parsed.msg.market_id,
                sid=parsed.sid,
                seq=parsed.seq,
                side=parsed.msg.side,
                price=_price(parsed.msg.price_dollars, "orderbook delta price"),
                delta=_decimal(parsed.msg.delta_fp, "orderbook delta size"),
            )

        if msg_type == "trade":
            parsed = _Trade.model_validate(payload)
            return TradeEvent(
                source=Source.KALSHI,
                event_ts_ns=parsed.msg.ts_ms * 1_000_000,
                recv_ts_ns=recv_ts_ns,
                market_ticker=parsed.msg.market_ticker,
                trade_id=parsed.msg.trade_id,
                price=_price(parsed.msg.yes_price_dollars, "trade YES price"),
                no_price=_price(parsed.msg.no_price_dollars, "trade NO price"),
                size=_decimal(parsed.msg.count_fp, "trade count", positive=True),
                taker_side=parsed.msg.taker_side,
                taker_book_side=parsed.msg.taker_book_side,
                is_block_trade=parsed.msg.is_block_trade,
            )

        if msg_type == "cfbenchmarks_value":
            parsed = _CFBenchmarksValue.model_validate(payload)
            if parsed.msg.index_id != "BRTI":
                raise UnknownKalshiMessage(
                    f"CF Benchmarks index {parsed.msg.index_id!r} is outside BTC15m scope"
                )
            upstream_ms, value = _source_value(parsed.msg.data)
            trailing = parsed.msg.avg_60s_data
            final = parsed.msg.last_60s_windowed_average_15min
            if final is not None and final.window_size > 60:
                raise KalshiSchemaError("final-minute BRTI sample count cannot exceed 60")
            return IndexTickEvent(
                source=Source.BRTI,
                event_ts_ns=upstream_ms * 1_000_000,
                recv_ts_ns=recv_ts_ns,
                index_id=parsed.msg.index_id,
                sid=parsed.sid,
                seq=parsed.seq,
                value=value,
                kalshi_received_ts_ns=parsed.msg.received_at * 1_000_000,
                trailing_60s_average=_decimal(
                    trailing.value, "BRTI trailing 60s average", positive=True
                ),
                trailing_60s_sample_count=trailing.window_size,
                final_minute_average=(
                    _decimal(final.value, "BRTI final-minute average", positive=True)
                    if final
                    else None
                ),
                final_minute_sample_count=final.window_size if final else None,
                final_minute_window_start_ts_ns=(
                    final.window_start_ts_ms * 1_000_000 if final else None
                ),
                final_minute_window_end_ts_ns=(
                    final.window_end_ts_exclusive * 1_000_000 if final else None
                ),
            )
    except ValidationError as exc:
        raise KalshiSchemaError(f"invalid {msg_type} payload: {exc}") from exc

    raise UnknownKalshiMessage(f"unsupported Kalshi websocket message type: {msg_type}")


class SubscriptionSequenceGuard:
    """Fail-closed sequence validation keyed by Kalshi subscription id."""

    def __init__(self) -> None:
        self._last_seq: dict[int, int] = {}
        self._book_ready: set[int] = set()

    def observe(self, event: ResearchEvent) -> None:
        if isinstance(event, OrderbookSnapshotEvent):
            if event.sid is None:
                raise SequenceGap("orderbook snapshot is missing subscription id")
            self._last_seq[event.sid] = event.seq
            self._book_ready.add(event.sid)
            return
        if isinstance(event, OrderbookDeltaEvent):
            if event.sid is None:
                raise SequenceGap("orderbook delta is missing subscription id")
            if event.sid not in self._book_ready:
                raise SequenceGap(f"delta for sid {event.sid} received before snapshot")
            self._expect_next(event.sid, event.seq)
            return
        if isinstance(event, IndexTickEvent) and event.sid is not None and event.seq is not None:
            if event.sid not in self._last_seq:
                self._last_seq[event.sid] = event.seq
            else:
                self._expect_next(event.sid, event.seq)

    def _expect_next(self, sid: int, seq: int) -> None:
        expected = self._last_seq[sid] + 1
        if seq != expected:
            raise SequenceGap(f"sid {sid}: expected seq {expected}, got {seq}")
        self._last_seq[sid] = seq
