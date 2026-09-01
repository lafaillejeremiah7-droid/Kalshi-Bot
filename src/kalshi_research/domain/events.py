from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Source(StrEnum):
    KALSHI = "kalshi"
    BRTI = "brti"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    FUTURES = "futures"
    SYSTEM = "system"


class EventKind(StrEnum):
    MARKET = "market"
    ORDERBOOK_SNAPSHOT = "orderbook_snapshot"
    ORDERBOOK_DELTA = "orderbook_delta"
    TRADE = "trade"
    INDEX_TICK = "index_tick"
    SPOT_TICK = "spot_tick"
    SETTLEMENT = "settlement"
    HEALTH = "health"


class BaseEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: EventKind
    source: Source
    event_ts_ns: int = Field(gt=0)
    recv_ts_ns: int = Field(gt=0)
    market_ticker: str | None = None

    @field_validator("recv_ts_ns")
    @classmethod
    def recv_not_absurdly_before_event(cls, value: int, info):
        event_ts = info.data.get("event_ts_ns")
        if event_ts and value + 5_000_000_000 < event_ts:
            raise ValueError("receive timestamp is more than 5s before source event timestamp")
        return value

    @property
    def latency_ms(self) -> float:
        return (self.recv_ts_ns - self.event_ts_ns) / 1_000_000


class MarketEvent(BaseEvent):
    kind: Literal[EventKind.MARKET] = EventKind.MARKET
    source: Literal[Source.KALSHI] = Source.KALSHI
    market_ticker: str
    event_ticker: str
    series_ticker: str
    target_price: Decimal
    open_ts_ns: int
    close_ts_ns: int
    status: str
    result: str | None = None
    settlement_value: Decimal | None = None


class PriceLevel(BaseModel):
    model_config = ConfigDict(frozen=True)
    price: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    size: Decimal = Field(ge=Decimal("0"))


class OrderbookSnapshotEvent(BaseEvent):
    kind: Literal[EventKind.ORDERBOOK_SNAPSHOT] = EventKind.ORDERBOOK_SNAPSHOT
    source: Literal[Source.KALSHI] = Source.KALSHI
    market_ticker: str
    sid: int | None = Field(default=None, ge=1)
    market_id: str | None = None
    seq: int = Field(ge=0)
    yes_bids: tuple[PriceLevel, ...] = ()
    no_bids: tuple[PriceLevel, ...] = ()


class OrderbookDeltaEvent(BaseEvent):
    kind: Literal[EventKind.ORDERBOOK_DELTA] = EventKind.ORDERBOOK_DELTA
    source: Literal[Source.KALSHI] = Source.KALSHI
    market_ticker: str
    sid: int | None = Field(default=None, ge=1)
    market_id: str | None = None
    seq: int = Field(ge=0)
    side: Literal["yes", "no"]
    price: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    delta: Decimal


class TradeEvent(BaseEvent):
    kind: Literal[EventKind.TRADE] = EventKind.TRADE
    market_ticker: str
    trade_id: str | None = None
    price: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    no_price: Decimal | None = Field(default=None, ge=Decimal("0"), le=Decimal("1"))
    size: Decimal = Field(gt=Decimal("0"))
    taker_side: str | None = None
    taker_book_side: str | None = None
    is_block_trade: bool | None = None


class IndexTickEvent(BaseEvent):
    kind: Literal[EventKind.INDEX_TICK] = EventKind.INDEX_TICK
    source: Literal[Source.BRTI] = Source.BRTI
    index_id: str = "BRTI"
    sid: int | None = Field(default=None, ge=1)
    seq: int | None = Field(default=None, ge=0)
    value: Decimal = Field(gt=Decimal("0"))
    kalshi_received_ts_ns: int | None = Field(default=None, gt=0)
    trailing_60s_average: Decimal | None = Field(default=None, gt=Decimal("0"))
    trailing_60s_sample_count: int | None = Field(default=None, ge=0)
    final_minute_average: Decimal | None = Field(default=None, gt=Decimal("0"))
    final_minute_sample_count: int | None = Field(default=None, ge=0, le=60)
    final_minute_window_start_ts_ns: int | None = Field(default=None, gt=0)
    final_minute_window_end_ts_ns: int | None = Field(default=None, gt=0)


class SpotTickEvent(BaseEvent):
    kind: Literal[EventKind.SPOT_TICK] = EventKind.SPOT_TICK
    source: Source
    venue: str
    symbol: str = "BTC-USD"
    bid: Decimal | None = None
    ask: Decimal | None = None
    last: Decimal | None = None


class SettlementEvent(BaseEvent):
    kind: Literal[EventKind.SETTLEMENT] = EventKind.SETTLEMENT
    source: Literal[Source.KALSHI] = Source.KALSHI
    market_ticker: str
    target_price: Decimal
    final_value: Decimal
    result: Literal["yes", "no"]


class HealthEvent(BaseEvent):
    kind: Literal[EventKind.HEALTH] = EventKind.HEALTH
    source: Literal[Source.SYSTEM] = Source.SYSTEM
    component: str
    status: Literal["ok", "degraded", "failed"]
    detail: str = ""


ResearchEvent = Annotated[
    Union[
        MarketEvent,
        OrderbookSnapshotEvent,
        OrderbookDeltaEvent,
        TradeEvent,
        IndexTickEvent,
        SpotTickEvent,
        SettlementEvent,
        HealthEvent,
    ],
    Field(discriminator="kind"),
]


def utc_now_ns() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1_000_000_000)
