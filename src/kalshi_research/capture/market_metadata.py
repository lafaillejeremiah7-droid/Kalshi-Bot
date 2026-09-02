from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from kalshi_research.config import ResearchConfig
from kalshi_research.domain.events import EventKind, MarketEvent, SettlementEvent
from kalshi_research.feeds.kalshi_rest import KalshiRestClient, market_target, parse_decimal
from kalshi_research.storage.raw_capture import RawJsonlCapture, RawRecord
from kalshi_research.storage.sqlite_store import SqliteEventStore


class MarketMetadataSchemaError(ValueError):
    """Raised when public market metadata cannot be normalized safely."""


def parse_timestamp_ns(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000.0
        if number <= 0:
            return None
        return int(number * 1_000_000_000)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            number = None
        if number is not None:
            return parse_timestamp_ns(number)
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        timestamp = dt.timestamp()
        if timestamp <= 0:
            return None
        return int(timestamp * 1_000_000_000)
    return None


def _required_timestamp(market: dict[str, Any], fields: tuple[str, ...], name: str) -> int:
    for field in fields:
        parsed = parse_timestamp_ns(market.get(field))
        if parsed is not None:
            return parsed
    raise MarketMetadataSchemaError(f"market metadata missing usable {name}")


def normalize_market_metadata(
    market: dict[str, Any],
    *,
    series_ticker: str,
    observed_ts_ns: int,
) -> tuple[MarketEvent, SettlementEvent | None]:
    ticker = market.get("ticker")
    event_ticker = market.get("event_ticker")
    status = market.get("status")
    if not isinstance(ticker, str) or not ticker:
        raise MarketMetadataSchemaError("market metadata missing ticker")
    if not ticker.startswith(series_ticker):
        raise MarketMetadataSchemaError(
            f"unexpected market ticker {ticker!r} for series {series_ticker!r}"
        )
    if not isinstance(event_ticker, str) or not event_ticker:
        raise MarketMetadataSchemaError("market metadata missing event_ticker")
    if not isinstance(status, str) or not status:
        raise MarketMetadataSchemaError("market metadata missing status")

    try:
        target = market_target(market)
    except (ValueError, ArithmeticError) as exc:
        raise MarketMetadataSchemaError(str(exc)) from exc

    open_ts_ns = _required_timestamp(
        market,
        ("open_time", "created_time"),
        "open time",
    )
    close_ts_ns = _required_timestamp(
        market,
        (
            "close_time",
            "expiration_time",
            "expected_expiration_time",
            "latest_expiration_time",
        ),
        "close time",
    )
    if close_ts_ns <= open_ts_ns:
        raise MarketMetadataSchemaError("market close time must be after open time")

    raw_result = market.get("result")
    result: str | None = None
    if raw_result not in (None, ""):
        normalized_result = str(raw_result).lower()
        if normalized_result not in {"yes", "no"}:
            raise MarketMetadataSchemaError(
                f"unexpected binary KXBTC15M result {raw_result!r}"
            )
        result = normalized_result

    settlement_value = parse_decimal(market.get("settlement_value_dollars"))
    if settlement_value is None:
        # Retain compatibility with older captured fixtures and earlier API names.
        settlement_value = parse_decimal(market.get("settlement_value"))

    market_event = MarketEvent(
        event_ts_ns=observed_ts_ns,
        recv_ts_ns=observed_ts_ns,
        market_ticker=ticker,
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        target_price=target,
        open_ts_ns=open_ts_ns,
        close_ts_ns=close_ts_ns,
        status=status,
        result=result,
        settlement_value=settlement_value,
    )

    settlement_event: SettlementEvent | None = None
    if result is not None and settlement_value is not None:
        settlement_ts_ns = parse_timestamp_ns(market.get("settlement_ts")) or observed_ts_ns
        # The public REST payload can be observed long after settlement; event time
        # preserves the exchange's settlement timestamp while recv time is the
        # actual research information frontier.
        settlement_event = SettlementEvent(
            event_ts_ns=settlement_ts_ns,
            recv_ts_ns=observed_ts_ns,
            market_ticker=ticker,
            target_price=target,
            final_value=Decimal(settlement_value),
            result=result,
        )

    return market_event, settlement_event


def capture_market_metadata(
    config: ResearchConfig,
    market_ticker: str,
    *,
    store: SqliteEventStore | None = None,
    raw_capture: RawJsonlCapture | None = None,
    connection_id: str | None = None,
) -> bool:
    """Persist one public market metadata observation and any official settlement.

    Returns True only when a complete binary settlement label was captured.
    Raw REST evidence is always written before normalization when a raw capture
    object is supplied.
    """
    observed_ts_ns = time.time_ns()
    market = KalshiRestClient(config.kalshi_rest_base).get_market(market_ticker)
    capture = raw_capture or RawJsonlCapture(config.raw_capture_dir)
    capture.append(
        RawRecord(
            source="kalshi_rest_market_metadata",
            recv_ts_ns=observed_ts_ns,
            connection_id=connection_id or uuid.uuid4().hex,
            payload=market,
        )
    )
    market_event, settlement_event = normalize_market_metadata(
        market,
        series_ticker=config.kalshi_series_ticker,
        observed_ts_ns=observed_ts_ns,
    )

    if store is not None:
        store.append(market_event)
        if settlement_event is not None:
            store.append(settlement_event)
    else:
        with SqliteEventStore(config.research_db_path) as local_store:
            local_store.append(market_event)
            if settlement_event is not None:
                local_store.append(settlement_event)
    return settlement_event is not None


def pending_settlement_tickers(
    store: SqliteEventStore,
    *,
    now_ns: int | None = None,
) -> tuple[str, ...]:
    """Find captured contracts that have closed but lack a settlement event."""
    current_ns = time.time_ns() if now_ns is None else now_ns
    closed_markets: set[str] = set()
    for event in store.iter_events_by_kind(EventKind.MARKET, order_by="receive"):
        if isinstance(event, MarketEvent) and event.close_ts_ns <= current_ns:
            closed_markets.add(event.market_ticker)

    settled = {
        event.market_ticker
        for event in store.iter_events_by_kind(EventKind.SETTLEMENT, order_by="receive")
        if isinstance(event, SettlementEvent)
    }
    return tuple(sorted(closed_markets - settled))
