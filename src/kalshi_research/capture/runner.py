from __future__ import annotations

import base64
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets

from kalshi_research.config import ResearchConfig
from kalshi_research.feeds.kalshi_normalize import (
    KalshiSchemaError,
    SubscriptionSequenceGuard,
    UnknownKalshiMessage,
    normalize_kalshi_message,
)
from kalshi_research.feeds.kalshi_rest import KalshiRestClient
from kalshi_research.feeds.kalshi_ws import SequenceGap, create_auth_headers
from kalshi_research.storage.raw_capture import RawJsonlCapture, RawRecord
from kalshi_research.storage.sqlite_store import SqliteEventStore


WS_PATH = "/trade-api/ws/v2"


def build_subscription_messages(market_ticker: str) -> list[dict[str, Any]]:
    """Build isolated research subscriptions.

    The orderbook gets its own subscription id so its sequence stream cannot be
    confused with another market. No user-order or trading channel is included.
    """
    if not market_ticker or not market_ticker.startswith("KXBTC15M"):
        raise ValueError("capture ticker must be a KXBTC15M market")
    return [
        {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": [market_ticker],
            },
        },
        {
            "id": 2,
            "cmd": "subscribe",
            "params": {"channels": ["trade"], "market_tickers": [market_ticker]},
        },
        {
            "id": 3,
            "cmd": "subscribe",
            "params": {"channels": ["cfbenchmarks_value"], "index_ids": ["BRTI"]},
        },
    ]


def _parse_close_time(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000.0
        return int(number * 1_000_000_000)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
    return None


def _market_close_ns(market: dict[str, Any]) -> int | None:
    for field in ("close_time", "expiration_time", "expected_expiration_time"):
        parsed = _parse_close_time(market.get(field))
        if parsed is not None:
            return parsed
    return None


def discover_open_btc15m_market(config: ResearchConfig) -> str:
    client = KalshiRestClient(config.kalshi_rest_base)
    markets = client.get_markets(series_ticker=config.kalshi_series_ticker, status="open")
    valid = [m for m in markets if isinstance(m.get("ticker"), str)]
    if not valid:
        raise RuntimeError("no open KXBTC15M market was returned by Kalshi")
    if len(valid) == 1:
        return str(valid[0]["ticker"])

    now_ns = time.time_ns()
    candidates: list[tuple[int, str]] = []
    for market in valid:
        close_ns = _market_close_ns(market)
        ticker = str(market["ticker"])
        if close_ns is not None and close_ns >= now_ns:
            candidates.append((close_ns, ticker))
    if not candidates:
        raise RuntimeError("multiple open KXBTC15M markets returned without usable future close times")
    candidates.sort()
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise RuntimeError("ambiguous open KXBTC15M markets share the nearest close time")
    return candidates[0][1]


def _raw_payload(raw: str | bytes) -> Any:
    if isinstance(raw, str):
        return raw
    return {"binary_b64": base64.b64encode(raw).decode("ascii")}


def _quarantine(
    capture: RawJsonlCapture,
    *,
    recv_ts_ns: int,
    connection_id: str,
    raw: str | bytes,
    exc: Exception,
) -> None:
    capture.append(
        RawRecord(
            source="kalshi_ws_quarantine",
            recv_ts_ns=recv_ts_ns,
            connection_id=connection_id,
            payload={
                "exception": type(exc).__name__,
                "reason": str(exc),
                "raw": _raw_payload(raw),
            },
        )
    )


def _require_credentials(config: ResearchConfig) -> tuple[str, Path]:
    if not config.kalshi_api_key_id:
        raise RuntimeError("KALSHI_API_KEY_ID is required for authenticated market-data capture")
    if config.kalshi_private_key_path is None:
        raise RuntimeError("KALSHI_PRIVATE_KEY_PATH is required for authenticated market-data capture")
    path = config.kalshi_private_key_path
    if not path.is_file():
        raise RuntimeError(f"Kalshi private key file does not exist: {path}")
    return config.kalshi_api_key_id, path


async def run_kalshi_capture(
    config: ResearchConfig,
    *,
    market_ticker: str,
    max_messages: int | None = None,
) -> int:
    """Capture raw + normalized KXBTC15M data without order-placement authority."""
    if max_messages is not None and max_messages <= 0:
        raise ValueError("max_messages must be positive when supplied")
    config.ensure_research_dirs()
    api_key_id, private_key_path = _require_credentials(config)
    headers = create_auth_headers(api_key_id, private_key_path, "GET", WS_PATH)
    connection_id = uuid.uuid4().hex
    raw_capture = RawJsonlCapture(config.raw_capture_dir)
    quarantine_capture = RawJsonlCapture(config.raw_capture_dir / "quarantine")
    sequence_guard = SubscriptionSequenceGuard()
    processed = 0

    with SqliteEventStore(config.research_db_path) as store:
        async with websockets.connect(
            config.kalshi_ws_url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_queue=4096,
        ) as websocket:
            for message in build_subscription_messages(market_ticker):
                await websocket.send(json.dumps(message, separators=(",", ":")))

            async for raw in websocket:
                recv_ts_ns = time.time_ns()
                raw_capture.append(
                    RawRecord(
                        source="kalshi_ws",
                        recv_ts_ns=recv_ts_ns,
                        connection_id=connection_id,
                        payload=_raw_payload(raw),
                    )
                )

                try:
                    event = normalize_kalshi_message(raw, recv_ts_ns=recv_ts_ns)
                    if event is not None:
                        sequence_guard.observe(event)
                        store.append(event)
                except SequenceGap as exc:
                    _quarantine(
                        quarantine_capture,
                        recv_ts_ns=recv_ts_ns,
                        connection_id=connection_id,
                        raw=raw,
                        exc=exc,
                    )
                    raise
                except (KalshiSchemaError, UnknownKalshiMessage) as exc:
                    _quarantine(
                        quarantine_capture,
                        recv_ts_ns=recv_ts_ns,
                        connection_id=connection_id,
                        raw=raw,
                        exc=exc,
                    )

                processed += 1
                if max_messages is not None and processed >= max_messages:
                    break

    return processed
