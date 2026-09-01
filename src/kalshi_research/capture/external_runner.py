from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Literal

import websockets

from kalshi_research.config import ResearchConfig
from kalshi_research.feeds.external import (
    COINBASE_WS_URL,
    KRAKEN_WS_URL,
    ExternalFeedError,
    KrakenBook,
    coinbase_subscription_messages,
    kraken_subscription_message,
    normalize_coinbase_ticker,
    quote_to_spot_event,
)
from kalshi_research.storage.raw_capture import RawJsonlCapture, RawRecord
from kalshi_research.storage.sqlite_store import SqliteEventStore


Venue = Literal["coinbase", "kraken"]


def _raw_payload(raw: str | bytes) -> str | dict[str, str]:
    if isinstance(raw, str):
        return raw
    return {"binary_b64": base64.b64encode(raw).decode("ascii")}


def _quarantine(
    capture: RawJsonlCapture,
    *,
    venue: Venue,
    recv_ts_ns: int,
    connection_id: str,
    raw: str | bytes,
    exc: Exception,
) -> None:
    capture.append(
        RawRecord(
            source=f"{venue}_ws_quarantine",
            recv_ts_ns=recv_ts_ns,
            connection_id=connection_id,
            payload={
                "exception": type(exc).__name__,
                "reason": str(exc),
                "raw": _raw_payload(raw),
            },
        )
    )


async def run_external_capture(
    config: ResearchConfig,
    *,
    venue: Venue,
    max_messages: int | None = None,
) -> int:
    """Capture public BTC spot data into the research event store.

    Raw frames are persisted before parsing. Coinbase ticker frames are
    self-contained, so malformed frames are quarantined and collection can
    continue. Kraken's L2 feed is stateful; any malformed book frame terminates
    capture after quarantine because continuing could silently build on a
    missing mutation.
    """
    if venue not in {"coinbase", "kraken"}:
        raise ValueError("venue must be coinbase or kraken")
    if max_messages is not None and max_messages <= 0:
        raise ValueError("max_messages must be positive when supplied")

    config.ensure_research_dirs()
    connection_id = uuid.uuid4().hex
    raw_capture = RawJsonlCapture(config.raw_capture_dir / "external")
    quarantine_capture = RawJsonlCapture(config.raw_capture_dir / "external_quarantine")
    processed = 0

    if venue == "coinbase":
        url = COINBASE_WS_URL
        subscriptions = coinbase_subscription_messages()
        kraken_book = None
    else:
        url = KRAKEN_WS_URL
        subscriptions = (kraken_subscription_message(),)
        kraken_book = KrakenBook()

    with SqliteEventStore(config.research_db_path) as store:
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_queue=4096,
        ) as websocket:
            for message in subscriptions:
                await websocket.send(json.dumps(message, separators=(",", ":")))

            async for raw in websocket:
                recv_ts_ns = time.time_ns()
                raw_capture.append(
                    RawRecord(
                        source=f"{venue}_ws",
                        recv_ts_ns=recv_ts_ns,
                        connection_id=connection_id,
                        payload=_raw_payload(raw),
                    )
                )

                try:
                    if venue == "coinbase":
                        quotes = normalize_coinbase_ticker(raw, recv_ts_ns=recv_ts_ns)
                    else:
                        assert kraken_book is not None
                        quotes = kraken_book.apply(raw, recv_ts_ns=recv_ts_ns)
                    for quote in quotes:
                        store.append(quote_to_spot_event(quote))
                except ExternalFeedError as exc:
                    _quarantine(
                        quarantine_capture,
                        venue=venue,
                        recv_ts_ns=recv_ts_ns,
                        connection_id=connection_id,
                        raw=raw,
                        exc=exc,
                    )
                    if venue == "kraken":
                        raise

                processed += 1
                if max_messages is not None and processed >= max_messages:
                    break

    return processed
