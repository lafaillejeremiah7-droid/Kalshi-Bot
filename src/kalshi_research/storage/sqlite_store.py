from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterable

from pydantic import TypeAdapter

from kalshi_research.domain.events import EventKind, ResearchEvent


EVENT_ADAPTER = TypeAdapter(ResearchEvent)


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uid TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    market_ticker TEXT,
    event_ts_ns INTEGER NOT NULL,
    recv_ts_ns INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_market_time ON events(market_ticker, event_ts_ns, id);
CREATE INDEX IF NOT EXISTS idx_events_kind_time ON events(kind, event_ts_ns, id);
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SqliteEventStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SqliteEventStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @staticmethod
    def _serialize(event: ResearchEvent) -> tuple[str, str]:
        dumped = event.model_dump(mode="json")
        payload = json.dumps(dumped, sort_keys=True, separators=(",", ":"))
        uid = hashlib.sha256(payload.encode()).hexdigest()
        return uid, payload

    def append(self, event: ResearchEvent) -> bool:
        uid, payload = self._serialize(event)
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO events(event_uid, kind, source, market_ticker, event_ts_ns, recv_ts_ns, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uid,
                str(event.kind),
                str(event.source),
                event.market_ticker,
                event.event_ts_ns,
                event.recv_ts_ns,
                payload,
            ),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def append_many(self, events: Iterable[ResearchEvent]) -> int:
        rows = []
        for event in events:
            uid, payload = self._serialize(event)
            rows.append(
                (
                    uid,
                    str(event.kind),
                    str(event.source),
                    event.market_ticker,
                    event.event_ts_ns,
                    event.recv_ts_ns,
                    payload,
                )
            )
        before = self.conn.total_changes
        with self.conn:
            self.conn.executemany(
                """
                INSERT OR IGNORE INTO events(event_uid, kind, source, market_ticker, event_ts_ns, recv_ts_ns, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return self.conn.total_changes - before

    @staticmethod
    def _order_clause(order_by: str) -> str:
        if order_by == "receive":
            return " ORDER BY recv_ts_ns ASC, event_ts_ns ASC, id ASC"
        if order_by == "event":
            return " ORDER BY event_ts_ns ASC, recv_ts_ns ASC, id ASC"
        raise ValueError("order_by must be receive or event")

    def iter_events(self, market_ticker: str | None = None, *, order_by: str = "receive"):
        query = "SELECT payload_json FROM events"
        args: tuple[object, ...] = ()
        if market_ticker:
            query += " WHERE market_ticker = ?"
            args = (market_ticker,)
        query += self._order_clause(order_by)
        for (payload,) in self.conn.execute(query, args):
            yield EVENT_ADAPTER.validate_json(payload)

    def iter_events_by_kind(
        self,
        kind: EventKind | str,
        *,
        order_by: str = "receive",
    ):
        """Iterate one canonical event kind using the existing kind/time index."""
        query = "SELECT payload_json FROM events WHERE kind = ?"
        query += self._order_clause(order_by)
        for (payload,) in self.conn.execute(query, (str(kind),)):
            yield EVENT_ADAPTER.validate_json(payload)
