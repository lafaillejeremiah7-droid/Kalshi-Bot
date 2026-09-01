from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RawRecord:
    source: str
    recv_ts_ns: int
    connection_id: str
    payload: Any
    event_ts_ns: int | None = None
    sequence: int | None = None


class RawJsonlCapture:
    """Append-only raw capture with a chained integrity hash.

    Each row contains the previous row hash, making accidental mutation or row
    deletion detectable during research audits. Files rotate hourly by receive
    timestamp and are never opened in rewrite mode.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._last_hash_by_path: dict[Path, str] = {}

    def _path_for(self, record: RawRecord) -> Path:
        dt = datetime.fromtimestamp(record.recv_ts_ns / 1e9, tz=timezone.utc)
        path = self.root / record.source / dt.strftime("%Y/%m/%d") / f"{dt:%H}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def append(self, record: RawRecord) -> tuple[Path, str]:
        path = self._path_for(record)
        prev_hash = self._last_hash_by_path.get(path)
        if prev_hash is None and path.exists() and path.stat().st_size:
            # Recover the final chain hash without rewriting history.
            with path.open("rb") as fh:
                last = None
                for line in fh:
                    if line.strip():
                        last = line
            if last:
                prev_hash = json.loads(last)["record_hash"]
        prev_hash = prev_hash or "0" * 64

        body = {
            "source": record.source,
            "recv_ts_ns": record.recv_ts_ns,
            "event_ts_ns": record.event_ts_ns,
            "connection_id": record.connection_id,
            "sequence": record.sequence,
            "payload": record.payload,
            "prev_hash": prev_hash,
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        record_hash = hashlib.sha256((prev_hash + canonical).encode()).hexdigest()
        row = {**body, "record_hash": record_hash}
        encoded = (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()

        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, encoded)
            os.fsync(fd)
        finally:
            os.close(fd)
        self._last_hash_by_path[path] = record_hash
        return path, record_hash


def verify_hash_chain(path: str | Path) -> bool:
    previous = "0" * 64
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            claimed = row.pop("record_hash")
            if row.get("prev_hash") != previous:
                return False
            canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            expected = hashlib.sha256((previous + canonical).encode()).hexdigest()
            if claimed != expected:
                return False
            previous = claimed
    return True
