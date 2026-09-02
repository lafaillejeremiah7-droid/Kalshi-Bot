from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kalshi_research.research.runner import (
    ResearchRunReport,
    research_report_digest,
    research_report_json,
)


class ReportArchiveError(RuntimeError):
    """Raised when immutable experiment evidence cannot be trusted."""


@dataclass(frozen=True, slots=True)
class ArchivedResearchReport:
    digest: str
    path: Path
    series_ticker: str
    plan_digest: str
    events_digest: str
    market_count: int
    event_count: int


class ExperimentReportArchive:
    """Content-addressed, append-only archive for deterministic research reports.

    Reports are stored under their SHA-256 digest. Publishing an identical report is
    idempotent. Existing content is never overwritten; any filename/content mismatch,
    malformed payload, or report that claims trading authority is treated as archive
    corruption and fails closed.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def publish(self, report: ResearchRunReport) -> ArchivedResearchReport:
        payload = research_report_json(report)
        digest = research_report_digest(report)
        if hashlib.sha256(payload.encode()).hexdigest() != digest:
            raise ReportArchiveError("report digest mismatch before archive write")

        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path_for_digest(digest)
        if target.exists():
            entry = self._verify_path(target)
            if target.read_text(encoding="utf-8") != payload:
                raise ReportArchiveError(f"archive content mismatch for digest {digest}")
            return entry

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{digest}.",
            suffix=".tmp",
            dir=self.root,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temp_path, target)
            except FileExistsError:
                # A concurrent identical publisher won the race. Never overwrite it.
                pass
            self._fsync_directory()
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

        entry = self._verify_path(target)
        if target.read_text(encoding="utf-8") != payload:
            raise ReportArchiveError(f"archive content mismatch for digest {digest}")
        return entry

    def get(self, digest: str) -> ArchivedResearchReport:
        return self._verify_path(self._path_for_digest(digest))

    def read_payload(self, digest: str) -> dict[str, Any]:
        entry = self.get(digest)
        return self._parse_payload(entry.path)

    def list(self) -> tuple[ArchivedResearchReport, ...]:
        if not self.root.exists():
            return ()
        if not self.root.is_dir():
            raise ReportArchiveError(f"archive root is not a directory:{self.root}")
        entries = [self._verify_path(path) for path in sorted(self.root.glob("*.json"))]
        return tuple(entries)

    def _path_for_digest(self, digest: str) -> Path:
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ReportArchiveError("report digest must be lowercase SHA-256 hex")
        return self.root / f"{digest}.json"

    def _verify_path(self, path: Path) -> ArchivedResearchReport:
        if not path.exists():
            raise ReportArchiveError(f"archived report not found:{path.stem}")
        if not path.is_file() or path.suffix != ".json":
            raise ReportArchiveError(f"invalid archive entry:{path.name}")

        digest = path.stem
        self._path_for_digest(digest)
        content = path.read_text(encoding="utf-8")
        actual = hashlib.sha256(content.encode()).hexdigest()
        if actual != digest:
            raise ReportArchiveError(
                f"archive digest mismatch:{path.name}:actual={actual}"
            )
        payload = self._parse_payload(path, content=content)
        if payload.get("mode") != "research_only" or payload.get("order_placement") is not False:
            raise ReportArchiveError(f"archive contains non-research report:{path.name}")

        markets = payload.get("markets")
        if not isinstance(markets, list):
            raise ReportArchiveError(f"archive markets payload is invalid:{path.name}")
        return ArchivedResearchReport(
            digest=digest,
            path=path,
            series_ticker=self._required_string(payload, "series_ticker", path),
            plan_digest=self._required_digest(payload, "plan_digest", path),
            events_digest=self._required_digest(payload, "events_digest", path),
            market_count=len(markets),
            event_count=self._required_nonnegative_int(payload, "event_count", path),
        )

    @staticmethod
    def _parse_payload(path: Path, *, content: str | None = None) -> dict[str, Any]:
        try:
            value = json.loads(content if content is not None else path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReportArchiveError(f"invalid archived JSON:{path.name}") from exc
        if not isinstance(value, dict):
            raise ReportArchiveError(f"archived report is not an object:{path.name}")
        return value

    @staticmethod
    def _required_string(payload: dict[str, Any], key: str, path: Path) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ReportArchiveError(f"archive field {key} is invalid:{path.name}")
        return value

    @classmethod
    def _required_digest(cls, payload: dict[str, Any], key: str, path: Path) -> str:
        value = cls._required_string(payload, key, path)
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ReportArchiveError(f"archive field {key} is not SHA-256:{path.name}")
        return value

    @staticmethod
    def _required_nonnegative_int(payload: dict[str, Any], key: str, path: Path) -> int:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ReportArchiveError(f"archive field {key} is invalid:{path.name}")
        return value

    def _fsync_directory(self) -> None:
        try:
            fd = os.open(self.root, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)
