from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from kalshi_research.domain.events import ResearchEvent


@dataclass(frozen=True, slots=True)
class QualityIssue:
    severity: str
    code: str
    detail: str


def validate_event_stream(events: Iterable[ResearchEvent]) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    last_source_ts: dict[str, int] = {}
    seen = 0
    for event in events:
        seen += 1
        key = f"{event.source}:{event.market_ticker or '-'}:{event.kind}"
        previous = last_source_ts.get(key)
        if previous is not None and event.event_ts_ns < previous:
            issues.append(QualityIssue("error", "TIME_REVERSAL", f"{key} moved backward in event time"))
        last_source_ts[key] = event.event_ts_ns
        latency_ms = event.latency_ms
        if latency_ms < -100:
            issues.append(QualityIssue("error", "NEGATIVE_LATENCY", f"{key} latency={latency_ms:.3f}ms"))
        elif latency_ms > 5_000:
            issues.append(QualityIssue("warning", "STALE_EVENT", f"{key} latency={latency_ms:.1f}ms"))
    if seen == 0:
        issues.append(QualityIssue("error", "EMPTY_STREAM", "no events supplied"))
    return issues
