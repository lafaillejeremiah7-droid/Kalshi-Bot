from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable

from kalshi_research.domain.events import FeeScheduleEvent, Source


class FeeScheduleError(RuntimeError):
    """Raised when a fee schedule cannot be reconstructed without guessing."""


def parse_exchange_timestamp_ns(value: Any) -> int:
    if isinstance(value, bool) or value in (None, ""):
        raise FeeScheduleError("fee schedule timestamp is missing")
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000.0
        return int(number * 1_000_000_000)
    if isinstance(value, str):
        text = value.strip()
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise FeeScheduleError(f"invalid fee schedule timestamp:{value}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
    raise FeeScheduleError(f"unsupported fee schedule timestamp:{type(value).__name__}")


def _fee_type(value: Any) -> str:
    text = str(value or "").strip()
    allowed = {"quadratic", "quadratic_with_maker_fees", "flat"}
    if text not in allowed:
        raise FeeScheduleError(f"unsupported fee_type:{text or '<missing>'}")
    return text


def _fee_multiplier(value: Any) -> Decimal:
    try:
        multiplier = Decimal(str(value))
    except Exception as exc:
        raise FeeScheduleError(f"invalid fee_multiplier:{value}") from exc
    if multiplier <= 0:
        raise FeeScheduleError("fee_multiplier must be positive")
    return multiplier


def normalize_series_fee_events(
    *,
    series: dict[str, Any],
    fee_changes: Iterable[dict[str, Any]],
    observed_ts_ns: int,
) -> tuple[FeeScheduleEvent, ...]:
    """Normalize public Kalshi fee metadata into canonical research events.

    Historical fee changes are safe for post-hoc transaction-cost accounting but are
    kept separate from predictive feature replay. The current series snapshot becomes
    effective at observation time unless a historical/scheduled change explicitly
    provides an exchange effective timestamp.
    """
    if observed_ts_ns <= 0:
        raise ValueError("observed_ts_ns must be positive")
    series_ticker = str(series.get("ticker") or "").strip()
    if not series_ticker:
        raise FeeScheduleError("series ticker missing from fee metadata")

    events: list[FeeScheduleEvent] = []
    for raw in fee_changes:
        if str(raw.get("series_ticker") or series_ticker) != series_ticker:
            raise FeeScheduleError("fee change belongs to a different series")
        effective_ts_ns = parse_exchange_timestamp_ns(raw.get("scheduled_ts"))
        events.append(
            FeeScheduleEvent(
                source=Source.KALSHI,
                event_ts_ns=observed_ts_ns,
                recv_ts_ns=observed_ts_ns,
                series_ticker=series_ticker,
                fee_change_id=(str(raw["id"]) if raw.get("id") not in (None, "") else None),
                fee_type=_fee_type(raw.get("fee_type")),
                fee_multiplier=_fee_multiplier(raw.get("fee_multiplier")),
                effective_ts_ns=effective_ts_ns,
                historical=effective_ts_ns <= observed_ts_ns,
            )
        )

    current = FeeScheduleEvent(
        source=Source.KALSHI,
        event_ts_ns=observed_ts_ns,
        recv_ts_ns=observed_ts_ns,
        series_ticker=series_ticker,
        fee_change_id=None,
        fee_type=_fee_type(series.get("fee_type")),
        fee_multiplier=_fee_multiplier(series.get("fee_multiplier")),
        effective_ts_ns=observed_ts_ns,
        historical=False,
    )
    events.append(current)
    events.sort(key=lambda event: (event.effective_ts_ns, event.recv_ts_ns, event.fee_change_id or ""))
    return tuple(events)


@dataclass(frozen=True, slots=True)
class FeeScheduleTimeline:
    events: tuple[FeeScheduleEvent, ...]

    @classmethod
    def from_events(
        cls,
        events: Iterable[FeeScheduleEvent],
        *,
        series_ticker: str,
    ) -> "FeeScheduleTimeline":
        filtered = tuple(
            sorted(
                (event for event in events if event.series_ticker == series_ticker),
                key=lambda event: (
                    event.effective_ts_ns,
                    event.recv_ts_ns,
                    event.fee_change_id or "",
                ),
            )
        )
        if not filtered:
            raise FeeScheduleError(f"no fee schedule events for {series_ticker}")
        return cls(filtered)

    def at(
        self,
        effective_ts_ns: int,
        *,
        knowledge_cutoff_ns: int | None = None,
        allow_posthoc_history: bool = True,
    ) -> FeeScheduleEvent:
        if effective_ts_ns <= 0:
            raise ValueError("effective_ts_ns must be positive")
        candidates = []
        for event in self.events:
            if event.effective_ts_ns > effective_ts_ns:
                continue
            if knowledge_cutoff_ns is not None and event.recv_ts_ns > knowledge_cutoff_ns:
                if not (allow_posthoc_history and event.historical):
                    continue
            candidates.append(event)
        if not candidates:
            raise FeeScheduleError(
                f"no known effective fee schedule at {effective_ts_ns}"
            )
        latest_effective = max(event.effective_ts_ns for event in candidates)
        same_effective = [event for event in candidates if event.effective_ts_ns == latest_effective]
        # Prefer an explicit exchange fee-change record over an observation-time
        # snapshot when both describe the same effective instant.
        same_effective.sort(
            key=lambda event: (
                event.fee_change_id is not None,
                event.recv_ts_ns,
            )
        )
        return same_effective[-1]
