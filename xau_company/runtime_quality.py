from __future__ import annotations

from math import ceil
from typing import Any

import pandas as pd


RESOLUTION_PREFERENCE: tuple[tuple[str, int], ...] = (("1min", 1), ("5min", 5))


def resolution_output_size(max_age_hours: int, interval_minutes: int, context_output_size: int) -> int:
    bars = ceil(max(1, int(max_age_hours)) * 60 / max(1, int(interval_minutes))) + 24
    return min(5000, max(int(context_output_size), bars))


def fetch_resolution_history(
    market: Any,
    quality: Any,
    symbol: str,
    max_age_hours: int,
    context_output_size: int,
    now,
) -> tuple[pd.DataFrame | None, int | None, str | None]:
    """Prefer completed 1m outcome bars, safely fall back to completed 5m bars."""
    for interval, minutes in RESOLUTION_PREFERENCE:
        raw = market.safe_candles(
            symbol,
            interval,
            resolution_output_size(max_age_hours, minutes, context_output_size),
        )
        if raw is None or raw.empty:
            continue
        cleaned, report = quality.clean_frame(raw, interval, now=now, require_fresh=True)
        if report.ok and cleaned is not None and not cleaned.empty:
            return cleaned, minutes, interval
    return None, None, None


def revalidate_optional_macro(quality: Any, frame: pd.DataFrame | None, interval: str, now) -> pd.DataFrame | None:
    """Reject cached macro context as soon as it becomes stale between refreshes."""
    if frame is None or frame.empty:
        return None
    cleaned, report = quality.clean_frame(frame, interval, now=now, require_fresh=True)
    return cleaned if report.ok else None
