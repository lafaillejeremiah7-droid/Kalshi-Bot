from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


INTERVAL_MINUTES = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "45min": 45,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "8h": 480,
    "1day": 1440,
}


@dataclass(frozen=True)
class FrameQuality:
    timeframe: str
    ok: bool
    reason: str
    rows: int
    latest_complete: pd.Timestamp | None = None


class MarketDataQualityAgent:
    """Validate, complete and freshness-check market frames before any bot uses them."""

    name = "Market Data Quality Guard"

    def __init__(
        self,
        max_stale_multiplier: float = 4.0,
        timezone_name: str = "America/Chicago",
    ) -> None:
        self.max_stale_multiplier = max(1.5, float(max_stale_multiplier))
        self.timezone = ZoneInfo(timezone_name)

    @staticmethod
    def interval_delta(interval: str) -> timedelta:
        minutes = INTERVAL_MINUTES.get(interval)
        if minutes is None:
            raise ValueError(f"Unsupported interval for quality checks: {interval}")
        return timedelta(minutes=minutes)

    def market_is_open(self, now: datetime | pd.Timestamp | None = None) -> bool:
        ts = pd.Timestamp(now or datetime.now(timezone.utc))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        local = ts.tz_convert(self.timezone)
        weekday = local.weekday()
        local_time = local.time().replace(tzinfo=None)

        if weekday == 5:
            return False
        if weekday == 6:
            return local_time >= time(17, 0)
        if weekday == 4 and local_time >= time(16, 0):
            return False
        # Conservative daily maintenance window used by many gold/FX venues.
        if weekday in {0, 1, 2, 3} and time(16, 0) <= local_time < time(17, 0):
            return False
        return True

    def clean_frame(
        self,
        df: pd.DataFrame,
        interval: str,
        now: datetime | pd.Timestamp | None = None,
        require_fresh: bool = True,
    ) -> tuple[pd.DataFrame, FrameQuality]:
        required = {"datetime", "open", "high", "low", "close"}
        if df is None or not required.issubset(df.columns):
            return pd.DataFrame(), FrameQuality(interval, False, "missing OHLC/datetime columns", 0)

        work = df.copy()
        work["datetime"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
        for col in ("open", "high", "low", "close"):
            work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work.dropna(subset=["datetime", "open", "high", "low", "close"])
        if work.empty:
            return work, FrameQuality(interval, False, "no valid OHLC rows", 0)

        if work["datetime"].duplicated().any():
            return pd.DataFrame(), FrameQuality(interval, False, "duplicate candle timestamps", len(work))

        work = work.sort_values("datetime").reset_index(drop=True)
        values = work[["open", "high", "low", "close"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            return pd.DataFrame(), FrameQuality(interval, False, "non-finite or non-positive prices", len(work))

        bad_geometry = (
            (work["high"] < work[["open", "close"]].max(axis=1))
            | (work["low"] > work[["open", "close"]].min(axis=1))
            | (work["high"] < work["low"])
        )
        if bool(bad_geometry.any()):
            return pd.DataFrame(), FrameQuality(interval, False, "invalid OHLC candle geometry", len(work))

        delta = self.interval_delta(interval)
        now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        else:
            now_ts = now_ts.tz_convert("UTC")

        # Twelve Data timestamps identify the candle start. Never let a still-forming
        # candle influence research, regime classification, votes or the final signal.
        complete_mask = work["datetime"] + delta <= now_ts
        work = work.loc[complete_mask].reset_index(drop=True)
        if work.empty:
            return work, FrameQuality(interval, False, "no completed candles", 0)

        latest_end = work["datetime"].iloc[-1] + delta
        if require_fresh and self.market_is_open(now_ts.to_pydatetime()):
            stale_limit = delta * self.max_stale_multiplier
            if now_ts - latest_end > stale_limit:
                return pd.DataFrame(), FrameQuality(
                    interval,
                    False,
                    f"stale completed candle ({now_ts - latest_end} old)",
                    len(work),
                    latest_end,
                )

        return work, FrameQuality(interval, True, "ok", len(work), latest_end)

    def clean_frames(
        self,
        frames: dict[str, pd.DataFrame],
        now: datetime | pd.Timestamp | None = None,
        required_context: tuple[str, ...] = ("15min", "1h", "4h"),
        execution_choices: tuple[str, ...] = ("1min", "5min"),
    ) -> tuple[dict[str, pd.DataFrame], list[FrameQuality]]:
        cleaned: dict[str, pd.DataFrame] = {}
        reports: list[FrameQuality] = []
        for interval, frame in frames.items():
            clean, report = self.clean_frame(frame, interval, now=now, require_fresh=True)
            reports.append(report)
            if report.ok:
                cleaned[interval] = clean

        for interval in required_context:
            if interval not in cleaned:
                reports.append(FrameQuality(interval, False, "required context timeframe unavailable", 0))
        if not any(interval in cleaned for interval in execution_choices):
            reports.append(FrameQuality("execution", False, "no fresh 1min/5min execution timeframe", 0))
        return cleaned, reports

    @staticmethod
    def required_frames_ok(reports: list[FrameQuality]) -> bool:
        return not any(
            (r.timeframe in {"15min", "1h", "4h", "execution"}) and not r.ok
            for r in reports
        )
