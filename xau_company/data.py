from __future__ import annotations

import lzma
import math
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from threading import Lock
from typing import Any

import pandas as pd
import requests


_CANDLE_STRUCT = struct.Struct(">5If")
_TICK_STRUCT = struct.Struct(">IIIff")
_INTERVAL_MINUTES = {
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
_RESAMPLE_RULE = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "45min": "45min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "8h": "8h",
    "1day": "1D",
}


class DukascopyClient:
    """Keyless Dukascopy public-datafeed client used by the XAU/USD company.

    Dukascopy publishes LZMA-compressed minute candles and tick files under its
    public datafeed host. The company downloads 1-minute BID candles, caches
    immutable past days in memory, and resamples that single source into every
    configured timeframe so desks cannot disagree because of mixed providers.

    ``api_key`` remains an accepted constructor argument only for backwards
    compatibility with older ``TwelveDataClient`` call sites. It is ignored.
    """

    BASE_URLS = (
        "https://datafeed.dukascopy.com/datafeed",
        "https://www.dukascopy.com/datafeed",
    )
    SYMBOLS: dict[str, tuple[str, float]] = {
        "XAU/USD": ("XAUUSD", 1000.0),
        "XAUUSD": ("XAUUSD", 1000.0),
        "DXY": ("DOLLARIDXUSD", 1000.0),
        "DOLLAR.IDX/USD": ("DOLLARIDXUSD", 1000.0),
        "DOLLARIDXUSD": ("DOLLARIDXUSD", 1000.0),
    }

    def __init__(
        self,
        api_key: str = "",
        timeout: int = 20,
        retries: int = 4,
        max_workers: int = 2,
        current_day_ttl_seconds: int = 30,
        max_history_days: int = 400,
        session: Any | None = None,
    ) -> None:
        del api_key
        self.timeout = max(1, int(timeout))
        self.retries = max(1, int(retries))
        self.max_workers = max(1, min(4, int(max_workers)))
        self.current_day_ttl_seconds = max(5, int(current_day_ttl_seconds))
        self.max_history_days = max(10, int(max_history_days))
        self.session = session or requests.Session()
        self._day_cache: dict[tuple[str, date], tuple[float, pd.DataFrame]] = {}
        self._cache_lock = Lock()

    @classmethod
    def _instrument(cls, symbol: str) -> tuple[str, float]:
        key = str(symbol).strip().upper().replace(" ", "")
        if key == "XAU/USD":
            key = "XAU/USD"
        elif key == "DOLLAR.IDX/USD":
            key = "DOLLAR.IDX/USD"
        mapped = cls.SYMBOLS.get(key)
        if mapped is None:
            raise ValueError(f"Unsupported Dukascopy symbol: {symbol}")
        return mapped

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    @staticmethod
    def _day_url(base: str, instrument: str, day: date, side: str = "BID") -> str:
        return (
            f"{base}/{instrument}/{day.year:04d}/{day.month - 1:02d}/{day.day:02d}/"
            f"{side.upper()}_candles_min_1.bi5"
        )

    @staticmethod
    def _tick_url(base: str, instrument: str, hour: datetime) -> str:
        return (
            f"{base}/{instrument}/{hour.year:04d}/{hour.month - 1:02d}/{hour.day:02d}/"
            f"{hour.hour:02d}h_ticks.bi5"
        )

    @staticmethod
    def _ohlc_valid(open_: int, high: int, low: int, close: int) -> bool:
        return high >= max(open_, close) and low <= min(open_, close) and high >= low

    @classmethod
    def decode_candle_payload(cls, payload: bytes, day: date, divisor: float) -> pd.DataFrame:
        if not payload:
            return cls._empty()
        try:
            raw = lzma.decompress(payload)
        except lzma.LZMAError as exc:
            raise RuntimeError("Dukascopy candle payload could not be decompressed") from exc
        if not raw:
            return cls._empty()
        if len(raw) % _CANDLE_STRUCT.size:
            raise RuntimeError("Unexpected Dukascopy candle record length")

        records = list(_CANDLE_STRUCT.iter_unpack(raw))
        score_oclh = sum(cls._ohlc_valid(r[1], r[4], r[3], r[2]) for r in records)
        score_ohlc = sum(cls._ohlc_valid(r[1], r[2], r[3], r[4]) for r in records)
        use_oclh = score_oclh >= score_ohlc

        midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        rows: list[dict[str, object]] = []
        for seconds, p1, p2, p3, p4, volume in records:
            if use_oclh:
                open_raw, close_raw, low_raw, high_raw = p1, p2, p3, p4
            else:
                open_raw, high_raw, low_raw, close_raw = p1, p2, p3, p4
            if not cls._ohlc_valid(open_raw, high_raw, low_raw, close_raw):
                continue
            rows.append(
                {
                    "datetime": midnight + timedelta(seconds=int(seconds)),
                    "open": float(open_raw) / divisor,
                    "high": float(high_raw) / divisor,
                    "low": float(low_raw) / divisor,
                    "close": float(close_raw) / divisor,
                    "volume": max(0.0, float(volume)),
                }
            )
        if not rows:
            return cls._empty()
        return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)

    @staticmethod
    def decode_tick_payload(payload: bytes, hour: datetime, divisor: float) -> tuple[datetime, float] | None:
        if not payload:
            return None
        try:
            raw = lzma.decompress(payload)
        except lzma.LZMAError as exc:
            raise RuntimeError("Dukascopy tick payload could not be decompressed") from exc
        if not raw:
            return None
        usable = len(raw) - (len(raw) % _TICK_STRUCT.size)
        if usable <= 0:
            return None
        latest: tuple[datetime, float] | None = None
        base = hour.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        for offset_ms, ask_raw, bid_raw, _ask_vol, _bid_vol in _TICK_STRUCT.iter_unpack(raw[:usable]):
            ask = float(ask_raw) / divisor
            bid = float(bid_raw) / divisor
            if ask <= 0 or bid <= 0 or ask < bid:
                continue
            stamp = base + timedelta(milliseconds=int(offset_ms))
            latest = (stamp, (ask + bid) / 2.0)
        return latest

    def _request_bytes(self, urls: list[str]) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            for url in urls:
                try:
                    response = self.session.get(
                        url,
                        timeout=self.timeout,
                        headers={
                            "User-Agent": "XAUUSD-Company/1.0",
                            "Accept-Encoding": "identity",
                            "Cache-Control": "no-cache",
                            "Connection": "close",
                        },
                    )
                    if response.status_code == 404:
                        continue
                    response.raise_for_status()
                    payload = bytes(response.content)
                    if payload:
                        return payload
                    last_error = RuntimeError("Dukascopy returned an empty payload")
                except (requests.RequestException, OSError, ValueError) as exc:
                    last_error = exc
            if attempt + 1 < self.retries:
                time.sleep(0.5 * (2**attempt))
        if last_error is not None:
            raise RuntimeError(f"Dukascopy request failed: {last_error}") from last_error
        return b""

    def _download_day(self, instrument: str, day: date, divisor: float) -> pd.DataFrame:
        urls = [self._day_url(base, instrument, day) for base in self.BASE_URLS]
        payload = self._request_bytes(urls)
        return self.decode_candle_payload(payload, day, divisor)

    def _load_day(self, instrument: str, day: date, divisor: float) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        key = (instrument, day)
        with self._cache_lock:
            cached = self._day_cache.get(key)
        if cached is not None:
            saved_at, frame = cached
            if day < now.date() or time.monotonic() - saved_at <= self.current_day_ttl_seconds:
                return frame.copy()

        frame = self._download_day(instrument, day, divisor)
        if not frame.empty:
            with self._cache_lock:
                self._day_cache[key] = (time.monotonic(), frame.copy())
        return frame

    @staticmethod
    def _calendar_days(interval: str, output_size: int) -> int:
        minutes = _INTERVAL_MINUTES.get(interval)
        if minutes is None:
            raise ValueError(f"Unsupported interval: {interval}")
        market_days = max(1.0, max(1, int(output_size)) * minutes / 1440.0)
        return max(3, int(math.ceil(market_days * 1.6)) + 4)

    def _fetch_m1(self, symbol: str, calendar_days: int) -> pd.DataFrame:
        instrument, divisor = self._instrument(symbol)
        days = min(max(3, int(calendar_days)), self.max_history_days)
        today = datetime.now(timezone.utc).date()
        requested = [today - timedelta(days=offset) for offset in range(days - 1, -1, -1)]
        frames: list[pd.DataFrame] = []
        recovered_days: set[date] = set()
        retry_days: set[date] = set()

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(requested))) as pool:
            future_map = {
                pool.submit(self._load_day, instrument, day, divisor): day for day in requested
            }
            for future in as_completed(future_map):
                day = future_map[future]
                try:
                    frame = future.result()
                except (RuntimeError, requests.RequestException, lzma.LZMAError):
                    retry_days.add(day)
                    continue
                if frame is not None and not frame.empty:
                    frames.append(frame)
                    recovered_days.add(day)
                else:
                    retry_days.add(day)

        # Dukascopy occasionally serves empty/transient day responses when several
        # files are requested together. Recover those days one-by-one, newest first,
        # so a temporary provider hiccup cannot zero the entire market-data snapshot.
        for day in sorted(retry_days, reverse=True):
            if day in recovered_days:
                continue
            try:
                frame = self._download_day(instrument, day, divisor)
            except (RuntimeError, requests.RequestException, lzma.LZMAError):
                continue
            if frame is not None and not frame.empty:
                frames.append(frame)
                recovered_days.add(day)
                with self._cache_lock:
                    self._day_cache[(instrument, day)] = (time.monotonic(), frame.copy())

        if not frames:
            raise RuntimeError(f"No Dukascopy minute data returned for {symbol} after recovery retries")
        combined = pd.concat(frames, ignore_index=True)
        combined["datetime"] = pd.to_datetime(combined["datetime"], utc=True, errors="coerce")
        return (
            combined.dropna(subset=["datetime", "open", "high", "low", "close"])
            .sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
            .reset_index(drop=True)
        )

    @staticmethod
    def _resample(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
        rule = _RESAMPLE_RULE.get(interval)
        if rule is None:
            raise ValueError(f"Unsupported interval: {interval}")
        if interval == "1min":
            return frame.copy().reset_index(drop=True)
        indexed = frame.set_index("datetime")
        out = indexed.resample(rule, origin="start_day", label="left", closed="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        return out.dropna(subset=["open", "high", "low", "close"]).reset_index()

    def candles(self, symbol: str, interval: str, output_size: int) -> pd.DataFrame:
        size = min(max(1, int(output_size)), 5000)
        days = self._calendar_days(interval, size)
        base = self._fetch_m1(symbol, days)
        return self._resample(base, interval).tail(size).reset_index(drop=True)

    def price(self, symbol: str) -> float:
        instrument, divisor = self._instrument(symbol)
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        for offset in range(6):
            hour = now - timedelta(hours=offset)
            urls = [self._tick_url(base, instrument, hour) for base in self.BASE_URLS]
            try:
                latest = self.decode_tick_payload(self._request_bytes(urls), hour, divisor)
            except RuntimeError:
                latest = None
            if latest is not None and latest[1] > 0:
                return float(latest[1])
        frame = self.candles(symbol, "1min", 3)
        if frame.empty:
            raise RuntimeError(f"No valid Dukascopy price returned for {symbol}")
        value = float(frame["close"].iloc[-1])
        if value <= 0:
            raise RuntimeError(f"No valid Dukascopy price returned for {symbol}")
        return value

    def safe_price(self, symbol: str) -> float | None:
        try:
            return self.price(symbol)
        except (requests.RequestException, RuntimeError, ValueError, OSError):
            return None

    def safe_candles(self, symbol: str, interval: str, output_size: int) -> pd.DataFrame | None:
        try:
            return self.candles(symbol, interval, output_size)
        except (requests.RequestException, RuntimeError, ValueError, OSError):
            return None

    def multi_timeframe(
        self,
        symbol: str,
        intervals: tuple[str, ...] = ("1min", "5min", "15min", "1h", "4h"),
        output_size: int = 1000,
    ) -> dict[str, pd.DataFrame]:
        requested = tuple(intervals)
        if not requested:
            return {}
        size = min(max(1, int(output_size)), 5000)
        days = max(self._calendar_days(interval, size) for interval in requested)
        try:
            base = self._fetch_m1(symbol, days)
        except (requests.RequestException, RuntimeError, ValueError, OSError):
            return {}
        frames: dict[str, pd.DataFrame] = {}
        for interval in requested:
            try:
                data = self._resample(base, interval).tail(size).reset_index(drop=True)
            except ValueError:
                continue
            if not data.empty:
                frames[interval] = data
        return frames


# Backwards-compatible import name used by main.py and older tests. This is now
# the keyless Dukascopy implementation; no Twelve Data request is made.
TwelveDataClient = DukascopyClient
