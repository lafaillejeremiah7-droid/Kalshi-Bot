from __future__ import annotations

import time

import pandas as pd
import requests


class TwelveDataClient:
    BASE_URL = "https://api.twelvedata.com/time_series"
    PRICE_URL = "https://api.twelvedata.com/price"

    def __init__(self, api_key: str, timeout: int = 20, retries: int = 3) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.retries = max(1, int(retries))

    def _get_json(self, url: str, params: dict[str, object]) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and payload.get("status") == "error":
                    raise RuntimeError(payload.get("message", payload))
                if not isinstance(payload, dict):
                    raise RuntimeError("Unexpected Twelve Data response payload")
                return payload
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(0.4 * (2**attempt))
        raise RuntimeError(f"Twelve Data request failed: {last_error}") from last_error

    def candles(self, symbol: str, interval: str, output_size: int) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": min(max(1, int(output_size)), 5000),
            "apikey": self.api_key,
            "format": "JSON",
        }
        payload = self._get_json(self.BASE_URL, params)
        values = payload.get("values")
        if not values:
            raise RuntimeError(f"No market data returned for {symbol} {interval}")
        df = pd.DataFrame(values)
        for col in ("open", "high", "low", "close"):
            if col not in df:
                raise RuntimeError(f"Missing {col} in Twelve Data response for {symbol} {interval}")
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" in df:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        if "datetime" not in df:
            raise RuntimeError(f"Missing datetime in Twelve Data response for {symbol} {interval}")
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        return df.sort_values("datetime").dropna(subset=["datetime", "open", "high", "low", "close"]).reset_index(drop=True)

    def price(self, symbol: str) -> float:
        payload = self._get_json(
            self.PRICE_URL,
            {"symbol": symbol, "apikey": self.api_key, "format": "JSON"},
        )
        value = pd.to_numeric(payload.get("price"), errors="coerce")
        if pd.isna(value) or float(value) <= 0:
            raise RuntimeError(f"No valid live price returned for {symbol}")
        return float(value)

    def safe_price(self, symbol: str) -> float | None:
        try:
            return self.price(symbol)
        except (requests.RequestException, RuntimeError, ValueError):
            return None

    def safe_candles(self, symbol: str, interval: str, output_size: int) -> pd.DataFrame | None:
        try:
            return self.candles(symbol, interval, output_size)
        except (requests.RequestException, RuntimeError, ValueError):
            return None

    def multi_timeframe(
        self,
        symbol: str,
        intervals: tuple[str, ...] = ("1min", "5min", "15min", "1h", "4h"),
        output_size: int = 1000,
    ) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for interval in intervals:
            data = self.safe_candles(symbol, interval, output_size)
            if data is not None and not data.empty:
                frames[interval] = data
        return frames
