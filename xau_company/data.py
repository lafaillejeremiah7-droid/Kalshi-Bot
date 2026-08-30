from __future__ import annotations

import requests
import pandas as pd


class TwelveDataClient:
    BASE_URL = "https://api.twelvedata.com/time_series"

    def __init__(self, api_key: str, timeout: int = 20) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def candles(self, symbol: str, interval: str, output_size: int) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": min(output_size, 5000),
            "apikey": self.api_key,
            "format": "JSON",
        }
        response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "error":
            raise RuntimeError(f"Twelve Data error: {payload.get('message', payload)}")
        values = payload.get("values")
        if not values:
            raise RuntimeError("No market data returned")
        df = pd.DataFrame(values)
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" in df:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        return df.sort_values("datetime").dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
