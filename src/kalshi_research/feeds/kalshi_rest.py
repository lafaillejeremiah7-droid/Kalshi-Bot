from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable

import httpx


@dataclass(slots=True)
class KalshiRestClient:
    base_url: str = "https://external-api.kalshi.com/trade-api/v2"
    timeout_s: float = 10.0

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout_s)

    def get_series(self, series_ticker: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.get(f"/series/{series_ticker}")
            response.raise_for_status()
            return response.json()["series"]

    def get_markets(self, *, series_ticker: str, status: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"series_ticker": series_ticker, "limit": 1000}
        if status:
            params["status"] = status
        out: list[dict[str, Any]] = []
        cursor = ""
        with self._client() as client:
            while True:
                if cursor:
                    params["cursor"] = cursor
                response = client.get("/markets", params=params)
                response.raise_for_status()
                body = response.json()
                out.extend(body.get("markets", []))
                cursor = body.get("cursor") or ""
                if not cursor:
                    return out

    def get_trades(self, ticker: str, min_ts: int | None = None, max_ts: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"ticker": ticker, "limit": 1000}
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        out: list[dict[str, Any]] = []
        cursor = ""
        with self._client() as client:
            while True:
                if cursor:
                    params["cursor"] = cursor
                response = client.get("/markets/trades", params=params)
                response.raise_for_status()
                body = response.json()
                out.extend(body.get("trades", []))
                cursor = body.get("cursor") or ""
                if not cursor:
                    return out

    def get_historical_cutoff(self) -> dict[str, Any]:
        with self._client() as client:
            response = client.get("/historical/cutoff")
            response.raise_for_status()
            return response.json()

    def get_fee_changes(self, series_ticker: str, show_historical: bool = True) -> list[dict[str, Any]]:
        with self._client() as client:
            response = client.get(
                "/series/fee_changes",
                params={"series_ticker": series_ticker, "show_historical": str(show_historical).lower()},
            )
            response.raise_for_status()
            return response.json().get("series_fee_change_arr", [])


def parse_decimal(value: str | int | float | None) -> Decimal | None:
    if value is None or value == "":
        return None
    return Decimal(str(value))


def market_target(market: dict[str, Any]) -> Decimal:
    """Extract target defensively from current market metadata.

    BTC15m market payloads/rules can expose strike information in different
    fields. We do not guess silently: ambiguous payloads should fail loudly so
    research does not train on the wrong target.
    """
    candidates: Iterable[Any] = (
        market.get("floor_strike"),
        market.get("cap_strike"),
        market.get("functional_strike"),
        (market.get("custom_strike") or {}).get("target_price")
        if isinstance(market.get("custom_strike"), dict)
        else None,
    )
    parsed = [parse_decimal(v) for v in candidates if v not in (None, "")]
    unique = {v for v in parsed if v is not None and v > 1}
    if len(unique) != 1:
        raise ValueError(f"unable to determine unique BTC target from payload: {unique}")
    return unique.pop()
