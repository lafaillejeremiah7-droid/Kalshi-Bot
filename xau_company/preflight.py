from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from xau_company.data import DukascopyClient


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for runtime preflight")
    return value


def check_dukascopy() -> tuple[float, str]:
    symbol = os.getenv("SYMBOL", "XAU/USD")
    client = DukascopyClient(timeout=6, retries=1, max_workers=1, recent_tick_hours=3)

    # A readiness check only needs one recent completed tick hour. Stop as soon
    # as one valid file is found instead of downloading the full intraday window.
    live_api = all(
        hasattr(client, name)
        for name in ("_instrument", "_tick_url", "_request_bytes", "decode_tick_candles", "BASE_URLS")
    )
    if live_api:
        instrument, divisor = client._instrument(symbol)
        now = datetime.now(timezone.utc)
        current_minute = now.replace(second=0, microsecond=0)
        current_hour = current_minute.replace(minute=0)
        candles = client._empty()
        for offset in range(3):
            hour = current_hour - timedelta(hours=offset)
            urls = [client._tick_url(base, instrument, hour) for base in client.BASE_URLS]
            try:
                payload = client._request_bytes(urls, attempts=1)
                candidate = client.decode_tick_candles(payload, hour, divisor)
            except RuntimeError:
                continue
            if candidate.empty:
                continue
            candidate = candidate[pd.to_datetime(candidate["datetime"], utc=True) < current_minute]
            if not candidate.empty:
                candles = candidate
                break
        if candles.empty:
            raise RuntimeError("Dukascopy returned no XAU/USD minute candles")
        price = float(candles["close"].iloc[-1])
    else:
        # Compatibility path used by lightweight tests/older clients.
        candles = client.candles(symbol, "1min", 10)
        if candles.empty:
            raise RuntimeError("Dukascopy returned no XAU/USD minute candles")
        price = client.price(symbol)

    if price <= 0:
        raise RuntimeError("Dukascopy returned a non-positive XAU/USD price")
    stamp = str(candles["datetime"].iloc[-1])
    return price, stamp


def check_telegram(session: Any = requests) -> None:
    token = _required("TELEGRAM_BOT_TOKEN")
    chat_id = _required("TELEGRAM_CHAT_ID")
    base = f"https://api.telegram.org/bot{token}"

    me = session.get(f"{base}/getMe", timeout=20)
    me.raise_for_status()
    me_payload = me.json()
    if not me_payload.get("ok"):
        raise RuntimeError("Telegram getMe rejected the configured bot token")

    chat = session.get(f"{base}/getChat", params={"chat_id": chat_id}, timeout=20)
    chat.raise_for_status()
    chat_payload = chat.json()
    if not chat_payload.get("ok"):
        raise RuntimeError("Telegram getChat rejected the configured chat ID")


def run() -> int:
    check_telegram()
    print("Telegram preflight: OK (bot token and chat reachable; no message sent)")

    price, stamp = check_dukascopy()
    print(
        f"Dukascopy preflight: OK ({os.getenv('SYMBOL', 'XAU/USD')} "
        f"price={price:.2f}, latest_1m={stamp})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
