from __future__ import annotations

import os
from typing import Any

import requests

from xau_company.data import DukascopyClient


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for runtime preflight")
    return value


def check_dukascopy() -> tuple[float, str]:
    symbol = os.getenv("SYMBOL", "XAU/USD")
    client = DukascopyClient(timeout=20, retries=3, max_workers=1)

    # Preflight is a live-readiness check, not a historical backfill. Validate
    # the same hourly tick source used by the runtime's intraday fallback so an
    # absent current-day candle archive cannot make startup appear unhealthy.
    instrument, divisor = client._instrument(symbol)
    candles = client._fetch_recent_tick_m1(instrument, divisor)
    if candles.empty:
        raise RuntimeError("Dukascopy returned no recent XAU/USD tick-derived minute candles")

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
