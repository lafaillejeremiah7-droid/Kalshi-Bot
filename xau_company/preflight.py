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
    client = DukascopyClient(timeout=8, retries=2, max_workers=1, recent_tick_hours=3)

    # Production Dukascopy clients validate the live hourly tick path directly.
    # The public candles() fallback keeps compatibility with lightweight test
    # doubles and older client implementations.
    if hasattr(client, "_fetch_recent_tick_m1") and hasattr(client, "_instrument"):
        instrument, divisor = client._instrument(symbol)
        candles = client._fetch_recent_tick_m1(instrument, divisor)
    else:
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
