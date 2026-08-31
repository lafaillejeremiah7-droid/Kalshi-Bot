from __future__ import annotations

import os
from typing import Any

import requests


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for runtime preflight")
    return value


def check_twelve_data(session: Any = requests) -> float:
    api_key = _required("TWELVE_DATA_API_KEY")
    symbol = os.getenv("SYMBOL", "XAU/USD")
    response = session.get(
        "https://api.twelvedata.com/price",
        params={"symbol": symbol, "apikey": api_key},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if "price" not in payload:
        raise RuntimeError(f"Twelve Data price check failed: {payload.get('message', 'missing price')}")
    price = float(payload["price"])
    if price <= 0:
        raise RuntimeError("Twelve Data returned a non-positive price")
    return price


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
    price = check_twelve_data()
    print(f"Twelve Data preflight: OK ({os.getenv('SYMBOL', 'XAU/USD')} price={price:.2f})")
    check_telegram()
    print("Telegram preflight: OK (bot token and chat reachable; no message sent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
