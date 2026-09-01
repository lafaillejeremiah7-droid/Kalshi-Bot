import pandas as pd
import pytest

from xau_company import preflight


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class _DukascopyOK:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def candles(self, symbol, interval, output_size):
        assert symbol == "XAU/USD"
        assert interval == "1min"
        assert output_size == 10
        return pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2026-08-31T23:59:00Z")],
                "open": [2500.0],
                "high": [2501.0],
                "low": [2499.0],
                "close": [2500.5],
                "volume": [1.0],
            }
        )

    def price(self, symbol):
        assert symbol == "XAU/USD"
        return 2500.25


class _DukascopyEmpty(_DukascopyOK):
    def candles(self, symbol, interval, output_size):
        return pd.DataFrame()


def test_dukascopy_preflight_accepts_live_data_without_api_key(monkeypatch):
    monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)
    monkeypatch.setattr(preflight, "DukascopyClient", _DukascopyOK)
    price, stamp = preflight.check_dukascopy()
    assert price == 2500.25
    assert "2026-08-31 23:59:00+00:00" in stamp


def test_dukascopy_preflight_rejects_missing_market_data(monkeypatch):
    monkeypatch.setattr(preflight, "DukascopyClient", _DukascopyEmpty)
    with pytest.raises(RuntimeError, match="no XAU/USD minute candles"):
        preflight.check_dukascopy()


def test_telegram_preflight_checks_bot_and_chat_without_sending(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    session = _Session([_Response({"ok": True}), _Response({"ok": True})])
    preflight.check_telegram(session)
    assert session.calls[0][0].endswith("/getMe")
    assert session.calls[1][0].endswith("/getChat")
    assert all("sendMessage" not in url for url, _ in session.calls)


def test_telegram_preflight_fails_when_required_secret_missing(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        preflight.check_telegram(_Session([]))
