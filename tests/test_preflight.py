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


def test_twelve_data_preflight_accepts_positive_price(monkeypatch):
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "test-key")
    session = _Session([_Response({"price": "2500.25"})])
    assert preflight.check_twelve_data(session) == 2500.25
    _, kwargs = session.calls[0]
    assert kwargs["params"]["symbol"] == "XAU/USD"


def test_telegram_preflight_checks_bot_and_chat_without_sending(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    session = _Session([_Response({"ok": True}), _Response({"ok": True})])
    preflight.check_telegram(session)
    assert session.calls[0][0].endswith("/getMe")
    assert session.calls[1][0].endswith("/getChat")
    assert all("sendMessage" not in url for url, _ in session.calls)


def test_preflight_fails_when_required_secret_missing(monkeypatch):
    monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        preflight.check_twelve_data(_Session([]))
