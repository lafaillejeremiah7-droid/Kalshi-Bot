import pytest
import requests

from xau_company.config import Settings
from xau_company.models import Direction, TradeSignal
from xau_company.telegram import TelegramNotifier, TelegramRejectedError


def _signal() -> TradeSignal:
    return TradeSignal(
        symbol="XAU/USD",
        direction=Direction.BUY,
        entry=2500.0,
        stop_loss=2490.0,
        take_profit=2517.0,
        confidence=0.80,
        regime="trend_up",
        reasons=["test"],
        votes=[],
        selected_strategy="trend(5, 30, 0.0)",
    )


def test_live_mode_requires_telegram_credentials():
    cfg = Settings(
        paper_mode=False,
        telegram_bot_token="",
        telegram_chat_id="",
    )
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        cfg.validate()


def test_consensus_cannot_exceed_directional_specialist_count():
    cfg = Settings(
        min_consensus=7,
    )
    with pytest.raises(ValueError, match="MIN_CONSENSUS"):
        cfg.validate()


def test_telegram_4xx_is_definitive_rejection(monkeypatch):
    class Response:
        status_code = 400
        text = "bad request"

        def raise_for_status(self):
            raise requests.HTTPError("400")

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Response())
    notifier = TelegramNotifier("token", "chat")
    with pytest.raises(TelegramRejectedError, match="HTTP 400"):
        notifier.send(_signal())


def test_telegram_5xx_remains_delivery_uncertain(monkeypatch):
    class Response:
        status_code = 500
        text = "server error"

        def raise_for_status(self):
            raise requests.HTTPError("500")

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Response())
    notifier = TelegramNotifier("token", "chat")
    with pytest.raises(requests.HTTPError):
        notifier.send(_signal())
