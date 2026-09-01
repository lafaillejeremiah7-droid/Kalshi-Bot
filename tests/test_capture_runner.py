from types import SimpleNamespace

import pytest

from kalshi_research.capture import runner
from kalshi_research.capture.runner import build_subscription_messages, discover_open_btc15m_market


def test_capture_subscriptions_are_isolated_and_non_trading():
    messages = build_subscription_messages("KXBTC15M-TEST")
    assert [m["id"] for m in messages] == [1, 2, 3]
    assert messages[0]["params"] == {
        "channels": ["orderbook_delta"],
        "market_tickers": ["KXBTC15M-TEST"],
    }
    assert messages[1]["params"] == {
        "channels": ["trade"],
        "market_tickers": ["KXBTC15M-TEST"],
    }
    assert messages[2]["params"] == {
        "channels": ["cfbenchmarks_value"],
        "index_ids": ["BRTI"],
    }
    serialized = repr(messages).lower()
    assert "order" not in serialized.replace("orderbook_delta", "")
    assert "fill" not in serialized


def test_capture_rejects_non_btc15m_ticker():
    with pytest.raises(ValueError):
        build_subscription_messages("KXETH15M-TEST")


def test_market_discovery_picks_nearest_future_close(monkeypatch):
    class FakeClient:
        def __init__(self, _base_url):
            pass

        def get_markets(self, *, series_ticker, status):
            assert series_ticker == "KXBTC15M"
            assert status == "open"
            return [
                {"ticker": "KXBTC15M-LATER", "close_time": "2099-01-01T00:30:00Z"},
                {"ticker": "KXBTC15M-NEXT", "close_time": "2099-01-01T00:15:00Z"},
            ]

    monkeypatch.setattr(runner, "KalshiRestClient", FakeClient)
    config = SimpleNamespace(
        kalshi_rest_base="https://example.invalid/trade-api/v2",
        kalshi_series_ticker="KXBTC15M",
    )
    assert discover_open_btc15m_market(config) == "KXBTC15M-NEXT"


def test_market_discovery_fails_without_open_market(monkeypatch):
    class FakeClient:
        def __init__(self, _base_url):
            pass

        def get_markets(self, *, series_ticker, status):
            return []

    monkeypatch.setattr(runner, "KalshiRestClient", FakeClient)
    config = SimpleNamespace(
        kalshi_rest_base="https://example.invalid/trade-api/v2",
        kalshi_series_ticker="KXBTC15M",
    )
    with pytest.raises(RuntimeError):
        discover_open_btc15m_market(config)
