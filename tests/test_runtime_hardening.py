from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from xau_company.models import Direction, TradeSignal
from xau_company.outcomes import OutcomeCalibrationAgent
from xau_company.runtime_quality import fetch_resolution_history, revalidate_optional_macro
from xau_company.telegram import TelegramNotifier


def _frame(freq: str, n: int = 20, start: str = "2026-08-31T15:00:00Z") -> pd.DataFrame:
    dt = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = [100.0 + i * 0.1 for i in range(n)]
    return pd.DataFrame({
        "datetime": dt,
        "open": close,
        "high": [x + 0.2 for x in close],
        "low": [x - 0.2 for x in close],
        "close": close,
    })


def test_resolution_history_falls_back_from_1m_to_5m():
    calls = []

    class Market:
        def safe_candles(self, symbol, interval, output_size):
            calls.append(interval)
            return None if interval == "1min" else _frame("5min")

    class Quality:
        def clean_frame(self, frame, interval, now, require_fresh=True):
            return frame, SimpleNamespace(ok=True)

    frame, minutes, interval = fetch_resolution_history(
        Market(), Quality(), "XAU/USD", 72, 500, pd.Timestamp("2026-08-31T17:00:00Z")
    )
    assert calls[:2] == ["1min", "5min"]
    assert frame is not None
    assert minutes == 5
    assert interval == "5min"


def test_cached_macro_context_is_revalidated_and_dropped_when_stale():
    class Quality:
        def clean_frame(self, frame, interval, now, require_fresh=True):
            return frame.iloc[0:0], SimpleNamespace(ok=False)

    assert revalidate_optional_macro(Quality(), _frame("1h"), "1h", pd.Timestamp("2026-08-31T17:00:00Z")) is None


def _signal(direction: Direction, entry: float = 100.0) -> TradeSignal:
    stop = 98.0 if direction == Direction.BUY else 102.0
    target = 104.0 if direction == Direction.BUY else 96.0
    return TradeSignal(
        "XAU/USD", direction, entry, stop, target, 0.80, "trend_up", ["test"], [],
        selected_strategy="trend(5, 30, 0.0)",
        strategy_stats={
            "research_interval": "15min",
            "max_holding_minutes": 30,
            "resolution_interval_minutes": 5,
        },
    )


def test_forward_timeout_uses_setup_end_and_timeout_close_like_backtester(tmp_path):
    tracker = OutcomeCalibrationAgent(str(tmp_path / "timeout.sqlite3"))
    setup = pd.Timestamp("2026-08-31T15:00:00Z")
    observed = pd.Timestamp("2026-08-31T15:16:00Z")
    assert tracker.record(_signal(Direction.BUY), observed, setup_at=setup)

    candles = pd.DataFrame({
        "datetime": pd.date_range("2026-08-31T15:20:00Z", periods=5, freq="5min", tz="UTC"),
        "high": [100.5, 100.7, 100.8, 101.0, 101.2],
        "low": [99.5, 99.7, 99.8, 100.0, 100.2],
        "close": [100.1, 100.2, 100.4, 100.6, 101.0],
    })
    resolved = tracker.resolve_open(candles, interval_minutes=5)
    assert resolved["wins"] == 1
    assert resolved["expired"] == 0
    summary = tracker.summary()
    assert summary["wins"] == 1
    assert summary["losses"] == 0


def test_forward_timeout_negative_close_is_loss(tmp_path):
    tracker = OutcomeCalibrationAgent(str(tmp_path / "timeout_loss.sqlite3"), max_age_hours=1)
    setup = pd.Timestamp("2026-08-31T15:00:00Z")
    observed = pd.Timestamp("2026-08-31T15:16:00Z")
    assert tracker.record(_signal(Direction.BUY), observed, setup_at=setup)

    candles = pd.DataFrame({
        "datetime": pd.date_range("2026-08-31T15:20:00Z", periods=5, freq="5min", tz="UTC"),
        "high": [100.3, 100.2, 100.1, 100.0, 99.9],
        "low": [99.4, 99.2, 99.0, 98.8, 98.6],
        "close": [99.9, 99.7, 99.5, 99.3, 99.0],
    })
    resolved = tracker.resolve_open(candles, interval_minutes=5)
    assert resolved["losses"] == 1
    assert resolved["expired"] == 0


def test_telegram_signal_is_capped_without_losing_core_trade_fields():
    signal = _signal(Direction.BUY)
    signal.selected_strategy = "strategy-" + ("X" * 2000)
    signal.reasons = [("reason " + str(i) + " ") + ("Y" * 900) for i in range(20)]
    text = TelegramNotifier("token", "chat").format_signal(signal)
    assert len(text) <= TelegramNotifier.MAX_MESSAGE_CHARS
    assert "Action: BUY" in text
    assert "Entry: 100.00" in text
    assert "TP: 104.00" in text
    assert "SL: 98.00" in text
