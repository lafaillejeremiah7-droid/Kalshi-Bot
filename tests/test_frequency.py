from datetime import datetime, timezone

from xau_company.frequency import TradeFrequencyGuard
from xau_company.models import Direction, TradeSignal
from xau_company.outcomes import OutcomeCalibrationAgent
from xau_company.telegram import TelegramNotifier


def _signal(strategy: str, entry: float) -> TradeSignal:
    return TradeSignal(
        "XAU/USD",
        Direction.BUY,
        entry,
        entry - 10,
        entry + 17,
        0.80,
        "trend_up",
        ["test"],
        [],
        selected_strategy=strategy,
    )


def test_frequency_guard_allows_zero_one_or_two_setups_but_never_third(tmp_path):
    tracker = OutcomeCalibrationAgent(str(tmp_path / "frequency.sqlite3"))
    guard = TradeFrequencyGuard("America/Chicago", max_trades_per_day=2)
    now = datetime(2026, 8, 31, 15, 0, tzinfo=timezone.utc)  # Monday, 10:00 Chicago
    start, end = guard.day_bounds_utc(now)

    assert tracker.count_emitted_between(start, end) == 0
    assert guard.evaluate(now, 0).allowed is True

    first = _signal("trend(5, 30, 0.0)", 2500)
    assert tracker.record(first, now, setup_at=now) is True
    assert tracker.count_emitted_between(start, end) == 1
    assert guard.evaluate(now, 1).allowed is True

    second_time = datetime(2026, 8, 31, 19, 0, tzinfo=timezone.utc)
    second = _signal("breakout(20, 0.0, 5)", 2510)
    assert tracker.record(second, second_time, setup_at=second_time) is True
    assert tracker.count_emitted_between(start, end) == 2

    decision = guard.evaluate(second_time, 2)
    assert decision.allowed is False
    assert decision.remaining_today == 0
    assert "daily trade cap reached" in decision.reason


def test_frequency_guard_blocks_weekends_even_with_zero_trades():
    guard = TradeFrequencyGuard("America/Chicago", max_trades_per_day=2)
    sunday = datetime(2026, 8, 30, 18, 0, tzinfo=timezone.utc)
    decision = guard.evaluate(sunday, 0)
    assert decision.allowed is False
    assert decision.remaining_today == 0
    assert "Monday-Friday" in decision.reason


def test_trade_day_uses_chicago_calendar_not_utc_calendar():
    guard = TradeFrequencyGuard("America/Chicago", max_trades_per_day=2)
    # 02:00 UTC Tuesday is still 21:00 Monday in Chicago during daylight time.
    now = datetime(2026, 9, 1, 2, 0, tzinfo=timezone.utc)
    decision = guard.evaluate(now, 0)
    assert decision.allowed is True
    assert decision.local_date == "2026-08-31"

    start, end = guard.day_bounds_utc(now)
    assert start == datetime(2026, 8, 31, 5, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 9, 1, 5, 0, tzinfo=timezone.utc)


def test_telegram_shows_daily_trade_slot():
    signal = _signal("trend(5, 30, 0.0)", 2500)
    signal.strategy_stats = {
        "trades_today_before_signal": 1,
        "daily_trade_cap": 2,
    }
    text = TelegramNotifier("", "").format_signal(signal)
    assert "Daily trade slot: 2/2" in text
