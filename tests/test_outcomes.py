import pandas as pd

from xau_company.models import Direction, TradeSignal
from xau_company.outcomes import OutcomeCalibrationAgent


def test_outcomes_ignore_price_action_before_actual_emission(tmp_path):
    tracker = OutcomeCalibrationAgent(str(tmp_path / "timing.sqlite3"))
    signal = TradeSignal(
        "XAU/USD",
        Direction.BUY,
        100.0,
        98.0,
        102.0,
        0.80,
        "trend_up",
        ["test"],
        [],
        selected_strategy="trend(5, 30, 0.0)",
    )

    setup_at = pd.Timestamp("2026-08-30T15:00:00Z")
    emitted_at = pd.Timestamp("2026-08-30T15:05:00Z")
    assert tracker.record(
        signal,
        emitted_at,
        selection_confidence=0.80,
        setup_at=setup_at,
    ) is True
    assert tracker.exists(signal, setup_at) is True

    candles = pd.DataFrame(
        {
            "datetime": [
                pd.Timestamp("2026-08-30T15:04:00Z"),
                pd.Timestamp("2026-08-30T15:06:00Z"),
            ],
            # The pre-emission candle hits SL and must be ignored. The genuinely
            # future candle hits only TP and therefore resolves as a win.
            "high": [101.0, 103.0],
            "low": [97.0, 99.0],
        }
    )

    resolved = tracker.resolve_open(candles)
    assert resolved == {"wins": 1, "losses": 0, "expired": 0}
    summary = tracker.summary()
    assert summary["wins"] == 1
    assert summary["losses"] == 0
