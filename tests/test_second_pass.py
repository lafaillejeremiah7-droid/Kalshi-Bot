import numpy as np
import pandas as pd

from xau_company.models import Direction, TradeSignal
from xau_company.outcomes import OutcomeCalibrationAgent
from xau_company.quality import MarketDataQualityAgent
from xau_company.research import Candidate, CandidateScore
from xau_company.strategy_evolution import StrategyEvolutionAgent


def _signal(strategy: str, direction: Direction = Direction.BUY) -> TradeSignal:
    return TradeSignal(
        symbol="XAU/USD",
        direction=direction,
        entry=2500.0,
        stop_loss=2490.0 if direction == Direction.BUY else 2510.0,
        take_profit=2517.0 if direction == Direction.BUY else 2483.0,
        confidence=0.80,
        regime="trend_up",
        reasons=["test"],
        votes=[],
        selected_strategy=strategy,
        strategy_stats={"max_holding_minutes": 90, "resolution_interval_minutes": 1},
    )


def _score(candidate: Candidate, score: float = 0.78) -> CandidateScore:
    return CandidateScore(
        candidate=candidate,
        train_hit_rate=0.64,
        valid_hit_rate=0.63,
        trades=120,
        score=score,
        walk_forward_hit_rate=0.63,
        walk_forward_std=0.03,
        expectancy=0.001,
        profit_factor=1.45,
        folds=4,
        regime_scores={"trend_up": 0.63},
        regime_trades={"trend_up": 60},
        avg_r_multiple=0.22,
        max_drawdown_r=3.0,
        max_loss_streak=4,
    )


def _frame(interval: str, now: pd.Timestamp, rows: int = 250) -> pd.DataFrame:
    minutes = {"1min": 1, "5min": 5, "15min": 15, "1h": 60, "4h": 240}[interval]
    end_start = now - pd.Timedelta(minutes=minutes)
    dates = pd.date_range(end=end_start, periods=rows, freq=f"{minutes}min", tz="UTC")
    close = np.linspace(2000.0, 2100.0, rows)
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
        }
    )


def test_one_research_candle_cannot_emit_two_different_strategy_decisions(tmp_path):
    tracker = OutcomeCalibrationAgent(str(tmp_path / "one-setup.sqlite3"))
    setup = pd.Timestamp("2026-08-31T15:00:00Z")
    first = _signal("trend(5, 30, 0.0)", Direction.BUY)
    changed = _signal("breakout(20, 0.0, 5)", Direction.SELL)

    assert tracker.record(first, setup + pd.Timedelta(seconds=10), setup_at=setup) is True
    assert tracker.exists(changed, setup) is True
    assert tracker.record(changed, setup + pd.Timedelta(minutes=1), setup_at=setup) is False


def test_candle_starting_at_holding_expiry_cannot_resolve_trade(tmp_path):
    tracker = OutcomeCalibrationAgent(str(tmp_path / "expiry.sqlite3"), max_age_hours=1)
    observed = pd.Timestamp("2026-08-31T15:00:00Z")
    signal = _signal("trend(5, 30, 0.0)")
    signal.strategy_stats["max_holding_minutes"] = 1
    assert tracker.record(signal, observed, setup_at=observed)

    candles = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2026-08-31T15:01:00Z")],
            "high": [2600.0],
            "low": [2500.0],
        }
    )
    resolved = tracker.resolve_open(candles, interval_minutes=1)
    assert resolved["wins"] == 0
    assert resolved["losses"] == 0
    assert resolved["expired"] == 1


def test_full_experimental_queue_rotates_but_keeps_fixed_capacity():
    bot = StrategyEvolutionAgent(":memory:", discoveries_per_cycle=10, max_library_size=100)
    bot._write(
        [
            {"family": "ensemble", "params": ["old", i], "status": "EXPERIMENTAL"}
            for i in range(100)
        ]
    )
    old_keys = {bot.spec_key(row["family"], row["params"]) for row in bot.entries()}
    parents = [
        _score(Candidate("trend", (5, 30, 0.0))),
        _score(Candidate("breakout", (20, 0.0002, 5))),
        _score(Candidate("momentum", (5, 0.0004, 50))),
        _score(Candidate("mean_reversion", (14, 30, 70, 1.0, 30))),
    ]

    added = bot.propose(parents)
    assert added == 10
    saved = bot.entries()
    assert len(saved) == 100
    new_keys = {bot.spec_key(row["family"], row["params"]) for row in saved}
    assert len(old_keys - new_keys) == 10
    assert len(new_keys - old_keys) == 10


def test_required_higher_timeframe_missing_causes_quality_failure():
    now = pd.Timestamp("2026-08-31T15:00:00Z")
    guard = MarketDataQualityAgent(max_stale_multiplier=4.0)
    frames = {
        "1min": _frame("1min", now),
        "5min": _frame("5min", now),
        "15min": _frame("15min", now),
        "1h": _frame("1h", now),
        # 4h intentionally missing.
    }
    _, reports = guard.clean_frames(
        frames,
        now=now,
        required_context=("15min", "1h", "4h"),
        execution_choices=("1min", "5min"),
    )
    assert any(
        report.timeframe == "4h" and not report.ok and "required context" in report.reason
        for report in reports
    )
