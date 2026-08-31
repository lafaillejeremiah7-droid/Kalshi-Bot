import numpy as np
import pandas as pd

from xau_company.backtest import TradeLifecycleBacktester
from xau_company.context import MultiTimeframeAgent, NewsRiskAgent
from xau_company.models import AgentVote, Direction, TradeSignal
from xau_company.outcomes import OutcomeCalibrationAgent
from xau_company.research import Candidate, CandidateScore, StrategyResearchAgent
from xau_company.selector import StrategySelectorAgent
from xau_company.telegram import TelegramNotifier


def sample_df(n=900):
    rng = np.random.default_rng(7)
    close = 2000 + np.cumsum(rng.normal(0.2, 2.0, n))
    open_ = close + rng.normal(0, 0.8, n)
    high = np.maximum(open_, close) + rng.uniform(0.2, 2.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.2, 2.0, n)
    return pd.DataFrame({
        "datetime": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
    })


def trending_df(n=500, upward=True):
    close = np.linspace(2000, 2200, n) if upward else np.linspace(2200, 2000, n)
    open_ = close - 0.4 if upward else close + 0.4
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    return pd.DataFrame({
        "datetime": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
    })


def test_trade_lifecycle_enters_next_bar_and_assumes_stop_first_on_collision():
    df = pd.DataFrame({
        "open":  [95, 95, 95, 100, 100, 100, 100],
        "high":  [96, 96, 96, 103, 101, 101, 101],
        "low":   [94, 94, 94, 97, 99, 99, 99],
        "close": [95, 95, 95, 100, 100, 100, 100],
    })
    signal = pd.Series([0, 0, 1, 0, 0, 0, 0], dtype="int8")
    atr_values = pd.Series([2.0] * len(df))
    tester = TradeLifecycleBacktester(spread_bps=0, slippage_bps=0, stop_atr=1.0, reward_risk=1.0)

    trades = tester.simulate(df, signal, atr_values, max_holding=3)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.signal_index == 2
    assert trade.entry_index == 3
    assert trade.entry_price == 100.0
    assert trade.exit_reason == "stop"
    assert trade.exit_price == 98.0
    assert trade.r_multiple == -1.0


def test_outcome_ledger_dedupes_and_resolves_collision_as_loss(tmp_path):
    tracker = OutcomeCalibrationAgent(str(tmp_path / "outcomes.sqlite3"))
    signal = TradeSignal(
        "XAU/USD", Direction.BUY, 100, 98, 102, 0.80, "trend_up", ["test"], [],
        selected_strategy="trend(5, 30, 0.0)",
    )
    observed = pd.Timestamp("2026-08-30T15:00:00Z")

    assert tracker.exists(signal, observed) is False
    assert tracker.record(signal, observed, selection_confidence=0.80) is True
    assert tracker.exists(signal, observed) is True
    assert tracker.record(signal, observed, selection_confidence=0.80) is False

    candles = pd.DataFrame({
        "datetime": [pd.Timestamp("2026-08-30T15:05:00Z")],
        "high": [103.0],
        "low": [97.0],
    })
    resolved = tracker.resolve_open(candles)
    assert resolved == {"wins": 0, "losses": 1, "expired": 0, "ambiguous": 0}
    summary = tracker.summary()
    assert summary["resolved"] == 1
    assert summary["losses"] == 1
    assert summary["win_rate"] == 0.0


def test_forward_losses_lower_calibrated_confidence(tmp_path):
    tracker = OutcomeCalibrationAgent(
        str(tmp_path / "calibration.sqlite3"),
        bin_width=0.05,
        prior_strength=20,
    )
    base = pd.Timestamp("2026-08-30T15:00:00Z")
    for i in range(10):
        signal = TradeSignal(
            "XAU/USD", Direction.BUY, 100 + i * 0.01, 98, 102, 0.80,
            "trend_up", ["test"], [], selected_strategy="trend(5, 30, 0.0)",
        )
        tracker.record(signal, base + pd.Timedelta(minutes=i), selection_confidence=0.80)

    candles = pd.DataFrame({
        "datetime": [base + pd.Timedelta(minutes=11)],
        "high": [101.0],
        "low": [97.0],
    })
    tracker.resolve_open(candles)
    calibrated = tracker.calibrate(0.80, "trend(5, 30, 0.0)", "trend_up")
    assert calibrated.samples == 10
    assert calibrated.wins == 0
    assert calibrated.probability < 0.80
    assert calibrated.brier_score is not None


def test_no_forward_samples_leave_confidence_unchanged(tmp_path):
    tracker = OutcomeCalibrationAgent(str(tmp_path / "empty.sqlite3"))
    calibrated = tracker.calibrate(0.76, "trend(5, 30, 0.0)", "trend_up")
    assert calibrated.samples == 0
    assert calibrated.probability == 0.76


def test_research_runs_walk_forward_lifecycle_and_caps_catalog():
    lab = StrategyResearchAgent(
        max_candidates=150,
        spread_bps=1.0,
        slippage_bps=0.4,
        walk_forward_folds=3,
        catalog_size=80,
    )
    top = lab.run(sample_df())
    assert lab.last_evaluated == 150
    assert len(top) <= 25
    assert len(lab.catalog) <= 80
    assert lab.last_universe_size >= 25_000
    assert all(0 <= x.valid_hit_rate <= 1 for x in top)
    assert all(x.folds >= 2 for x in top)
    assert all(x.walk_forward_std >= 0 for x in top)
    assert all(np.isfinite(x.avg_r_multiple) for x in top)
    assert all(x.max_drawdown_r >= 0 for x in top)
    assert all(x.max_loss_streak >= 0 for x in top)
    assert all(x.backtest_model == "next_bar_atr" for x in top)


def test_large_budget_reaches_tens_of_thousands_and_spans_families():
    lab = StrategyResearchAgent(max_candidates=20_000)
    selected = lab._balanced_candidates()
    assert len(selected) == 20_000
    assert lab.last_universe_size >= 25_000
    families = {c.family for c in selected}
    assert families == {
        "trend", "triple_trend", "mean_reversion", "bollinger_reversion",
        "breakout", "momentum", "pullback", "volatility_breakout",
        "rsi_trend", "bollinger_breakout", "range_fade",
    }


def test_selector_chooses_researched_strategy_matching_market_and_analysts():
    df = trending_df()
    lab = StrategyResearchAgent(max_candidates=10)
    candidate = Candidate("trend", (5, 30, 0.0))
    lab.catalog = [CandidateScore(candidate, 0.66, 0.68, 180, 0.72)]
    votes = [
        AgentVote("Trend Analyst", Direction.BUY, 0.84, "uptrend"),
        AgentVote("Structure Analyst", Direction.BUY, 0.76, "higher highs"),
        AgentVote("Price Action Analyst", Direction.HOLD, 0.50, "neutral candle"),
    ]
    selector = StrategySelectorAgent(min_probability=0.40, min_agreement=2)
    pick = selector.select(df, "trend_up", votes, lab)
    assert pick is not None
    assert pick.direction == Direction.BUY
    assert pick.score.candidate == candidate
    assert pick.analyst_agreement == 2


def test_selector_prefers_strategy_with_better_current_regime_history():
    df = trending_df()
    lab = StrategyResearchAgent(max_candidates=10)
    strong = Candidate("trend", (5, 30, 0.0))
    weak = Candidate("trend", (8, 30, 0.0))
    lab.catalog = [
        CandidateScore(
            weak, 0.62, 0.62, 200, 0.72, walk_forward_hit_rate=0.62,
            profit_factor=1.4, folds=4, regime_scores={"trend_up": 0.46}, regime_trades={"trend_up": 60},
            avg_r_multiple=0.10, max_drawdown_r=4.0, max_loss_streak=5,
        ),
        CandidateScore(
            strong, 0.62, 0.62, 200, 0.72, walk_forward_hit_rate=0.62,
            profit_factor=1.4, folds=4, regime_scores={"trend_up": 0.72}, regime_trades={"trend_up": 60},
            avg_r_multiple=0.30, max_drawdown_r=2.0, max_loss_streak=3,
        ),
    ]
    votes = [
        AgentVote("Trend Analyst", Direction.BUY, 0.80, "uptrend"),
        AgentVote("Structure Analyst", Direction.BUY, 0.75, "higher highs"),
    ]
    pick = StrategySelectorAgent(min_probability=0.0, min_agreement=2).select(df, "trend_up", votes, lab)
    assert pick is not None
    assert pick.score.candidate == strong
    assert pick.regime_history > 0.5
    assert pick.lifecycle_quality > 0.5


def test_multi_timeframe_agent_creates_independent_horizon_votes():
    agent = MultiTimeframeAgent()
    frames = {
        "1min": trending_df(),
        "5min": trending_df(),
        "15min": trending_df(),
        "1h": trending_df(),
        "4h": trending_df(),
    }
    votes = agent.analyze(frames)
    assert len(votes) == 5
    assert all(v.direction == Direction.BUY for v in votes)
    assert {v.metadata["timeframe"] for v in votes} == {"1min", "5min", "15min", "1h", "4h"}


def test_selector_weights_higher_timeframes_more_than_1m_noise():
    selector = StrategySelectorAgent(min_probability=0.0, min_agreement=1)
    votes = [
        AgentVote("1m Execution Desk", Direction.SELL, 0.90, "noise", {"timeframe": "1min"}),
        AgentVote("1h Trend Desk", Direction.BUY, 0.75, "trend", {"timeframe": "1h"}),
        AgentVote("4h Macro Trend Desk", Direction.BUY, 0.72, "macro trend", {"timeframe": "4h"}),
    ]
    alignment = selector._timeframe_alignment(Direction.BUY, votes)
    assert alignment > 0.5


def test_news_risk_vetoes_inside_blackout_window():
    from datetime import datetime, timezone

    event = "2026-08-30T15:00:00Z"
    agent = NewsRiskAgent.from_csv(event, block_minutes=20)
    vote = agent.vote(datetime(2026, 8, 30, 15, 10, tzinfo=timezone.utc))
    assert vote.metadata["veto"] is True


def test_telegram_format_contains_trade_strategy_lifecycle_and_calibration_fields():
    s = TradeSignal(
        "XAU/USD", Direction.BUY, 2500, 2490, 2518, 0.74, "trend_up", ["test"], [],
        selected_strategy="trend(5, 30, 0.0)",
        strategy_stats={
            "valid_hit_rate": 0.68,
            "walk_forward_hit_rate": 0.67,
            "trades": 180,
            "folds": 4,
            "profit_factor": 1.58,
            "avg_r_multiple": 0.35,
            "max_drawdown_r": 2.4,
            "max_loss_streak": 3,
            "selection_confidence_raw": 0.80,
            "calibration_samples": 12,
            "calibration_brier_score": 0.19,
        },
    )
    text = TelegramNotifier("", "").format_signal(s)
    assert "Action: BUY" in text
    assert "Strategy: trend(5, 30, 0.0)" in text
    assert "executed trades" in text
    assert "Profit factor: 1.58" in text
    assert "Avg R: +0.35" in text
    assert "Max DD: 2.40R" in text
    assert "TP: 2518.00" in text
    assert "SL: 2490.00" in text
    assert "Selection confidence: 80.0%" in text
    assert "Forward-calibrated confidence: 74.0% from 12 resolved outcomes" in text
    assert "Calibration Brier score: 0.1900" in text
