import numpy as np
import pandas as pd

from xau_company.context import MultiTimeframeAgent, NewsRiskAgent
from xau_company.models import AgentVote, Direction, TradeSignal
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


def test_research_runs_walk_forward_and_caps_catalog():
    lab = StrategyResearchAgent(max_candidates=150, spread_bps=1.0, walk_forward_folds=3, catalog_size=80)
    top = lab.run(sample_df())
    assert lab.last_evaluated == 150
    assert len(top) <= 25
    assert len(lab.catalog) <= 80
    assert lab.last_universe_size >= 25_000
    assert all(0 <= x.valid_hit_rate <= 1 for x in top)
    assert all(x.folds >= 2 for x in top)
    assert all(x.walk_forward_std >= 0 for x in top)


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
        ),
        CandidateScore(
            strong, 0.62, 0.62, 200, 0.72, walk_forward_hit_rate=0.62,
            profit_factor=1.4, folds=4, regime_scores={"trend_up": 0.72}, regime_trades={"trend_up": 60},
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


def test_telegram_format_contains_trade_and_strategy_fields():
    s = TradeSignal(
        "XAU/USD", Direction.BUY, 2500, 2490, 2518, 0.8, "trend_up", ["test"], [],
        selected_strategy="trend(5, 30, 0.0)",
        strategy_stats={"valid_hit_rate": 0.68, "walk_forward_hit_rate": 0.67, "trades": 180},
    )
    text = TelegramNotifier("", "").format_signal(s)
    assert "Action: BUY" in text
    assert "Strategy: trend(5, 30, 0.0)" in text
    assert "TP: 2518.00" in text
    assert "SL: 2490.00" in text
    assert "Selection confidence: 80.0%" in text
