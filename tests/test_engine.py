import numpy as np
import pandas as pd

from xau_company.models import AgentVote, Direction, TradeSignal
from xau_company.research import Candidate, CandidateScore, StrategyResearchAgent
from xau_company.selector import StrategySelectorAgent
from xau_company.telegram import TelegramNotifier


def sample_df(n=600):
    rng = np.random.default_rng(7)
    close = 2000 + np.cumsum(rng.normal(0.2, 2.0, n))
    open_ = close + rng.normal(0, 0.8, n)
    high = np.maximum(open_, close) + rng.uniform(0.2, 2.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.2, 2.0, n)
    return pd.DataFrame({
        "datetime": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
    })


def trending_df(n=400):
    close = np.linspace(2000, 2200, n)
    open_ = close - 0.4
    high = close + 1.0
    low = open_ - 1.0
    return pd.DataFrame({
        "datetime": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
    })


def test_research_runs_and_caps_candidates():
    lab = StrategyResearchAgent(max_candidates=120, spread_bps=1.0)
    top = lab.run(sample_df())
    assert lab.last_evaluated == 120
    assert len(top) <= 25
    assert len(lab.catalog) <= 300
    assert all(0 <= x.valid_hit_rate <= 1 for x in top)


def test_large_budget_really_reaches_thousands_and_is_balanced():
    lab = StrategyResearchAgent(max_candidates=3000)
    selected = lab._balanced_candidates()
    assert len(selected) == 3000
    families = {c.family for c in selected}
    assert families == {"trend", "mean_reversion", "breakout", "momentum"}


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


def test_telegram_format_contains_trade_and_strategy_fields():
    s = TradeSignal(
        "XAU/USD", Direction.BUY, 2500, 2490, 2518, 0.8, "trend_up", ["test"], [],
        selected_strategy="trend(5, 30, 0.0)",
        strategy_stats={"valid_hit_rate": 0.68, "trades": 180},
    )
    text = TelegramNotifier("", "").format_signal(s)
    assert "Action: BUY" in text
    assert "Strategy: trend(5, 30, 0.0)" in text
    assert "Validation: 68.0% over 180 historical signals" in text
    assert "TP: 2518.00" in text
    assert "SL: 2490.00" in text
    assert "Selection confidence: 80.0%" in text
