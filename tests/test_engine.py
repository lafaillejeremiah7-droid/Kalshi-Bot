import numpy as np
import pandas as pd

from xau_company.models import Direction, TradeSignal
from xau_company.research import StrategyResearchAgent
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


def test_research_runs_and_caps_candidates():
    lab = StrategyResearchAgent(max_candidates=120, spread_bps=1.0)
    top = lab.run(sample_df())
    assert lab.last_evaluated == 120
    assert len(top) <= 25
    assert all(0 <= x.valid_hit_rate <= 1 for x in top)


def test_large_budget_really_reaches_thousands_and_is_balanced():
    lab = StrategyResearchAgent(max_candidates=3000)
    selected = lab._balanced_candidates()
    assert len(selected) == 3000
    families = {c.family for c in selected}
    assert families == {"trend", "mean_reversion", "breakout", "momentum"}


def test_telegram_format_contains_trade_fields():
    s = TradeSignal("XAU/USD", Direction.BUY, 2500, 2490, 2518, 0.8, "trend_up", ["test"], [])
    text = TelegramNotifier("", "").format_signal(s)
    assert "Action: BUY" in text
    assert "TP: 2518.00" in text
    assert "SL: 2490.00" in text
    assert "Confidence: 80.0%" in text
