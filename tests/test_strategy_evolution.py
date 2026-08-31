import numpy as np
import pandas as pd

from xau_company.adaptive_research import AdaptiveStrategyResearchAgent
from xau_company.research import Candidate, CandidateScore, StrategyResearchAgent
from xau_company.strategy_evolution import StrategyEvolutionAgent


def _score(candidate: Candidate, score: float = 0.75) -> CandidateScore:
    return CandidateScore(
        candidate=candidate,
        train_hit_rate=0.64,
        valid_hit_rate=0.63,
        trades=120,
        score=score,
        walk_forward_hit_rate=0.63,
        profit_factor=1.45,
        folds=4,
        avg_r_multiple=0.22,
        max_drawdown_r=3.0,
        max_loss_streak=4,
    )


def _trending_df(n: int = 300) -> pd.DataFrame:
    close = np.linspace(2000, 2200, n)
    open_ = close - 0.2
    high = close + 0.8
    low = open_ - 0.8
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def test_evolution_bot_creates_persistent_cross_family_experiments(tmp_path):
    path = tmp_path / "strategies.json"
    bot = StrategyEvolutionAgent(str(path), discoveries_per_cycle=20, max_library_size=100)
    parents = [
        _score(Candidate("trend", (5, 30, 0.0)), 0.80),
        _score(Candidate("momentum", (5, 0.0004, 50)), 0.77),
        _score(Candidate("mean_reversion", (14, 30, 70, 1.0, 30)), 0.73),
    ]

    added = bot.propose(parents)
    assert added > 0
    entries = bot.entries()
    assert len(entries) == added
    assert all(row["family"] == "ensemble" for row in entries)
    assert all(row["status"] == "EXPERIMENTAL" for row in entries)
    assert all(len(row["parents"]) == 2 for row in entries)

    # A fresh bot instance reads the same persistent library and does not add
    # duplicates when given the same parent set again.
    restarted = StrategyEvolutionAgent(str(path), discoveries_per_cycle=20, max_library_size=100)
    before = restarted.size()
    restarted.propose(parents)
    assert restarted.size() >= before
    keys = {
        restarted.spec_key(row["family"], row["params"])
        for row in restarted.entries()
    }
    assert len(keys) == restarted.size()


def test_adaptive_universe_grows_beyond_fixed_seed_count(tmp_path):
    path = tmp_path / "strategies.json"
    adaptive = AdaptiveStrategyResearchAgent(
        max_candidates=20_000,
        strategy_library_path=str(path),
        discoveries_per_cycle=10,
        discovery_library_size=100,
    )
    parents = [
        _score(Candidate("trend", (5, 30, 0.0))),
        _score(Candidate("breakout", (20, 0.0002, 5))),
    ]
    adaptive.evolution.propose(parents)

    fixed_count = sum(1 for _ in StrategyResearchAgent(max_candidates=20_000).candidates())
    adaptive_count = sum(1 for _ in adaptive.candidates())
    assert fixed_count >= 27_000
    assert adaptive_count > fixed_count
    assert any(c.family == "ensemble" for c in adaptive.candidates())


def test_evolved_confirm_ensemble_uses_existing_signal_engine(tmp_path):
    adaptive = AdaptiveStrategyResearchAgent(
        max_candidates=1000,
        strategy_library_path=str(tmp_path / "strategies.json"),
        discoveries_per_cycle=0,
    )
    candidate = Candidate(
        "ensemble",
        (
            "trend",
            (5, 30, 0.0),
            "momentum",
            (5, 0.0001, 20),
            "confirm",
        ),
    )
    signal = adaptive._signal(_trending_df(), candidate, cache={})
    assert int(signal.iloc[-1]) == 1
