import numpy as np
import pandas as pd

from xau_company.adaptive_research import AdaptiveStrategyResearchAgent
from xau_company.overfit import OverfitAuditor
from xau_company.research import Candidate, CandidateScore, StrategyResearchAgent
from xau_company.strategy_evolution import StrategyEvolutionAgent


def _score(candidate: Candidate, score: float = 0.75, **overrides) -> CandidateScore:
    values = {
        "candidate": candidate,
        "train_hit_rate": 0.64,
        "valid_hit_rate": 0.63,
        "trades": 120,
        "score": score,
        "walk_forward_hit_rate": 0.63,
        "walk_forward_std": 0.03,
        "profit_factor": 1.45,
        "folds": 4,
        "avg_r_multiple": 0.22,
        "max_drawdown_r": 3.0,
        "max_loss_streak": 4,
    }
    values.update(overrides)
    return CandidateScore(**values)


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


def test_overfit_penalty_gets_stricter_as_more_variants_are_tested():
    auditor = OverfitAuditor()
    strong = _score(Candidate("ensemble", ("x", (), "y", (), "confirm")), score=0.78)
    small_search = auditor.audit(strong, tested_trials=100)
    large_search = auditor.audit(strong, tested_trials=20_000)

    assert large_search.multiplicity_penalty > small_search.multiplicity_penalty
    assert large_search.adjusted_score < small_search.adjusted_score
    assert large_search.passed is True

    weak = _score(
        strong.candidate,
        score=0.68,
        profit_factor=1.05,
        avg_r_multiple=0.01,
        max_drawdown_r=12.0,
    )
    rejected = auditor.audit(weak, tested_trials=20_000)
    assert rejected.passed is False
    assert len(rejected.reasons) >= 2


def test_experimental_strategy_is_hidden_until_overfit_audit_promotes_it(tmp_path):
    path = tmp_path / "strategies.json"
    adaptive = AdaptiveStrategyResearchAgent(
        max_candidates=1000,
        strategy_library_path=str(path),
        discoveries_per_cycle=10,
        discovery_library_size=100,
    )
    parents = [
        _score(Candidate("trend", (5, 30, 0.0))),
        _score(Candidate("breakout", (20, 0.0002, 5))),
    ]
    assert adaptive.evolution.propose(parents) > 0
    family, params = adaptive.evolution.candidate_specs()[0]
    evolved = Candidate(family, params)
    seed = _score(Candidate("trend", (5, 30, 0.0)), score=0.80)
    evolved_score = _score(evolved, score=0.78)
    research_catalog = [seed, evolved_score]

    adaptive._refresh_live_catalog(research_catalog)
    assert all(result.candidate.family != "ensemble" for result in adaptive.catalog)

    promoted, quarantined = adaptive.evolution.audit_promotions(
        research_catalog,
        adaptive.overfit_auditor,
        tested_trials=20_000,
    )
    assert promoted == 1
    assert quarantined == 0
    adaptive._refresh_live_catalog(research_catalog)
    assert any(result.candidate == evolved for result in adaptive.catalog)


def test_promoted_strategy_is_quarantined_when_evidence_deteriorates(tmp_path):
    path = tmp_path / "strategies.json"
    bot = StrategyEvolutionAgent(str(path), discoveries_per_cycle=10, max_library_size=100)
    parents = [
        _score(Candidate("trend", (5, 30, 0.0))),
        _score(Candidate("momentum", (5, 0.0004, 50))),
    ]
    bot.propose(parents)
    family, params = bot.candidate_specs()[0]
    candidate = Candidate(family, params)
    auditor = OverfitAuditor()

    strong = _score(candidate, score=0.80)
    promoted, quarantined = bot.audit_promotions([strong], auditor, tested_trials=20_000)
    assert promoted == 1
    assert quarantined == 0
    assert bot.spec_key(candidate.family, candidate.params) in bot.promoted_keys()

    degraded = _score(
        candidate,
        score=0.60,
        profit_factor=0.95,
        avg_r_multiple=-0.04,
        walk_forward_std=0.20,
        max_drawdown_r=14.0,
        max_loss_streak=10,
    )
    promoted, quarantined = bot.audit_promotions([degraded], auditor, tested_trials=20_000)
    assert promoted == 0
    assert quarantined == 1
    assert bot.spec_key(candidate.family, candidate.params) not in bot.promoted_keys()
    row = bot.entries()[0]
    assert row["status"] == "QUARANTINED"
    assert row["overfit_passed"] is False
