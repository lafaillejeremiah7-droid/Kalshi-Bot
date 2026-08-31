import numpy as np
import pandas as pd

from xau_company import dashboard
from xau_company.adaptive_research import AdaptiveStrategyResearchAgent
from xau_company.models import Direction
from xau_company.research import Candidate, CandidateScore
from xau_company.selector import StrategyPick
from xau_company.strategy_invention import StrategyInventionAgent


def _df(n=420):
    close = np.linspace(2000.0, 2240.0, n)
    open_ = close - 0.4
    high = close + 0.8
    low = open_ - 0.8
    return pd.DataFrame({
        "datetime": pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
    })


def _score(candidate, score=0.80, **overrides):
    values = dict(
        candidate=candidate,
        train_hit_rate=0.66,
        valid_hit_rate=0.64,
        trades=180,
        score=score,
        walk_forward_hit_rate=0.64,
        walk_forward_std=0.03,
        expectancy=0.001,
        profit_factor=1.55,
        folds=4,
        regime_scores={"trend_up": 0.66},
        regime_trades={"trend_up": 80},
        avg_r_multiple=0.24,
        max_drawdown_r=3.0,
        max_loss_streak=4,
    )
    values.update(overrides)
    return CandidateScore(**values)


def test_invention_grammar_has_many_genuinely_distinct_family_structures():
    templates = StrategyInventionAgent.family_templates()
    assert len(templates) == 1980
    assert len(set(templates)) == len(templates)


def test_invention_bot_persists_new_families_and_parameter_variants(tmp_path):
    bot = StrategyInventionAgent(
        str(tmp_path / "invented.json"),
        families_per_cycle=2,
        variants_per_family=3,
        max_library_size=100,
    )
    families, variants = bot.invent()
    assert (families, variants) == (2, 6)
    assert bot.family_count() == 2
    assert bot.size() == 6
    assert all(row["family"] == "invented" for row in bot.entries())
    assert all(row["status"] == "EXPERIMENTAL" for row in bot.entries())

    # The persistent cursor creates the next structural families instead of
    # repeating or renaming the same formulas.
    families, variants = bot.invent()
    assert (families, variants) == (2, 6)
    assert bot.family_count() == 4
    assert bot.size() == 12
    keys = {bot.spec_key(row["family"], row["params"]) for row in bot.entries()}
    assert len(keys) == bot.size()


def test_invented_formula_engine_can_generate_a_direction_without_lookahead():
    params = (
        "INV-TEST",
        1,
        "majority",
        (
            ("ema_gap", 5, 30, 0.0),
            ("momentum", 3, 0.0),
            ("rsi_trend", 7, 50, 50),
        ),
        ("none",),
    )
    signal = StrategyInventionAgent.signal(_df(), params, cache={})
    assert int(signal.iloc[-1]) == 1
    assert set(signal.dropna().unique()).issubset({-1, 0, 1})


def test_malformed_invention_fails_closed_to_hold():
    signal = StrategyInventionAgent.signal(_df(), ("bad",), cache={})
    assert int(signal.abs().sum()) == 0


def test_experimental_invention_is_hidden_until_overfit_promotion(tmp_path):
    lab = AdaptiveStrategyResearchAgent(
        max_candidates=1000,
        strategy_library_path=str(tmp_path / "evolved.json"),
        discoveries_per_cycle=0,
        enable_invention=True,
        invention_library_path=str(tmp_path / "invented.json"),
        invented_families_per_cycle=1,
        invented_variants_per_family=1,
        invention_library_size=100,
    )
    assert lab.invention.invent() == (1, 1)
    family, params = lab.invention.candidate_specs()[0]
    invented = Candidate(family, params)
    seed = _score(Candidate("trend", (5, 30, 0.0)))
    invented_score = _score(invented, score=0.82)
    research_catalog = [seed, invented_score]

    lab._refresh_live_catalog(research_catalog)
    assert all(row.candidate.family != "invented" for row in lab.catalog)

    promoted, quarantined = lab.invention.audit_promotions(
        research_catalog,
        lab.overfit_auditor,
        tested_trials=20_000,
    )
    assert promoted == 1
    assert quarantined == 0
    lab._refresh_live_catalog(research_catalog)
    assert any(row.candidate == invented for row in lab.catalog)


def test_invention_quarantine_is_sticky_after_promoted_evidence_degrades(tmp_path):
    bot = StrategyInventionAgent(
        str(tmp_path / "invented.json"), families_per_cycle=1, variants_per_family=1, max_library_size=100
    )
    bot.invent()
    family, params = bot.candidate_specs()[0]
    candidate = Candidate(family, params)
    from xau_company.overfit import OverfitAuditor
    auditor = OverfitAuditor()

    assert bot.audit_promotions([_score(candidate, 0.82)], auditor, 20_000) == (1, 0)
    weak = _score(
        candidate,
        0.55,
        profit_factor=0.90,
        avg_r_multiple=-0.10,
        walk_forward_std=0.25,
        max_drawdown_r=14.0,
        max_loss_streak=10,
    )
    assert bot.audit_promotions([weak], auditor, 20_000) == (0, 1)
    assert bot.entries()[0]["status"] == "QUARANTINED"
    assert bot.audit_promotions([weak], auditor, 20_000) == (0, 0)
    assert bot.entries()[0]["status"] == "QUARANTINED"


def test_evolution_can_use_promoted_invention_as_non_recursive_parent(tmp_path):
    lab = AdaptiveStrategyResearchAgent(
        max_candidates=1000,
        strategy_library_path=str(tmp_path / "evolved.json"),
        discoveries_per_cycle=0,
        enable_invention=True,
        invention_library_path=str(tmp_path / "invented.json"),
        invented_families_per_cycle=0,
    )
    invented_params = (
        "INV-TEST", 1, "majority",
        (("ema_gap", 5, 30, 0.0), ("momentum", 3, 0.0), ("rsi_trend", 7, 50, 50)),
        ("none",),
    )
    ensemble = Candidate(
        "ensemble",
        ("invented", invented_params, "trend", (5, 30, 0.0), "confirm"),
    )
    signal = lab._signal(_df(), ensemble, cache={})
    assert int(signal.iloc[-1]) == 1


def test_invented_strategy_labels_are_compact_for_telegram_and_dashboard():
    params = (
        "INV-0001", 3, "majority",
        (("ema_gap", 5, 30, 0.0), ("momentum", 3, 0.0), ("rsi_trend", 7, 55, 45)),
        ("atr_normal", 14, 120, 1.8),
    )
    pick = StrategyPick(
        score=_score(Candidate("invented", params)),
        direction=Direction.BUY,
        probability_score=0.8,
        analyst_agreement=3,
        analyst_opposition=0,
        regime_fit=0.7,
    )
    assert "INV-0001" in pick.label
    assert len(pick.label) < 180


def test_dashboard_includes_strategy_invention_as_28th_employee():
    ids = [employee["id"] for employee in dashboard.EMPLOYEES]
    assert "strategy_invention" in ids
    assert len(ids) == 28
