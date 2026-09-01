import json

import numpy as np
import pandas as pd

from xau_company.canonical_strategies import BY_ID, STRATEGIES
from xau_company.canonical_strategy_engine import CanonicalSignalEngine
from xau_company.research import Candidate, StrategyResearchAgent


def _frame(n=900):
    rng = np.random.default_rng(17)
    close = 2000 + np.cumsum(rng.normal(0.03, 2.1, n))
    open_ = close + rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + rng.uniform(0.2, 1.8, n)
    low = np.minimum(open_, close) - rng.uniform(0.2, 1.8, n)
    volume = rng.integers(100, 2000, n)
    return pd.DataFrame({
        "datetime": pd.date_range("2021-01-01", periods=n, freq="15min", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


def test_catalog_is_exactly_437_unique_methodologies():
    assert len(STRATEGIES) == 437
    assert len(BY_ID) == 437
    assert len({s.name for s in STRATEGIES}) == 437
    assert [s.strategy_id for s in STRATEGIES] == [f"S{i:03d}" for i in range(1, 438)]


def test_canonical_candidate_model_has_exactly_one_id_per_methodology(monkeypatch):
    monkeypatch.setenv("XAU_RESEARCH_USE_ALL_437", "1")
    lab = StrategyResearchAgent(max_candidates=437, catalog_size=109)
    candidates = list(lab.candidates())
    assert len(candidates) == 437
    assert len({c.strategy_id for c in candidates}) == 437
    assert not hasattr(candidates[0], "params")
    assert lab.last_universe_size == 437
    assert lab.catalog_size == 109


def test_live_research_is_fail_closed_until_survivors_exist(monkeypatch, tmp_path):
    monkeypatch.delenv("XAU_RESEARCH_USE_ALL_437", raising=False)
    path = tmp_path / "survivors.json"
    path.write_text(json.dumps({"survivors": []}))
    monkeypatch.setattr(StrategyResearchAgent, "SURVIVOR_FILE", path)
    lab = StrategyResearchAgent()
    assert list(lab.candidates()) == []


def test_survivor_file_cannot_promote_more_than_109(monkeypatch, tmp_path):
    monkeypatch.delenv("XAU_RESEARCH_USE_ALL_437", raising=False)
    path = tmp_path / "survivors.json"
    path.write_text(json.dumps({"survivors": [{"strategy_id": s.strategy_id} for s in STRATEGIES]}))
    monkeypatch.setattr(StrategyResearchAgent, "SURVIVOR_FILE", path)
    lab = StrategyResearchAgent(max_candidates=437)
    assert len(list(lab.candidates())) == 109


def test_supported_canonical_signals_are_causal_prefix_stable():
    frame = _frame(1000)
    cut = 760
    full_engine = CanonicalSignalEngine(frame)
    prefix_engine = CanonicalSignalEngine(frame.iloc[:cut].copy())
    checked = 0
    for strategy in STRATEGIES:
        if any(req not in {"xau_ohlc", "volume"} for req in strategy.requires):
            continue
        full = full_engine.signal(strategy).iloc[:cut]
        prefix = prefix_engine.signal(strategy)
        np.testing.assert_array_equal(full.iloc[250:].to_numpy(), prefix.iloc[250:].to_numpy())
        assert set(np.unique(full.dropna().to_numpy())).issubset({-1, 0, 1})
        checked += 1
    assert checked >= 350


def test_candidate_constructor_has_no_parameter_slot():
    try:
        Candidate("S001", (1, 2, 3))
    except TypeError:
        pass
    else:
        raise AssertionError("canonical Candidate must accept only one strategy ID")
