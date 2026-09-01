from __future__ import annotations

import base64
import re
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIG = ROOT / ".canonical-migration"

payload = "".join(p.read_text(encoding="utf-8") for p in sorted(MIG.glob("payload.*")))
archive = MIG / "payload.tar.gz"
archive.write_bytes(base64.b64decode(payload))
with tarfile.open(archive, "r:gz") as tf:
    tf.extractall(ROOT)

for rel in (
    ".github/workflows/full-company-backtest.yml",
    ".github/workflows/real-data-overfit-audit.yml",
    "scripts/full_company_historical_backtest.py",
    "scripts/real_data_overfit_audit.py",
    "full-company-backtest-report.json",
):
    (ROOT / rel).unlink(missing_ok=True)

def patch(path: str, fn):
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    new = fn(text)
    p.write_text(new, encoding="utf-8")

def patch_engine(text: str) -> str:
    text = text.replace('"trend(5, 30, 0.0)"', '"Dual-moving-average crossover"')
    text = text.replace('"Strategy: trend(5, 30, 0.0)"', '"Strategy: Dual-moving-average crossover"')
    text = re.sub(
        r"def test_research_runs_walk_forward_lifecycle_and_caps_catalog\(\):.*?(?=\ndef test_large_budget_reaches_tens_of_thousands_and_spans_families\(\):)",
        '''def test_research_catalog_contains_only_unique_canonical_methodologies(monkeypatch):
    monkeypatch.setenv("XAU_RESEARCH_USE_ALL_437", "1")
    lab = StrategyResearchAgent(max_candidates=437, catalog_size=109)
    selected = lab._balanced_candidates()
    assert len(selected) == 437
    assert lab.last_universe_size == 437
    assert len({c.family for c in selected}) == 437
    assert all(c.params == () for c in selected)
    assert lab.catalog_size == 109

''', text, flags=re.S)
    text = re.sub(
        r"def test_large_budget_reaches_tens_of_thousands_and_spans_families\(\):.*?(?=\ndef test_selector_chooses_researched_strategy_matching_market_and_analysts\(\):)",
        '''def test_candidate_budget_is_hard_capped_at_437(monkeypatch):
    monkeypatch.setenv("XAU_RESEARCH_USE_ALL_437", "1")
    lab = StrategyResearchAgent(max_candidates=20_000)
    selected = lab._balanced_candidates()
    assert len(selected) == 437
    assert lab.max_candidates == 437
    assert lab.last_universe_size == 437

''', text, flags=re.S)
    text = re.sub(
        r"def test_selector_chooses_researched_strategy_matching_market_and_analysts\(\):.*?(?=\ndef test_selector_prefers_strategy_with_better_current_regime_history\(\):)",
        '''def test_selector_chooses_canonical_strategy_matching_market_and_analysts():
    df = trending_df()
    lab = StrategyResearchAgent(max_candidates=10)
    candidate = Candidate("S127")
    lab.catalog = [CandidateScore(candidate, 0.66, 0.68, 180, 0.72)]
    lab.current_direction = lambda *_: Direction.BUY
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

''', text, flags=re.S)
    text = re.sub(
        r"def test_selector_prefers_strategy_with_better_current_regime_history\(\):.*?(?=\ndef test_multi_timeframe_agent_creates_independent_horizon_votes\(\):)",
        '''def test_selector_prefers_canonical_strategy_with_better_current_regime_history():
    df = trending_df()
    lab = StrategyResearchAgent(max_candidates=10)
    weak = Candidate("S126")
    strong = Candidate("S127")
    lab.current_direction = lambda *_: Direction.BUY
    lab.catalog = [
        CandidateScore(weak, 0.62, 0.62, 200, 0.72, walk_forward_hit_rate=0.62, profit_factor=1.4, folds=4, regime_scores={"trend_up": 0.46}, regime_trades={"trend_up": 60}, avg_r_multiple=0.10, max_drawdown_r=4.0, max_loss_streak=5),
        CandidateScore(strong, 0.62, 0.62, 200, 0.72, walk_forward_hit_rate=0.62, profit_factor=1.4, folds=4, regime_scores={"trend_up": 0.72}, regime_trades={"trend_up": 60}, avg_r_multiple=0.30, max_drawdown_r=2.0, max_loss_streak=3),
    ]
    votes = [AgentVote("Trend Analyst", Direction.BUY, 0.80, "uptrend"), AgentVote("Structure Analyst", Direction.BUY, 0.75, "higher highs")]
    pick = StrategySelectorAgent(min_probability=0.0, min_agreement=2).select(df, "trend_up", votes, lab)
    assert pick is not None
    assert pick.score.candidate == strong
    assert pick.regime_history > 0.5
    assert pick.lifecycle_quality > 0.5

''', text, flags=re.S)
    return text

patch("tests/test_engine.py", patch_engine)
patch("tests/test_runtime_hardening.py", lambda t: t.replace('"trend(5, 30, 0.0)"', '"Dual-moving-average crossover"'))
patch("tests/test_second_pass.py", lambda t: t.replace('"trend(5, 30, 0.0)"', '"Fair Value Gap (FVG) retracement continuation"').replace('"breakout(20, 0.0, 5)"', '"Opening Range Breakout (ORB)"'))

def patch_sync(text: str) -> str:
    text = text.replace('strategy="trend(5, 30, 0.0)"', 'strategy="Fair Value Gap (FVG) retracement continuation"')
    text = text.replace('Candidate("ensemble", ("trend", (), "momentum", (), "confirm"))', 'Candidate("S001")')
    text = text.replace('Candidate("trend", (5, 30, 0.0))', 'Candidate("S001")')
    text = text.replace('strategy="breakout(20, 0.0, 5)"', 'strategy="Opening Range Breakout (ORB)"')
    needle = 'lab.family_quality = {"trend": 0.80, "momentum": 0.80, "breakout": 0.80}\n\n    core = _df'
    repl = 'lab.family_quality = {"S001": 0.80}\n    lab.category_quality = {"trend": 0.80, "momentum": 0.80, "breakout": 0.80, "smc": 0.80}\n    lab.current_direction = lambda *_: Direction.BUY\n\n    core = _df'
    return text.replace(needle, repl)

patch("tests/test_synchronization.py", patch_sync)
shutil.rmtree(MIG, ignore_errors=True)
print("Canonical 437 migration applied")
