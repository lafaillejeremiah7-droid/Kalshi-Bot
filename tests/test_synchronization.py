from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from xau_company.adaptive_research import AdaptiveStrategyResearchAgent
from xau_company.agents import SessionAgent
from xau_company.config import Settings
from xau_company.models import AgentVote, Direction, TradeSignal
from xau_company.orchestrator import BossAgent
from xau_company.outcomes import OutcomeCalibrationAgent
from xau_company.overfit import OverfitAuditor
from xau_company.quality import MarketDataQualityAgent
from xau_company.research import Candidate, CandidateScore, StrategyResearchAgent
from xau_company.selector import StrategySelectorAgent
from xau_company.strategy_evolution import StrategyEvolutionAgent


def _df(n=500, freq="15min", start="2026-08-31T07:00:00Z"):
    close = np.linspace(2000, 2200, n)
    open_ = close - 0.2
    high = close + 0.8
    low = open_ - 0.8
    return pd.DataFrame(
        {
            "datetime": pd.date_range(start, periods=n, freq=freq, tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def _score(candidate: Candidate, **overrides) -> CandidateScore:
    values = dict(
        candidate=candidate,
        train_hit_rate=0.64,
        valid_hit_rate=0.63,
        trades=300,
        score=0.78,
        walk_forward_hit_rate=0.63,
        walk_forward_std=0.03,
        expectancy=0.001,
        profit_factor=1.5,
        folds=4,
        regime_scores={"trend_up": 0.65},
        regime_trades={"trend_up": 80},
        avg_r_multiple=0.25,
        max_drawdown_r=3.0,
        max_loss_streak=4,
    )
    values.update(overrides)
    return CandidateScore(**values)


def _signal(entry=2500.0, strategy="trend(5, 30, 0.0)"):
    return TradeSignal(
        "XAU/USD",
        Direction.BUY,
        entry,
        entry - 10,
        entry + 17,
        0.80,
        "trend_up",
        ["test"],
        [],
        selected_strategy=strategy,
        strategy_stats={"max_holding_minutes": 90, "resolution_interval_minutes": 1},
    )


def test_quality_guard_drops_forming_candle_and_rejects_bad_geometry():
    quality = MarketDataQualityAgent(max_stale_multiplier=4.0)
    now = pd.Timestamp("2026-08-31T15:07:00Z")
    frame = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2026-08-31T14:55:00Z"), pd.Timestamp("2026-08-31T15:05:00Z")],
            "open": [2000.0, 2001.0],
            "high": [2002.0, 2003.0],
            "low": [1999.0, 2000.0],
            "close": [2001.0, 2002.0],
        }
    )
    cleaned, report = quality.clean_frame(frame, "5min", now=now, require_fresh=True)
    assert report.ok is True
    assert len(cleaned) == 1
    assert cleaned.datetime.iloc[-1] == pd.Timestamp("2026-08-31T14:55:00Z")

    bad = frame.iloc[:1].copy()
    bad.loc[0, "high"] = 1990.0
    cleaned, report = quality.clean_frame(bad, "5min", now=now, require_fresh=False)
    assert report.ok is False
    assert cleaned.empty


def test_selector_does_not_double_count_timeframe_or_macro_votes_as_analysts():
    selector = StrategySelectorAgent(min_probability=0.0, min_agreement=2)
    votes = [
        AgentVote("Trend Desk", Direction.BUY, 0.8, "trend"),
        AgentVote("4h Macro Trend Desk", Direction.BUY, 0.9, "4h", {"timeframe": "4h"}),
        AgentVote("USD Strength Desk", Direction.BUY, 0.7, "usd"),
    ]
    _, _, agreeing, _ = selector._analyst_support(Direction.BUY, votes)
    assert agreeing == 1


def test_overfit_minimum_trade_gate_uses_oos_sample_not_total_backtest_trades():
    candidate = Candidate("ensemble", ("trend", (), "momentum", (), "confirm"))
    score = _score(candidate, trades=500, regime_trades={"trend_up": 12, "range": 8})
    result = OverfitAuditor(min_trades=40).audit(score, tested_trials=20_000)
    assert result.passed is False
    assert any("OOS trade sample 20" in reason for reason in result.reasons)


def test_boss_uses_exact_requested_consensus_not_off_by_one():
    lab = StrategyResearchAgent(max_candidates=1000)
    boss = BossAgent(lab, min_consensus=3)
    assert boss.selector.min_agreement == 3


def test_session_desk_vetoes_outside_approved_liquidity_window():
    frame = _df(n=250, start="2026-08-30T00:00:00Z")
    frame.loc[frame.index[-1], "datetime"] = pd.Timestamp("2026-08-31T03:00:00Z")
    vote = SessionAgent().vote(frame, "trend_up", StrategyResearchAgent(max_candidates=1000))
    assert vote.metadata["liquidity_ok"] is False
    assert vote.metadata["veto"] is True


def test_boss_live_risk_geometry_matches_research_and_uses_live_price():
    lab = StrategyResearchAgent(
        max_candidates=1000,
        backtest_stop_atr=1.2,
        backtest_reward_risk=1.7,
    )
    candidate = Candidate("trend", (5, 30, 0.0))
    lab.catalog = [_score(candidate)]
    lab.top = list(lab.catalog)
    lab.family_quality = {"trend": 0.80, "momentum": 0.80, "breakout": 0.80}

    core = _df(n=500, freq="15min", start="2026-08-26T07:00:00Z")
    frames = {
        "1min": _df(n=500, freq="1min", start="2026-08-31T07:00:00Z"),
        "5min": _df(n=500, freq="5min", start="2026-08-29T07:00:00Z"),
        "15min": core,
        "1h": _df(n=500, freq="1h", start="2026-08-10T07:00:00Z"),
        "4h": _df(n=500, freq="4h", start="2026-06-09T07:00:00Z"),
    }
    # Keep the Session Desk inside its 07:00-17:00 UTC window.
    for frame in frames.values():
        frame.loc[frame.index[-1], "datetime"] = pd.Timestamp("2026-08-31T15:00:00Z")

    boss = BossAgent(lab, min_confidence=0.0, min_consensus=2, research_interval="15min")
    signal = boss.decide("XAU/USD", frames, entry_price=2500.0)
    assert signal is not None
    assert signal.entry == 2500.0
    risk = abs(signal.entry - signal.stop_loss)
    assert signal.risk_reward == pytest.approx(lab.backtester.reward_risk)
    assert signal.strategy_stats["stop_atr"] == lab.backtester.stop_atr
    assert signal.strategy_stats["max_holding_minutes"] == 6 * 15
    assert risk > 0


def test_setup_dedupe_is_stable_when_entry_sl_tp_refresh(tmp_path):
    tracker = OutcomeCalibrationAgent(str(tmp_path / "dedupe.sqlite3"))
    setup = pd.Timestamp("2026-08-31T15:00:00Z")
    first = _signal(2500.0)
    second = _signal(2501.5)
    assert tracker.record(first, setup, setup_at=setup) is True
    assert tracker.record(second, setup, setup_at=setup) is False


def test_daily_slot_reservation_is_atomic_and_counts_reserved_signal(tmp_path):
    tracker = OutcomeCalibrationAgent(str(tmp_path / "reserve.sqlite3"))
    now = pd.Timestamp("2026-08-31T15:00:00Z")
    start = pd.Timestamp("2026-08-31T05:00:00Z")
    end = pd.Timestamp("2026-09-01T05:00:00Z")
    first = tracker.reserve_if_under_cap(_signal(), now, 0.8, now, start, end, 1)
    second = tracker.reserve_if_under_cap(
        _signal(strategy="breakout(20, 0.0, 5)"),
        now + pd.Timedelta(minutes=5),
        0.8,
        now + pd.Timedelta(minutes=5),
        start,
        end,
        1,
    )
    assert first.reserved is True
    assert second.reserved is False
    assert tracker.count_emitted_between(start, end) == 1


def test_crashed_reservation_becomes_unknown_and_cannot_duplicate(tmp_path):
    path = str(tmp_path / "crash.sqlite3")
    now = pd.Timestamp("2026-08-31T15:00:00Z")
    start = pd.Timestamp("2026-08-31T05:00:00Z")
    end = pd.Timestamp("2026-09-01T05:00:00Z")
    tracker = OutcomeCalibrationAgent(path)
    signal = _signal()
    assert tracker.reserve_if_under_cap(signal, now, 0.8, now, start, end, 2).reserved

    restarted = OutcomeCalibrationAgent(path)
    assert restarted.exists(signal, now) is True
    assert restarted.count_emitted_between(start, end) == 1


def test_partial_emission_candle_is_marked_ambiguous_not_win_or_loss(tmp_path):
    tracker = OutcomeCalibrationAgent(str(tmp_path / "partial.sqlite3"))
    observed = pd.Timestamp("2026-08-31T15:00:30Z")
    signal = _signal(100.0)
    signal.stop_loss = 98.0
    signal.take_profit = 102.0
    assert tracker.record(signal, observed, setup_at=pd.Timestamp("2026-08-31T15:00:00Z"))
    candles = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2026-08-31T15:00:00Z")],
            "high": [103.0],
            "low": [99.0],
        }
    )
    resolved = tracker.resolve_open(candles, interval_minutes=1)
    assert resolved["ambiguous"] == 1
    summary = tracker.summary()
    assert summary["resolved"] == 0
    assert summary["ambiguous"] == 1


def test_invalid_zero_schedule_is_rejected():
    cfg = Settings(
        research_every_cycles=0,
    )
    with pytest.raises(ValueError, match="RESEARCH_EVERY_CYCLES"):
        cfg.validate()


def test_lifetime_trial_count_persists_and_accumulates(tmp_path):
    path = str(tmp_path / "strategies.json")
    first = StrategyEvolutionAgent(path, discoveries_per_cycle=0, max_library_size=100)
    assert first.increment_tested_trials(20_000) == 20_000
    second = StrategyEvolutionAgent(path, discoveries_per_cycle=0, max_library_size=100)
    assert second.increment_tested_trials(20_000) == 40_000
    assert second.tested_trials_lifetime() == 40_000


def test_quarantined_history_is_not_evicted_by_library_cap():
    bot = StrategyEvolutionAgent(":memory:", discoveries_per_cycle=0, max_library_size=100)
    rows = [
        {"family": "ensemble", "params": ["q", i], "status": "QUARANTINED"}
        for i in range(110)
    ] + [
        {"family": "ensemble", "params": ["e", i], "status": "EXPERIMENTAL"}
        for i in range(40)
    ]
    bot._write(rows)
    saved = bot.entries()
    assert len(saved) == 110
    assert all(row["status"] == "QUARANTINED" for row in saved)


def test_two_fold_configuration_does_not_create_impossible_three_fold_promotion_gate(tmp_path):
    lab = AdaptiveStrategyResearchAgent(
        max_candidates=1000,
        walk_forward_folds=2,
        min_walk_forward_folds=2,
        strategy_library_path=str(tmp_path / "strategies.json"),
        discoveries_per_cycle=0,
    )
    assert lab.overfit_auditor.min_folds == 2
