from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .backtest import TradeLifecycleBacktester, TradeOutcome
from .canonical_strategies import BY_ID, STRATEGIES
from .canonical_strategy_engine import CanonicalSignalEngine
from .indicators import atr
from .models import Direction


@dataclass(frozen=True)
class Candidate:
    """Exactly one canonical strategy methodology, identified only by immutable ID."""
    strategy_id: str


@dataclass
class CandidateScore:
    candidate: Candidate
    train_hit_rate: float
    valid_hit_rate: float
    trades: int
    score: float
    walk_forward_hit_rate: float = 0.0
    walk_forward_std: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 1.0
    folds: int = 0
    regime_scores: dict[str, float] = field(default_factory=dict)
    regime_trades: dict[str, int] = field(default_factory=dict)
    avg_r_multiple: float = 0.0
    max_drawdown_r: float = 0.0
    max_loss_streak: int = 0
    backtest_model: str = "next_bar_atr"


class StrategyResearchAgent:
    """Research only canonical, pre-audited strategies.

    A strategy is one methodology with one immutable ID. The research model has
    no parameter-variant candidate type and can therefore represent each methodology once.

    The live engine is fail-closed: until the historical audit publishes
    ``surviving_strategies.json``, no strategy is eligible for live research.
    """

    SURVIVOR_FILE = Path(__file__).with_name("surviving_strategies.json")
    HORIZONS = {s.strategy_id: s.horizon for s in STRATEGIES}

    def __init__(
        self,
        max_candidates: int = 437,
        spread_bps: float = 1.5,
        walk_forward_folds: int = 4,
        catalog_size: int = 109,
        min_walk_forward_folds: int = 2,
        slippage_bps: float = 0.5,
        backtest_stop_atr: float = 1.20,
        backtest_reward_risk: float = 1.70,
    ) -> None:
        self.max_candidates = min(437, max(0, int(max_candidates)))
        self.spread_bps = spread_bps
        self.walk_forward_folds = max(2, int(walk_forward_folds))
        self.catalog_size = min(109, max(1, int(catalog_size)))
        self.min_walk_forward_folds = max(2, int(min_walk_forward_folds))
        self.backtester = TradeLifecycleBacktester(
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            stop_atr=backtest_stop_atr,
            reward_risk=backtest_reward_risk,
        )
        self.family_quality: dict[str, float] = {}
        self.category_quality: dict[str, float] = {}
        self.top: list[CandidateScore] = []
        self.catalog: list[CandidateScore] = []
        self.last_evaluated = 0
        self.last_universe_size = 0

    def _survivor_ids(self) -> list[str]:
        if os.getenv("XAU_RESEARCH_USE_ALL_437", "").strip().lower() in {"1", "true", "yes"}:
            return [s.strategy_id for s in STRATEGIES]
        try:
            payload = json.loads(self.SURVIVOR_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        rows = payload.get("survivors", payload if isinstance(payload, list) else [])
        ids: list[str] = []
        for row in rows:
            sid = row if isinstance(row, str) else row.get("strategy_id")
            if sid in BY_ID and sid not in ids:
                ids.append(sid)
        # Never allow an audit artifact to promote more than the requested top 25%.
        return ids[:109]

    def candidates(self) -> Iterable[Candidate]:
        ids = self._survivor_ids()
        self.last_universe_size = len(ids)
        for sid in ids[: self.max_candidates]:
            yield Candidate(sid)

    def _balanced_candidates(self) -> list[Candidate]:
        # Each methodology appears exactly once.
        return list(self.candidates())

    def _signal(self, df: pd.DataFrame, candidate: Candidate, cache=None) -> pd.Series:
        strategy = BY_ID.get(candidate.strategy_id)
        if strategy is None:
            raise ValueError(f"Unknown canonical strategy id: {candidate.strategy_id}")
        return CanonicalSignalEngine(df).signal(strategy)

    def current_direction(self, df: pd.DataFrame, candidate: Candidate) -> Direction:
        try:
            signal = self._signal(df, candidate)
        except ValueError:
            return Direction.HOLD
        if signal.empty:
            return Direction.HOLD
        last = int(signal.iloc[-1])
        if last > 0:
            return Direction.BUY
        if last < 0:
            return Direction.SELL
        return Direction.HOLD

    def _historical_regimes(self, df: pd.DataFrame, cache=None) -> pd.Series:
        close = df["close"]
        e20 = close.ewm(span=20, adjust=False).mean()
        e50 = close.ewm(span=50, adjust=False).mean()
        a = atr(df, 14)
        vol_ratio = a / close.replace(0, np.nan)
        vol_median = vol_ratio.rolling(120, min_periods=30).median()
        strength = (e20 - e50).abs() / a.replace(0, np.nan)
        labels = pd.Series("range", index=df.index, dtype="object")
        trending = strength > 1.0
        labels[trending & (e20 > e50)] = "trend_up"
        labels[trending & (e20 < e50)] = "trend_down"
        labels[vol_ratio > vol_median * 1.5] = "volatile"
        return labels

    def _walk_forward_slices(self, n: int) -> list[tuple[int, int, int]]:
        if n < 240:
            return []
        initial = max(180, int(n * 0.40))
        remaining = n - initial
        if remaining < self.walk_forward_folds * 20:
            return []
        step = max(20, remaining // self.walk_forward_folds)
        folds = []
        for i in range(self.walk_forward_folds):
            train_end = initial + i * step
            valid_start = train_end
            valid_end = n if i == self.walk_forward_folds - 1 else min(n, valid_start + step)
            if valid_end - valid_start >= 20:
                folds.append((train_end, valid_start, valid_end))
        return folds

    @staticmethod
    def _hit_rate(trades: list[TradeOutcome]) -> float:
        return float(np.mean([trade.won for trade in trades])) if trades else 0.0

    def _evaluate(self, df: pd.DataFrame, candidate: Candidate, cache, regimes: pd.Series) -> CandidateScore | None:
        try:
            signal = self._signal(df, candidate)
        except ValueError:
            return None
        atr_values = atr(df, 14)
        trades = self.backtester.simulate(
            df, signal, atr_values, max_holding=self.HORIZONS.get(candidate.strategy_id, 6)
        )
        if len(trades) < 20:
            return None
        folds = self._walk_forward_slices(len(df))
        if not folds:
            return None

        train_fold_hits: list[float] = []
        fold_hits: list[float] = []
        oos_by_signal: dict[int, TradeOutcome] = {}
        for train_end, valid_start, valid_end in folds:
            train_trades = [t for t in trades if t.exit_index < train_end]
            valid_trades = [
                t for t in trades
                if t.signal_index >= valid_start and t.entry_index >= valid_start and t.exit_index < valid_end
            ]
            if len(train_trades) < 10 or len(valid_trades) < 3:
                continue
            train_fold_hits.append(self._hit_rate(train_trades))
            fold_hits.append(self._hit_rate(valid_trades))
            for trade in valid_trades:
                oos_by_signal[trade.signal_index] = trade

        oos_trades = sorted(oos_by_signal.values(), key=lambda t: t.signal_index)
        if len(fold_hits) < self.min_walk_forward_folds or len(oos_trades) < 10 or not train_fold_hits:
            return None

        train_hit = float(np.mean(train_fold_hits))
        valid_hit = self._hit_rate(oos_trades)
        wf_std = float(np.std(fold_hits))
        net_returns = np.asarray([t.net_return for t in oos_trades], dtype=float)
        r_values = np.asarray([t.r_multiple for t in oos_trades], dtype=float)
        expectancy = float(np.mean(net_returns))
        avg_r = float(np.mean(r_values))
        pf = self.backtester.profit_factor(oos_trades)
        max_dd_r = self.backtester.max_drawdown_r(oos_trades)
        max_loss_streak = self.backtester.max_loss_streak(oos_trades)

        expectancy_score = float(np.clip(0.5 + 0.5 * np.tanh(avg_r / 0.50), 0.0, 1.0))
        pf_score = pf / (1.0 + pf)
        sample_bonus = float(np.clip(np.log1p(len(oos_trades)) / np.log(201), 0.0, 1.0))
        stability_score = float(np.clip(1.0 - abs(train_hit - valid_hit) - wf_std * 1.5, 0.0, 1.0))
        fold_coverage = len(fold_hits) / max(1, self.walk_forward_folds)
        drawdown_score = float(np.exp(-max_dd_r / 8.0))
        loss_streak_score = float(np.exp(-max_loss_streak / 8.0))

        regime_np = regimes.to_numpy(dtype=object)
        regime_scores: dict[str, float] = {}
        regime_trades: dict[str, int] = {}
        for regime in ("trend_up", "trend_down", "range", "volatile"):
            subset = [t for t in oos_trades if regime_np[t.signal_index] == regime]
            if not subset:
                continue
            wins = sum(t.won for t in subset)
            regime_scores[regime] = float((wins + 5) / (len(subset) + 10))
            regime_trades[regime] = len(subset)
        regime_diversity = sum(n >= 4 for n in regime_trades.values()) / 4.0

        score = (
            valid_hit * 0.25
            + train_hit * 0.04
            + sample_bonus * 0.10
            + pf_score * 0.11
            + expectancy_score * 0.14
            + drawdown_score * 0.10
            + fold_coverage * 0.08
            + stability_score * 0.10
            + regime_diversity * 0.04
            + loss_streak_score * 0.04
        )
        return CandidateScore(
            candidate=candidate,
            train_hit_rate=train_hit,
            valid_hit_rate=valid_hit,
            trades=len(trades),
            score=float(np.clip(score, 0.0, 1.0)),
            walk_forward_hit_rate=valid_hit,
            walk_forward_std=wf_std,
            expectancy=expectancy,
            profit_factor=pf,
            folds=len(fold_hits),
            regime_scores=regime_scores,
            regime_trades=regime_trades,
            avg_r_multiple=avg_r,
            max_drawdown_r=max_dd_r,
            max_loss_streak=max_loss_streak,
        )

    def _build_catalog(self, scores: list[CandidateScore]) -> list[CandidateScore]:
        return sorted(scores, key=lambda x: x.score, reverse=True)[: self.catalog_size]

    def run(self, df: pd.DataFrame) -> list[CandidateScore]:
        selected = self._balanced_candidates()
        self.last_evaluated = len(selected)
        regimes = self._historical_regimes(df)
        scores: list[CandidateScore] = []
        for candidate in selected:
            result = self._evaluate(df, candidate, None, regimes)
            if result is not None:
                scores.append(result)
        self.catalog = self._build_catalog(scores)
        self.top = self.catalog[:25]

        buckets: dict[str, list[float]] = {}
        category_buckets: dict[str, list[float]] = {}
        for result in self.catalog:
            buckets.setdefault(result.candidate.strategy_id, []).append(result.score)
            category = BY_ID[result.candidate.strategy_id].category
            category_buckets.setdefault(category, []).append(result.score)
        self.family_quality = {k: float(np.clip(np.mean(v), 0.35, 0.90)) for k, v in buckets.items()}
        self.category_quality = {k: float(np.clip(np.mean(v), 0.35, 0.90)) for k, v in category_buckets.items()}
        return self.top

    def quality(self, family: str) -> float:
        if family in self.family_quality:
            return self.family_quality[family]
        return self.category_quality.get(family, 0.50)
