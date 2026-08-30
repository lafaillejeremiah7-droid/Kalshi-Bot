from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .indicators import atr, donchian, ema, roc, rsi, rolling_zscore
from .models import Direction


@dataclass(frozen=True)
class Candidate:
    family: str
    params: tuple


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


class StrategyResearchAgent:
    """Research tens of thousands of strategy variants with walk-forward validation.

    The lab builds a broad strategy universe, spreads the configured research budget
    across families/parameter ranges, evaluates each candidate on expanding
    walk-forward folds, and records out-of-sample performance by market regime.
    """

    HORIZONS = {
        "trend": 6,
        "triple_trend": 8,
        "mean_reversion": 3,
        "bollinger_reversion": 3,
        "breakout": 5,
        "momentum": 4,
        "pullback": 4,
        "volatility_breakout": 5,
        "rsi_trend": 6,
        "bollinger_breakout": 4,
        "range_fade": 3,
    }

    def __init__(
        self,
        max_candidates: int = 20_000,
        spread_bps: float = 1.5,
        walk_forward_folds: int = 4,
        catalog_size: int = 600,
        min_walk_forward_folds: int = 2,
    ) -> None:
        self.max_candidates = max_candidates
        self.spread_bps = spread_bps
        self.walk_forward_folds = max(2, int(walk_forward_folds))
        self.catalog_size = max(50, int(catalog_size))
        self.min_walk_forward_folds = max(2, int(min_walk_forward_folds))
        self.family_quality: dict[str, float] = {}
        self.top: list[CandidateScore] = []
        self.catalog: list[CandidateScore] = []
        self.last_evaluated = 0
        self.last_universe_size = 0

    def candidates(self) -> Iterable[Candidate]:
        for fast, slow, threshold in product(
            [5, 8, 10, 12, 15, 20, 25, 30],
            [30, 40, 50, 60, 75, 100, 150, 200],
            [0.0, 0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012],
        ):
            if fast < slow:
                yield Candidate("trend", (fast, slow, threshold))

        for fast, mid, slow, mom_period, min_mom in product(
            [5, 8, 10, 12, 15],
            [20, 25, 30, 40, 50],
            [60, 75, 100, 150, 200],
            [3, 5, 8, 14],
            [0.0, 0.0003, 0.0006, 0.0010],
        ):
            if fast < mid < slow:
                yield Candidate("triple_trend", (fast, mid, slow, mom_period, min_mom))

        for period, low, high, zlim, zperiod in product(
            [5, 7, 9, 11, 14, 18, 21, 28],
            [20, 25, 30, 35, 40],
            [60, 65, 70, 75, 80],
            [0.5, 0.8, 1.0, 1.2, 1.5, 1.8],
            [15, 20, 30, 40, 60],
        ):
            if low < 50 < high:
                yield Candidate("mean_reversion", (period, low, high, zlim, zperiod))

        for period, band, rsi_period, low, high in product(
            [10, 14, 20, 25, 30, 40, 50, 60],
            [1.2, 1.5, 1.8, 2.0, 2.2, 2.5],
            [5, 7, 9, 14, 18, 21],
            [20, 25, 30, 35],
            [65, 70, 75, 80],
        ):
            yield Candidate("bollinger_reversion", (period, band, rsi_period, low, high))

        for lookback, buffer, mom_period in product(
            [10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150],
            [0.0, 0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012],
            [2, 3, 5, 8, 10, 14, 20, 30],
        ):
            yield Candidate("breakout", (lookback, buffer, mom_period))

        for period, threshold, trend_span in product(
            [2, 3, 5, 8, 10, 14, 20, 30, 40, 60],
            [0.0001, 0.0002, 0.0004, 0.0006, 0.001, 0.0015, 0.002, 0.003],
            [20, 30, 40, 50, 75, 100, 150, 200],
        ):
            yield Candidate("momentum", (period, threshold, trend_span))

        for fast, slow, rsi_period, buy_level, sell_level, recovery in product(
            [10, 15, 20, 25],
            [50, 75, 100, 150],
            [7, 9, 14, 21],
            [30, 35, 40, 45],
            [55, 60, 65, 70],
            [2, 5, 8],
        ):
            if fast < slow:
                yield Candidate("pullback", (fast, slow, rsi_period, buy_level, sell_level, recovery))

        for lookback, atr_period, atr_mult, trend_span in product(
            [10, 15, 20, 30, 40, 60, 80, 100],
            [7, 10, 14, 18, 21, 28],
            [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
            [20, 30, 50, 75, 100, 150],
        ):
            yield Candidate("volatility_breakout", (lookback, atr_period, atr_mult, trend_span))

        for fast, slow, rsi_period, bull_floor, bear_ceiling in product(
            [5, 8, 10, 12, 15],
            [30, 40, 50, 75, 100, 150, 200],
            [5, 7, 9, 14, 18, 21],
            [50, 55, 60, 65],
            [50, 45, 40, 35],
        ):
            if fast < slow:
                yield Candidate("rsi_trend", (fast, slow, rsi_period, bull_floor, bear_ceiling))

        for period, band, mom_period, trend_span in product(
            [10, 14, 20, 25, 30, 40, 50, 60],
            [1.0, 1.3, 1.5, 1.8, 2.0],
            [2, 3, 5, 8, 14, 20],
            [20, 30, 50, 75, 100, 150],
        ):
            yield Candidate("bollinger_breakout", (period, band, mom_period, trend_span))

        for lookback, edge, rsi_period, low, high in product(
            [15, 20, 30, 40, 50, 75, 100, 150],
            [0.05, 0.10, 0.15, 0.20, 0.25],
            [5, 7, 9, 14, 18, 21],
            [20, 25, 30, 35],
            [65, 70, 75, 80],
        ):
            yield Candidate("range_fade", (lookback, edge, rsi_period, low, high))

    def _balanced_candidates(self) -> list[Candidate]:
        groups: dict[str, list[Candidate]] = {}
        for candidate in self.candidates():
            groups.setdefault(candidate.family, []).append(candidate)

        self.last_universe_size = sum(len(items) for items in groups.values())
        if not groups or self.max_candidates <= 0:
            return []

        target = min(self.max_candidates, self.last_universe_size)
        quota = max(1, target // len(groups))
        selected: list[Candidate] = []
        leftovers: dict[str, list[Candidate]] = {}

        for family, items in groups.items():
            take = min(quota, len(items))
            if take == len(items):
                chosen_indices = set(range(len(items)))
            else:
                chosen_indices = set(np.linspace(0, len(items) - 1, num=take, dtype=int).tolist())
            selected.extend(items[i] for i in sorted(chosen_indices))
            leftovers[family] = [item for i, item in enumerate(items) if i not in chosen_indices]

        offsets = {family: 0 for family in leftovers}
        families = list(leftovers)
        while len(selected) < target:
            added = False
            for family in families:
                idx = offsets[family]
                items = leftovers[family]
                if idx < len(items):
                    selected.append(items[idx])
                    offsets[family] = idx + 1
                    added = True
                    if len(selected) >= target:
                        break
            if not added:
                break

        return selected

    @staticmethod
    def _cached(
        cache: dict[tuple, pd.Series | tuple[pd.Series, pd.Series]],
        key: tuple,
        builder: Callable[[], pd.Series | tuple[pd.Series, pd.Series]],
    ):
        if key not in cache:
            cache[key] = builder()
        return cache[key]

    def _signal(
        self,
        df: pd.DataFrame,
        candidate: Candidate,
        cache: dict[tuple, pd.Series | tuple[pd.Series, pd.Series]] | None = None,
    ) -> pd.Series:
        cache = {} if cache is None else cache
        close = df["close"]
        out = pd.Series(0, index=df.index, dtype="int8")
        p = candidate.params

        def cema(span: int) -> pd.Series:
            return self._cached(cache, ("ema", span), lambda: ema(close, span))

        def croc(period: int) -> pd.Series:
            return self._cached(cache, ("roc", period), lambda: roc(close, period))

        def crsi(period: int) -> pd.Series:
            return self._cached(cache, ("rsi", period), lambda: rsi(close, period))

        def cz(period: int) -> pd.Series:
            return self._cached(cache, ("z", period), lambda: rolling_zscore(close, period))

        def catr(period: int) -> pd.Series:
            return self._cached(cache, ("atr", period), lambda: atr(df, period))

        def cdonchian(period: int) -> tuple[pd.Series, pd.Series]:
            return self._cached(cache, ("donchian", period), lambda: donchian(df, period))

        def mean_std(period: int) -> tuple[pd.Series, pd.Series]:
            return self._cached(
                cache,
                ("mean_std", period),
                lambda: (close.rolling(period).mean(), close.rolling(period).std().replace(0, np.nan)),
            )

        if candidate.family == "trend":
            fast, slow, threshold = p
            gap = (cema(fast) - cema(slow)) / close
            out[gap > threshold] = 1
            out[gap < -threshold] = -1

        elif candidate.family == "triple_trend":
            fast, mid, slow, mom_period, min_mom = p
            ef, em, es = cema(fast), cema(mid), cema(slow)
            mom = croc(mom_period)
            out[(ef > em) & (em > es) & (mom > min_mom)] = 1
            out[(ef < em) & (em < es) & (mom < -min_mom)] = -1

        elif candidate.family == "mean_reversion":
            period, low, high, zlim, zperiod = p
            rs, z = crsi(period), cz(zperiod)
            out[(rs < low) & (z < -zlim)] = 1
            out[(rs > high) & (z > zlim)] = -1

        elif candidate.family == "bollinger_reversion":
            period, band, rsi_period, low, high = p
            mean, std = mean_std(period)
            rs = crsi(rsi_period)
            out[(close < mean - std * band) & (rs < low)] = 1
            out[(close > mean + std * band) & (rs > high)] = -1

        elif candidate.family == "breakout":
            lookback, buffer, mom_period = p
            hi, lo = cdonchian(lookback)
            mom = croc(mom_period)
            out[(close > hi * (1 + buffer)) & (mom > 0)] = 1
            out[(close < lo * (1 - buffer)) & (mom < 0)] = -1

        elif candidate.family == "momentum":
            period, threshold, trend_span = p
            mom = croc(period)
            trend = cema(trend_span)
            out[(mom > threshold) & (close > trend)] = 1
            out[(mom < -threshold) & (close < trend)] = -1

        elif candidate.family == "pullback":
            fast, slow, rsi_period, buy_level, sell_level, recovery = p
            ef, es, rs = cema(fast), cema(slow), crsi(rsi_period)
            out[(ef > es) & (rs.shift(1) < buy_level) & (rs >= buy_level + recovery)] = 1
            out[(ef < es) & (rs.shift(1) > sell_level) & (rs <= sell_level - recovery)] = -1

        elif candidate.family == "volatility_breakout":
            lookback, atr_period, atr_mult, trend_span = p
            center = self._cached(
                cache,
                ("center", lookback),
                lambda: close.rolling(lookback).mean().shift(1),
            )
            a = catr(atr_period)
            trend = cema(trend_span)
            out[(close > center + a * atr_mult) & (close > trend)] = 1
            out[(close < center - a * atr_mult) & (close < trend)] = -1

        elif candidate.family == "rsi_trend":
            fast, slow, rsi_period, bull_floor, bear_ceiling = p
            ef, es, rs = cema(fast), cema(slow), crsi(rsi_period)
            out[(ef > es) & (rs > bull_floor)] = 1
            out[(ef < es) & (rs < bear_ceiling)] = -1

        elif candidate.family == "bollinger_breakout":
            period, band, mom_period, trend_span = p
            mean, std = mean_std(period)
            mom, trend = croc(mom_period), cema(trend_span)
            out[(close > mean + std * band) & (mom > 0) & (close > trend)] = 1
            out[(close < mean - std * band) & (mom < 0) & (close < trend)] = -1

        elif candidate.family == "range_fade":
            lookback, edge, rsi_period, low, high = p
            hi, lo = cdonchian(lookback)
            width = (hi - lo).replace(0, np.nan)
            location = (close - lo) / width
            rs = crsi(rsi_period)
            out[(location < edge) & (rs < low)] = 1
            out[(location > 1 - edge) & (rs > high)] = -1

        return out

    def current_direction(self, df: pd.DataFrame, candidate: Candidate) -> Direction:
        """Return a strategy's direction on the newest completed candle."""
        signal = self._signal(df, candidate, cache={})
        if signal.empty:
            return Direction.HOLD
        last = int(signal.iloc[-1])
        if last > 0:
            return Direction.BUY
        if last < 0:
            return Direction.SELL
        return Direction.HOLD

    def _historical_regimes(
        self,
        df: pd.DataFrame,
        cache: dict[tuple, pd.Series | tuple[pd.Series, pd.Series]],
    ) -> pd.Series:
        close = df["close"]
        e20 = self._cached(cache, ("ema", 20), lambda: ema(close, 20))
        e50 = self._cached(cache, ("ema", 50), lambda: ema(close, 50))
        a = self._cached(cache, ("atr", 14), lambda: atr(df, 14))
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

        folds: list[tuple[int, int, int]] = []
        for i in range(self.walk_forward_folds):
            train_end = initial + i * step
            valid_start = train_end
            valid_end = n if i == self.walk_forward_folds - 1 else min(n, valid_start + step)
            if valid_end - valid_start >= 20:
                folds.append((train_end, valid_start, valid_end))
        return folds

    @staticmethod
    def _profit_factor(returns: np.ndarray) -> float:
        positive = float(returns[returns > 0].sum())
        negative = float(-returns[returns < 0].sum())
        if negative <= 1e-12:
            return 4.0 if positive > 0 else 1.0
        return float(np.clip(positive / negative, 0.0, 4.0))

    def _evaluate(
        self,
        df: pd.DataFrame,
        candidate: Candidate,
        cache: dict[tuple, pd.Series | tuple[pd.Series, pd.Series]],
        regimes: pd.Series,
    ) -> CandidateScore | None:
        signal = self._signal(df, candidate, cache)
        horizon = self.HORIZONS.get(candidate.family, 4)
        future = df["close"].shift(-horizon) / df["close"] - 1
        cost = self.spread_bps / 10_000
        signed = signal * future - (signal != 0).astype(float) * cost

        signal_np = signal.to_numpy()
        future_np = future.to_numpy(dtype=float)
        signed_np = signed.to_numpy(dtype=float)
        active = (signal_np != 0) & np.isfinite(future_np) & np.isfinite(signed_np)
        if int(active.sum()) < 30:
            return None

        folds = self._walk_forward_slices(len(df))
        if not folds:
            return None

        train_wins = 0
        train_total = 0
        valid_wins = 0
        valid_total = 0
        fold_hits: list[float] = []
        oos_mask = np.zeros(len(df), dtype=bool)
        positions = np.arange(len(df))

        for train_end, valid_start, valid_end in folds:
            train_mask = active & (positions < train_end)
            valid_mask = active & (positions >= valid_start) & (positions < valid_end)
            train_n = int(train_mask.sum())
            valid_n = int(valid_mask.sum())
            if train_n < 12 or valid_n < 4:
                continue

            train_returns = signed_np[train_mask]
            valid_returns = signed_np[valid_mask]
            train_wins += int((train_returns > 0).sum())
            train_total += train_n
            valid_wins += int((valid_returns > 0).sum())
            valid_total += valid_n
            fold_hits.append(float((valid_returns > 0).mean()))
            oos_mask |= valid_mask

        if len(fold_hits) < self.min_walk_forward_folds or valid_total < 12 or train_total == 0:
            return None

        train_hit = train_wins / train_total
        valid_hit = valid_wins / valid_total
        wf_std = float(np.std(fold_hits))
        oos_returns = signed_np[oos_mask]
        expectancy = float(np.mean(oos_returns)) if len(oos_returns) else 0.0
        pf = self._profit_factor(oos_returns)

        baseline_move = float(np.nanmedian(np.abs(future_np)))
        exp_scale = max(baseline_move, cost, 1e-6)
        expectancy_score = float(np.clip(0.5 + 0.5 * np.tanh(expectancy / exp_scale), 0.0, 1.0))
        pf_score = pf / (1.0 + pf)
        sample_bonus = float(np.clip(np.log1p(valid_total) / np.log(301), 0.0, 1.0))
        stability_gap = abs(train_hit - valid_hit)
        stability_score = float(np.clip(1.0 - stability_gap - wf_std * 1.5, 0.0, 1.0))
        fold_coverage = len(fold_hits) / max(1, self.walk_forward_folds)

        regime_np = regimes.to_numpy(dtype=object)
        regime_scores: dict[str, float] = {}
        regime_trades: dict[str, int] = {}
        for regime in ("trend_up", "trend_down", "range", "volatile"):
            mask = oos_mask & (regime_np == regime)
            n = int(mask.sum())
            if n == 0:
                continue
            wins = int((signed_np[mask] > 0).sum())
            regime_scores[regime] = float((wins + 5) / (n + 10))
            regime_trades[regime] = n

        regime_diversity = sum(n >= 5 for n in regime_trades.values()) / 4.0
        score = (
            valid_hit * 0.34
            + train_hit * 0.06
            + sample_bonus * 0.13
            + pf_score * 0.12
            + expectancy_score * 0.12
            + fold_coverage * 0.09
            + stability_score * 0.10
            + regime_diversity * 0.04
        )

        return CandidateScore(
            candidate=candidate,
            train_hit_rate=float(train_hit),
            valid_hit_rate=float(valid_hit),
            trades=int(active.sum()),
            score=float(np.clip(score, 0.0, 1.0)),
            walk_forward_hit_rate=float(valid_hit),
            walk_forward_std=wf_std,
            expectancy=expectancy,
            profit_factor=pf,
            folds=len(fold_hits),
            regime_scores=regime_scores,
            regime_trades=regime_trades,
        )

    def _build_catalog(self, scores: list[CandidateScore]) -> list[CandidateScore]:
        if not scores:
            return []

        by_family: dict[str, list[CandidateScore]] = {}
        for result in scores:
            by_family.setdefault(result.candidate.family, []).append(result)

        reserve_per_family = max(5, self.catalog_size // max(1, len(by_family) * 2))
        catalog: list[CandidateScore] = []
        seen: set[Candidate] = set()
        for family in sorted(by_family):
            for result in by_family[family][:reserve_per_family]:
                catalog.append(result)
                seen.add(result.candidate)

        for result in scores:
            if len(catalog) >= self.catalog_size:
                break
            if result.candidate not in seen:
                catalog.append(result)
                seen.add(result.candidate)

        catalog.sort(key=lambda x: x.score, reverse=True)
        return catalog[: self.catalog_size]

    def run(self, df: pd.DataFrame) -> list[CandidateScore]:
        selected = self._balanced_candidates()
        self.last_evaluated = len(selected)
        cache: dict[tuple, pd.Series | tuple[pd.Series, pd.Series]] = {}
        regimes = self._historical_regimes(df, cache)

        scores: list[CandidateScore] = []
        for candidate in selected:
            result = self._evaluate(df, candidate, cache, regimes)
            if result is not None:
                scores.append(result)

        scores.sort(key=lambda x: x.score, reverse=True)
        self.catalog = self._build_catalog(scores)
        self.top = self.catalog[:25]

        buckets: dict[str, list[float]] = {}
        for result in self.catalog:
            buckets.setdefault(result.candidate.family, []).append(result.score)
        self.family_quality = {
            family: float(np.clip(np.mean(vals[:20]), 0.35, 0.90))
            for family, vals in buckets.items()
        }
        return self.top

    def quality(self, family: str) -> float:
        return self.family_quality.get(family, 0.50)
