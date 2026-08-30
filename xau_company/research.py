from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from .indicators import donchian, ema, roc, rsi, rolling_zscore


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


class StrategyResearchAgent:
    """Search large strategy parameter spaces and rank out-of-sample stability."""

    def __init__(self, max_candidates: int = 3000, spread_bps: float = 1.5) -> None:
        self.max_candidates = max_candidates
        self.spread_bps = spread_bps
        self.family_quality: dict[str, float] = {}
        self.top: list[CandidateScore] = []
        self.last_evaluated = 0

    def candidates(self) -> Iterable[Candidate]:
        for fast, slow, threshold in product(
            [5, 8, 10, 12, 15, 20, 25, 30],
            [30, 40, 50, 60, 75, 100, 150, 200],
            [0.0, 0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012],
        ):
            if fast < slow:
                yield Candidate("trend", (fast, slow, threshold))

        for period, low, high, zlim, zperiod in product(
            [5, 7, 9, 11, 14, 18, 21, 28],
            [20, 25, 30, 35, 40],
            [60, 65, 70, 75, 80],
            [0.5, 0.8, 1.0, 1.2, 1.5, 1.8],
            [15, 20, 30, 40, 60],
        ):
            if low < 50 < high:
                yield Candidate("mean_reversion", (period, low, high, zlim, zperiod))

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

    def _balanced_candidates(self) -> list[Candidate]:
        groups: dict[str, list[Candidate]] = {}
        for c in self.candidates():
            groups.setdefault(c.family, []).append(c)
        if not groups:
            return []
        quota = max(1, self.max_candidates // len(groups))
        selected: list[Candidate] = []
        offsets: dict[str, int] = {}
        for family, items in groups.items():
            take = min(quota, len(items))
            selected.extend(items[:take])
            offsets[family] = take
        families = list(groups)
        while len(selected) < self.max_candidates:
            added = False
            for family in families:
                idx = offsets[family]
                items = groups[family]
                if idx < len(items):
                    selected.append(items[idx])
                    offsets[family] = idx + 1
                    added = True
                    if len(selected) >= self.max_candidates:
                        break
            if not added:
                break
        return selected

    def _signal(self, df: pd.DataFrame, c: Candidate) -> pd.Series:
        close = df["close"]
        out = pd.Series(0, index=df.index, dtype="int8")
        if c.family == "trend":
            fast, slow, threshold = c.params
            ef, es = ema(close, fast), ema(close, slow)
            gap = (ef - es) / close
            out[gap > threshold] = 1
            out[gap < -threshold] = -1
        elif c.family == "mean_reversion":
            period, low, high, zlim, zperiod = c.params
            rs = rsi(close, period)
            z = rolling_zscore(close, zperiod)
            out[(rs < low) & (z < -zlim)] = 1
            out[(rs > high) & (z > zlim)] = -1
        elif c.family == "breakout":
            lookback, buffer, mom_period = c.params
            hi, lo = donchian(df, lookback)
            mom = roc(close, mom_period)
            out[(close > hi * (1 + buffer)) & (mom > 0)] = 1
            out[(close < lo * (1 - buffer)) & (mom < 0)] = -1
        elif c.family == "momentum":
            period, threshold, trend_span = c.params
            m = roc(close, period)
            trend = ema(close, trend_span)
            out[(m > threshold) & (close > trend)] = 1
            out[(m < -threshold) & (close < trend)] = -1
        return out

    def _evaluate(self, df: pd.DataFrame, c: Candidate) -> CandidateScore | None:
        sig = self._signal(df, c)
        horizon = 3
        future = df["close"].shift(-horizon) / df["close"] - 1
        cost = self.spread_bps / 10_000
        signed = sig * future - (sig != 0).astype(float) * cost
        active = sig != 0
        if int(active.sum()) < 24:
            return None
        cut = int(len(df) * 0.70)
        train_mask = active & (df.index < cut)
        valid_mask = active & (df.index >= cut) & future.notna()
        train_n = int(train_mask.sum())
        valid_n = int(valid_mask.sum())
        if train_n < 16 or valid_n < 8:
            return None
        train_hit = float((signed[train_mask] > 0).mean())
        valid_hit = float((signed[valid_mask] > 0).mean())
        gap = abs(train_hit - valid_hit)
        sample_bonus = min(1.0, np.log1p(valid_n) / np.log(101))
        score = valid_hit * 0.72 + train_hit * 0.12 + sample_bonus * 0.16 - gap * 0.45
        return CandidateScore(c, train_hit, valid_hit, train_n + valid_n, float(score))

    def run(self, df: pd.DataFrame) -> list[CandidateScore]:
        scores: list[CandidateScore] = []
        selected = self._balanced_candidates()
        self.last_evaluated = len(selected)
        for candidate in selected:
            score = self._evaluate(df, candidate)
            if score is not None:
                scores.append(score)
        scores.sort(key=lambda x: x.score, reverse=True)
        self.top = scores[:25]
        buckets: dict[str, list[float]] = {}
        for s in scores[:300]:
            buckets.setdefault(s.candidate.family, []).append(s.score)
        self.family_quality = {
            family: float(np.clip(np.mean(vals[:20]), 0.35, 0.85))
            for family, vals in buckets.items()
        }
        return self.top

    def quality(self, family: str) -> float:
        return self.family_quality.get(family, 0.50)
