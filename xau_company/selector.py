from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .canonical_strategies import BY_ID
from .models import AgentVote, Direction
from .research import CandidateScore, StrategyResearchAgent


@dataclass
class StrategyPick:
    score: CandidateScore
    direction: Direction
    probability_score: float
    analyst_agreement: int
    analyst_opposition: int
    regime_fit: float
    timeframe_alignment: float = 0.5
    macro_alignment: float = 0.5
    regime_history: float = 0.5
    regime_samples: int = 0
    walk_forward_stability: float = 0.5
    lifecycle_quality: float = 0.5

    @property
    def label(self) -> str:
        strategy = BY_ID.get(self.score.candidate.family)
        return strategy.name if strategy is not None else self.score.candidate.family


class StrategySelectorAgent:
    """Choose the surviving canonical strategy most compatible with the current market."""

    name = "Strategy Selection Desk"

    REGIME_FIT = {
        "trend_up": {
            "trend": 1.00,
            "momentum": 0.94,
            "breakout": 0.90,
            "smc": 0.82,
            "price_action": 0.80,
            "geometry": 0.72,
            "volume": 0.78,
            "quant": 0.78,
            "macro": 0.72,
            "mean_reversion": 0.42,
        },
        "trend_down": {
            "trend": 1.00,
            "momentum": 0.94,
            "breakout": 0.90,
            "smc": 0.82,
            "price_action": 0.80,
            "geometry": 0.72,
            "volume": 0.78,
            "quant": 0.78,
            "macro": 0.72,
            "mean_reversion": 0.42,
        },
        "range": {
            "mean_reversion": 1.00,
            "price_action": 0.88,
            "smc": 0.80,
            "geometry": 0.82,
            "volume": 0.83,
            "quant": 0.77,
            "trend": 0.48,
            "momentum": 0.50,
            "breakout": 0.44,
            "macro": 0.55,
        },
        "volatile": {
            "breakout": 1.00,
            "momentum": 0.88,
            "smc": 0.84,
            "price_action": 0.78,
            "trend": 0.76,
            "volume": 0.82,
            "quant": 0.80,
            "macro": 0.76,
            "geometry": 0.68,
            "mean_reversion": 0.38,
        },
    }

    TF_WEIGHTS = {"1min": 0.60, "5min": 0.80, "15min": 1.00, "1h": 1.30, "4h": 1.55}
    MACRO_AGENTS = {"USD Strength Desk", "Treasury Yield Desk"}

    def __init__(self, min_probability: float = 0.66, min_agreement: int = 2) -> None:
        self.min_probability = min_probability
        self.min_agreement = max(1, int(min_agreement))

    def _family_regime_fit(self, family: str, regime: str) -> float:
        strategy = BY_ID.get(family)
        category = strategy.category if strategy is not None else family
        return self.REGIME_FIT.get(regime, {}).get(category, 0.55)

    def _historical_regime_fit(self, result: CandidateScore, regime: str) -> tuple[float, int]:
        raw = result.regime_scores.get(regime, 0.50)
        samples = result.regime_trades.get(regime, 0)
        trust = min(1.0, samples / 40.0)
        return float(0.5 + (raw - 0.5) * trust), samples

    @staticmethod
    def _oos_sample_size(result: CandidateScore) -> int:
        regime_total = sum(max(0, int(n)) for n in result.regime_trades.values())
        return regime_total if regime_total > 0 else max(0, int(result.trades))

    def _directional_analyst_votes(self, votes: list[AgentVote]) -> list[AgentVote]:
        filtered: list[AgentVote] = []
        for vote in votes:
            if vote.metadata.get("timeframe") in self.TF_WEIGHTS:
                continue
            if vote.agent in self.MACRO_AGENTS:
                continue
            if vote.metadata.get("veto") is not None:
                continue
            if vote.agent == "Session Desk":
                continue
            filtered.append(vote)
        return filtered

    def _analyst_support(self, direction: Direction, votes: list[AgentVote]) -> tuple[float, float, int, int]:
        specialist_votes = self._directional_analyst_votes(votes)
        support = [v for v in specialist_votes if v.direction == direction]
        opposite_direction = Direction.SELL if direction == Direction.BUY else Direction.BUY
        opposition = [v for v in specialist_votes if v.direction == opposite_direction]
        support_strength = sum(v.confidence for v in support) / max(1, len(support))
        opposition_strength = sum(v.confidence for v in opposition) / max(1, len(opposition))
        return support_strength, opposition_strength, len(support), len(opposition)

    def _timeframe_alignment(self, direction: Direction, votes: list[AgentVote]) -> float:
        tf_votes = [v for v in votes if v.metadata.get("timeframe") in self.TF_WEIGHTS]
        if not tf_votes:
            return 0.50
        signed = 0.0
        total = 0.0
        for vote in tf_votes:
            weight = self.TF_WEIGHTS[vote.metadata["timeframe"]]
            total += weight
            if vote.direction == direction:
                signed += weight * vote.confidence
            elif vote.direction != Direction.HOLD:
                signed -= weight * vote.confidence
        return float(np.clip(0.5 + signed / max(total, 1e-9) * 0.5, 0.0, 1.0))

    def _macro_alignment(self, direction: Direction, votes: list[AgentVote]) -> float:
        macro = [v for v in votes if v.agent in self.MACRO_AGENTS]
        if not macro:
            return 0.50
        signed = 0.0
        active = 0
        for vote in macro:
            if vote.direction == Direction.HOLD:
                continue
            active += 1
            signed += vote.confidence if vote.direction == direction else -vote.confidence
        if active == 0:
            return 0.50
        return float(np.clip(0.5 + signed / active * 0.5, 0.0, 1.0))

    @staticmethod
    def _lifecycle_quality(result: CandidateScore) -> float:
        avg_r_score = float(np.clip(0.5 + 0.5 * np.tanh(result.avg_r_multiple / 0.50), 0.0, 1.0))
        drawdown_score = float(np.exp(-max(0.0, result.max_drawdown_r) / 8.0))
        streak_score = float(np.exp(-max(0, result.max_loss_streak) / 8.0))
        return avg_r_score * 0.50 + drawdown_score * 0.35 + streak_score * 0.15

    def select(self, df: pd.DataFrame, regime: str, votes: list[AgentVote], lab: StrategyResearchAgent) -> StrategyPick | None:
        catalog = lab.catalog if lab.catalog else lab.top
        if not catalog:
            return None

        best: StrategyPick | None = None
        for result in catalog:
            direction = lab.current_direction(df, result.candidate)
            if direction == Direction.HOLD:
                continue

            support, opposition, agreeing, opposing = self._analyst_support(direction, votes)
            if agreeing < self.min_agreement:
                continue

            family_fit = self._family_regime_fit(result.candidate.family, regime)
            regime_history, regime_samples = self._historical_regime_fit(result, regime)
            regime_fit = family_fit * 0.35 + regime_history * 0.65
            timeframe_alignment = self._timeframe_alignment(direction, votes)
            macro_alignment = self._macro_alignment(direction, votes)

            wf_hit = result.walk_forward_hit_rate if result.walk_forward_hit_rate > 0 else result.valid_hit_rate
            stability_gap = abs(result.train_hit_rate - result.valid_hit_rate)
            stability = float(np.clip(1.0 - stability_gap - result.walk_forward_std * 1.5, 0.0, 1.0))
            pf_score = result.profit_factor / (1.0 + max(0.0, result.profit_factor))
            lifecycle_quality = self._lifecycle_quality(result)

            probability = (
                wf_hit * 0.20
                + result.score * 0.13
                + regime_fit * 0.18
                + support * 0.09
                + timeframe_alignment * 0.13
                + macro_alignment * 0.04
                + pf_score * 0.05
                + stability * 0.07
                + lifecycle_quality * 0.11
                - opposition * 0.08
            )
            sample_n = self._oos_sample_size(result)
            sample_trust = float(np.clip(np.log1p(sample_n) / np.log(401), 0.0, 1.0))
            probability -= (1.0 - sample_trust) * 0.08
            probability = float(np.clip(probability, 0.0, 0.97))

            pick = StrategyPick(
                score=result,
                direction=direction,
                probability_score=probability,
                analyst_agreement=agreeing,
                analyst_opposition=opposing,
                regime_fit=regime_fit,
                timeframe_alignment=timeframe_alignment,
                macro_alignment=macro_alignment,
                regime_history=regime_history,
                regime_samples=regime_samples,
                walk_forward_stability=stability,
                lifecycle_quality=lifecycle_quality,
            )
            if best is None or pick.probability_score > best.probability_score:
                best = pick

        if best is None or best.probability_score < self.min_probability:
            return None
        return best
