from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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

    @property
    def label(self) -> str:
        params = ", ".join(str(x) for x in self.score.candidate.params)
        return f"{self.score.candidate.family}({params})"


class StrategySelectorAgent:
    """Chooses the researched strategy most compatible with the market right now."""

    name = "Strategy Selection Desk"

    REGIME_FIT = {
        "trend_up": {"trend": 1.00, "momentum": 0.95, "breakout": 0.88, "mean_reversion": 0.45},
        "trend_down": {"trend": 1.00, "momentum": 0.95, "breakout": 0.88, "mean_reversion": 0.45},
        "range": {"mean_reversion": 1.00, "trend": 0.52, "momentum": 0.55, "breakout": 0.48},
        "volatile": {"breakout": 0.92, "momentum": 0.86, "trend": 0.70, "mean_reversion": 0.38},
    }

    TF_WEIGHTS = {"1min": 0.60, "5min": 0.80, "15min": 1.00, "1h": 1.30, "4h": 1.55}

    def __init__(self, min_probability: float = 0.66, min_agreement: int = 2) -> None:
        self.min_probability = min_probability
        self.min_agreement = min_agreement

    def _regime_fit(self, family: str, regime: str) -> float:
        return self.REGIME_FIT.get(regime, {}).get(family, 0.60)

    def _analyst_support(self, direction: Direction, votes: list[AgentVote]) -> tuple[float, float, int, int]:
        support = [v for v in votes if v.direction == direction]
        opposite_direction = Direction.SELL if direction == Direction.BUY else Direction.BUY
        opposition = [v for v in votes if v.direction == opposite_direction]
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
        macro = [v for v in votes if v.agent in {"USD Strength Desk", "Treasury Yield Desk"}]
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

            regime_fit = self._regime_fit(result.candidate.family, regime)
            timeframe_alignment = self._timeframe_alignment(direction, votes)
            macro_alignment = self._macro_alignment(direction, votes)

            # Historical robustness remains dominant, but activation is conditional
            # on current regime, independent analysts, higher timeframes and macro context.
            probability = (
                result.valid_hit_rate * 0.31
                + result.score * 0.20
                + regime_fit * 0.14
                + support * 0.13
                + timeframe_alignment * 0.15
                + macro_alignment * 0.07
                - opposition * 0.12
            )
            sample_factor = float(np.clip(np.log1p(result.trades) / np.log(301), 0.55, 1.0))
            stability_gap = abs(result.train_hit_rate - result.valid_hit_rate)
            probability = probability * sample_factor - stability_gap * 0.18
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
            )
            if best is None or pick.probability_score > best.probability_score:
                best = pick

        if best is None or best.probability_score < self.min_probability:
            return None
        return best
