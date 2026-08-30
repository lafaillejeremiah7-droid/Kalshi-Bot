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

    @property
    def label(self) -> str:
        params = ", ".join(str(x) for x in self.score.candidate.params)
        return f"{self.score.candidate.family}({params})"


class StrategySelectorAgent:
    """Selects the researched strategy with the strongest fit to the current market.

    The research lab answers: "Which strategies have been robust out of sample?"
    The analyst desks answer: "What does the market look like now?"
    This agent combines the two and chooses one strategy.  It does not assume the
    historically highest win-rate strategy is always the best strategy right now.
    """

    name = "Strategy Selection Desk"

    REGIME_FIT = {
        "trend_up": {"trend": 1.00, "momentum": 0.95, "breakout": 0.88, "mean_reversion": 0.45},
        "trend_down": {"trend": 1.00, "momentum": 0.95, "breakout": 0.88, "mean_reversion": 0.45},
        "range": {"mean_reversion": 1.00, "trend": 0.52, "momentum": 0.55, "breakout": 0.48},
        "volatile": {"breakout": 0.92, "momentum": 0.86, "trend": 0.70, "mean_reversion": 0.38},
    }

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

    def select(
        self,
        df: pd.DataFrame,
        regime: str,
        votes: list[AgentVote],
        lab: StrategyResearchAgent,
    ) -> StrategyPick | None:
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
            # OOS robustness is dominant; current-market fit and independent analyst
            # confirmation decide which robust strategy should be active right now.
            probability = (
                result.valid_hit_rate * 0.38
                + result.score * 0.26
                + regime_fit * 0.18
                + support * 0.18
                - opposition * 0.14
            )
            # Penalize small samples and train/validation instability again at the
            # activation stage so a lucky backtest is less likely to be selected.
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
            )
            if best is None or pick.probability_score > best.probability_score:
                best = pick

        if best is None or best.probability_score < self.min_probability:
            return None
        return best
