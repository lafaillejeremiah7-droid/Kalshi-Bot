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
    regime_history: float = 0.5
    regime_samples: int = 0
    walk_forward_stability: float = 0.5
    lifecycle_quality: float = 0.5

    @property
    def label(self) -> str:
        candidate = self.score.candidate
        if candidate.family == "invented":
            try:
                family_id, variant_id, logic, feature_specs, gate_spec = candidate.params
                features = "+".join(str(spec[0]).replace("_", " ") for spec in feature_specs)
                gate = str(gate_spec[0]).replace("_", " ")
                return f"Invention {family_id} v{variant_id}: {features} | {gate} | {logic}"
            except (TypeError, ValueError, IndexError):
                return "Invented strategy"
        if candidate.family == "ensemble":
            try:
                family_a, params_a, family_b, params_b, mode = candidate.params
                def parent_label(family, params):
                    if family == "invented" and params:
                        return f"{params[0]}v{params[1]}"
                    return str(family)
                return f"Ensemble {parent_label(family_a, params_a)} + {parent_label(family_b, params_b)} | {mode}"
            except (TypeError, ValueError, IndexError):
                return "Ensemble strategy"
        params = ", ".join(str(x) for x in candidate.params)
        return f"{candidate.family}({params})"


class StrategySelectorAgent:
    """Choose the researched strategy most compatible with the market right now."""

    name = "Strategy Selection Desk"

    REGIME_FIT = {
        "trend_up": {
            "trend": 1.00, "triple_trend": 1.00, "rsi_trend": 0.96, "pullback": 0.94,
            "momentum": 0.93, "breakout": 0.86, "bollinger_breakout": 0.83,
            "volatility_breakout": 0.78, "mean_reversion": 0.43, "bollinger_reversion": 0.40,
            "range_fade": 0.35,
        },
        "trend_down": {
            "trend": 1.00, "triple_trend": 1.00, "rsi_trend": 0.96, "pullback": 0.94,
            "momentum": 0.93, "breakout": 0.86, "bollinger_breakout": 0.83,
            "volatility_breakout": 0.78, "mean_reversion": 0.43, "bollinger_reversion": 0.40,
            "range_fade": 0.35,
        },
        "range": {
            "mean_reversion": 1.00, "bollinger_reversion": 0.98, "range_fade": 0.96,
            "pullback": 0.58, "trend": 0.50, "triple_trend": 0.48, "rsi_trend": 0.50,
            "momentum": 0.50, "breakout": 0.45, "bollinger_breakout": 0.43,
            "volatility_breakout": 0.42,
        },
        "volatile": {
            "volatility_breakout": 1.00, "breakout": 0.94, "bollinger_breakout": 0.92,
            "momentum": 0.84, "trend": 0.70, "triple_trend": 0.72, "rsi_trend": 0.66,
            "pullback": 0.60, "mean_reversion": 0.35, "bollinger_reversion": 0.34,
            "range_fade": 0.30,
        },
    }

    TF_WEIGHTS = {"1min": 0.60, "5min": 0.80, "15min": 1.00, "1h": 1.30, "4h": 1.55}
    MACRO_AGENTS = {"USD Strength Desk", "Treasury Yield Desk"}

    def __init__(self, min_probability: float = 0.66, min_agreement: int = 2) -> None:
        self.min_probability = min_probability
        self.min_agreement = max(1, int(min_agreement))

    def _family_regime_fit(self, family: str, regime: str) -> float:
        return self.REGIME_FIT.get(regime, {}).get(family, 0.55)

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
        """Return specialist directional votes without double-counting context."""
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
