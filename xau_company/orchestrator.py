from __future__ import annotations

import numpy as np
import pandas as pd

from .agents import (
    BreakoutAgent,
    MeanReversionAgent,
    MomentumAgent,
    PriceActionAgent,
    RegimeAgent,
    SessionAgent,
    StructureAgent,
    TrendAgent,
    VolatilityGuardAgent,
)
from .indicators import atr
from .models import AgentVote, TradeSignal
from .research import StrategyResearchAgent
from .selector import StrategySelectorAgent


class BossAgent:
    """CEO/orchestrator.

    Analysts diagnose the current market. The Strategy Selection Desk then chooses
    one robust strategy from the research library whose live trigger and historical
    validation best fit that diagnosis. The boss applies hard risk vetoes and emits
    only the final strategy-backed signal.
    """

    def __init__(self, lab: StrategyResearchAgent, min_confidence: float = 0.72, min_consensus: int = 3) -> None:
        self.lab = lab
        self.min_confidence = min_confidence
        self.regime_agent = RegimeAgent()
        self.selector = StrategySelectorAgent(min_probability=min_confidence, min_agreement=max(2, min_consensus - 1))
        self.desks = [
            TrendAgent(),
            BreakoutAgent(),
            MeanReversionAgent(),
            MomentumAgent(),
            PriceActionAgent(),
            StructureAgent(),
            VolatilityGuardAgent(),
            SessionAgent(),
        ]

    def decide(self, symbol: str, df: pd.DataFrame) -> TradeSignal | None:
        if len(df) < 220 or not self.lab.catalog:
            return None

        regime = self.regime_agent.classify(df)
        votes: list[AgentVote] = [desk.vote(df, regime, self.lab) for desk in self.desks]

        # Risk employees have authority to stop the company from entering a trade.
        if any(v.metadata.get("veto") for v in votes):
            return None

        pick = self.selector.select(df, regime, votes, self.lab)
        if pick is None:
            return None

        entry = float(df.close.iloc[-1])
        a = float(atr(df, 14).iloc[-1])
        if not np.isfinite(a) or a <= 0:
            return None

        # Risk geometry adapts to the current regime rather than using a fixed stop.
        stop_mult = 1.35 if regime == "volatile" else (1.05 if regime == "range" else 1.20)
        rr = 1.85 if regime.startswith("trend") else (1.55 if regime == "range" else 1.70)
        risk = a * stop_mult
        if pick.direction.value == "BUY":
            sl, tp = entry - risk, entry + risk * rr
        else:
            sl, tp = entry + risk, entry - risk * rr

        candidate = pick.score
        reasons = [
            f"Strategy Selector: {pick.label}",
            f"Research validation hit rate: {candidate.valid_hit_rate:.1%} across {candidate.trades} signals",
            f"Current regime fit: {pick.regime_fit:.0%}",
            f"Analyst confirmation: {pick.analyst_agreement} supporting vs {pick.analyst_opposition} opposing",
        ]
        supporting = [v for v in votes if v.direction == pick.direction]
        reasons.extend(f"{v.agent}: {v.reason}" for v in supporting[:3])

        return TradeSignal(
            symbol=symbol,
            direction=pick.direction,
            entry=entry,
            stop_loss=float(sl),
            take_profit=float(tp),
            confidence=pick.probability_score,
            regime=regime,
            reasons=reasons,
            votes=votes,
            selected_strategy=pick.label,
            strategy_stats={
                "family": candidate.candidate.family,
                "params": candidate.candidate.params,
                "train_hit_rate": candidate.train_hit_rate,
                "valid_hit_rate": candidate.valid_hit_rate,
                "trades": candidate.trades,
                "research_score": candidate.score,
                "regime_fit": pick.regime_fit,
            },
        )
