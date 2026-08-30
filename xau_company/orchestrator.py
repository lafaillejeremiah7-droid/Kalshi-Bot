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
from .models import AgentVote, Direction, TradeSignal
from .research import StrategyResearchAgent


class BossAgent:
    """CEO/orchestrator: gathers departments, applies vetoes, sets risk and emits one decision."""

    def __init__(self, lab: StrategyResearchAgent, min_confidence: float = 0.72, min_consensus: int = 3) -> None:
        self.lab = lab
        self.min_confidence = min_confidence
        self.min_consensus = min_consensus
        self.regime_agent = RegimeAgent()
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
        if len(df) < 220:
            return None
        regime = self.regime_agent.classify(df)
        votes: list[AgentVote] = [desk.vote(df, regime, self.lab) for desk in self.desks]
        if any(v.metadata.get("veto") for v in votes):
            return None

        directional = [v for v in votes if v.direction != Direction.HOLD]
        if not directional:
            return None
        buy = [v for v in directional if v.direction == Direction.BUY]
        sell = [v for v in directional if v.direction == Direction.SELL]
        side = Direction.BUY if sum(v.confidence for v in buy) >= sum(v.confidence for v in sell) else Direction.SELL
        agreeing = buy if side == Direction.BUY else sell
        opposing = sell if side == Direction.BUY else buy
        if len(agreeing) < self.min_consensus:
            return None

        raw = sum(v.confidence for v in agreeing) / len(agreeing)
        opposition = sum(v.confidence for v in opposing) / max(1, len(agreeing))
        consensus_bonus = min(0.10, 0.025 * max(0, len(agreeing) - self.min_consensus))
        confidence = float(np.clip(raw - 0.35 * opposition + consensus_bonus, 0, 0.97))
        if confidence < self.min_confidence:
            return None

        entry = float(df.close.iloc[-1])
        a = float(atr(df, 14).iloc[-1])
        if not np.isfinite(a) or a <= 0:
            return None
        stop_mult = 1.35 if regime == "volatile" else (1.05 if regime == "range" else 1.20)
        rr = 1.8 if regime.startswith("trend") else 1.6
        risk = a * stop_mult
        if side == Direction.BUY:
            sl, tp = entry - risk, entry + risk * rr
        else:
            sl, tp = entry + risk, entry - risk * rr
        reasons = [f"{v.agent}: {v.reason}" for v in agreeing[:5]]
        return TradeSignal(symbol, side, entry, float(sl), float(tp), confidence, regime, reasons, votes)
