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
from .context import MacroContextAgent, MultiTimeframeAgent, NewsRiskAgent
from .indicators import atr
from .models import AgentVote, TradeSignal
from .research import StrategyResearchAgent
from .selector import StrategySelectorAgent


class BossAgent:
    """CEO: research -> market diagnosis -> strategy selection -> risk -> signal."""

    def __init__(
        self,
        lab: StrategyResearchAgent,
        min_confidence: float = 0.72,
        min_consensus: int = 3,
        high_impact_events_utc: str = "",
        news_block_minutes: int = 20,
    ) -> None:
        self.lab = lab
        self.min_confidence = min_confidence
        self.regime_agent = RegimeAgent()
        self.selector = StrategySelectorAgent(min_probability=min_confidence, min_agreement=max(2, min_consensus - 1))
        self.multi_timeframe = MultiTimeframeAgent()
        self.macro_context = MacroContextAgent()
        self.news_risk = NewsRiskAgent.from_csv(high_impact_events_utc, news_block_minutes)
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

    @staticmethod
    def _normalize_frames(frames: dict[str, pd.DataFrame] | pd.DataFrame) -> dict[str, pd.DataFrame]:
        if isinstance(frames, pd.DataFrame):
            # Backward-compatible path for tests/one-frame use.
            return {"5min": frames, "15min": frames, "1h": frames, "4h": frames}
        return frames

    @staticmethod
    def _pick_frame(frames: dict[str, pd.DataFrame], preferences: tuple[str, ...]) -> pd.DataFrame | None:
        for key in preferences:
            df = frames.get(key)
            if df is not None and not df.empty:
                return df
        return None

    def decide(
        self,
        symbol: str,
        frames: dict[str, pd.DataFrame] | pd.DataFrame,
        dxy: pd.DataFrame | None = None,
        yield_df: pd.DataFrame | None = None,
    ) -> TradeSignal | None:
        frame_map = self._normalize_frames(frames)
        core = self._pick_frame(frame_map, ("15min", "5min", "1h", "4h", "1min"))
        execution = self._pick_frame(frame_map, ("5min", "1min", "15min"))
        if core is None or execution is None or len(core) < 220 or not self.lab.catalog:
            return None

        regime = self.regime_agent.classify(core)

        # Existing specialist desks diagnose the core timeframe.
        votes: list[AgentVote] = [desk.vote(core, regime, self.lab) for desk in self.desks]
        # Independent timeframe employees diagnose different horizons.
        votes.extend(self.multi_timeframe.analyze(frame_map))
        # Macro context is evidence rather than a hard trading rule.
        votes.extend(self.macro_context.analyze(dxy, yield_df))
        # High-impact event blackout has hard veto authority.
        votes.append(self.news_risk.vote())

        if any(v.metadata.get("veto") for v in votes):
            return None

        # The researched strategy is evaluated on the 15m/core frame; analysts do
        # not directly create the trade, they determine whether that strategy fits now.
        pick = self.selector.select(core, regime, votes, self.lab)
        if pick is None:
            return None

        entry = float(execution.close.iloc[-1])
        a = float(atr(execution, 14).iloc[-1])
        if not np.isfinite(a) or a <= 0:
            return None

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
            f"Multi-timeframe alignment: {pick.timeframe_alignment:.0%}",
            f"USD/yield macro alignment: {pick.macro_alignment:.0%}",
            f"Analyst confirmation: {pick.analyst_agreement} supporting vs {pick.analyst_opposition} opposing",
        ]
        supporting = [v for v in votes if v.direction == pick.direction]
        reasons.extend(f"{v.agent}: {v.reason}" for v in supporting[:4])

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
                "timeframe_alignment": pick.timeframe_alignment,
                "macro_alignment": pick.macro_alignment,
            },
        )
