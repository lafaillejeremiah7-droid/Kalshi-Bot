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
        votes: list[AgentVote] = [desk.vote(core, regime, self.lab) for desk in self.desks]
        votes.extend(self.multi_timeframe.analyze(frame_map))
        votes.extend(self.macro_context.analyze(dxy, yield_df))
        votes.append(self.news_risk.vote())

        if any(v.metadata.get("veto") for v in votes):
            return None

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
        wf_hit = candidate.walk_forward_hit_rate if candidate.walk_forward_hit_rate > 0 else candidate.valid_hit_rate
        reasons = [
            f"Strategy Selector: {pick.label}",
            f"Walk-forward OOS hit rate: {wf_hit:.1%} over {candidate.folds} folds; stability {pick.walk_forward_stability:.0%}",
            f"Current {regime} history: {pick.regime_history:.1%} over {pick.regime_samples} OOS signals",
            f"Profit factor: {candidate.profit_factor:.2f}; OOS expectancy: {candidate.expectancy:.4%}",
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
                "walk_forward_hit_rate": candidate.walk_forward_hit_rate,
                "walk_forward_std": candidate.walk_forward_std,
                "folds": candidate.folds,
                "expectancy": candidate.expectancy,
                "profit_factor": candidate.profit_factor,
                "trades": candidate.trades,
                "research_score": candidate.score,
                "regime_fit": pick.regime_fit,
                "regime_history": pick.regime_history,
                "regime_samples": pick.regime_samples,
                "timeframe_alignment": pick.timeframe_alignment,
                "macro_alignment": pick.macro_alignment,
            },
        )
