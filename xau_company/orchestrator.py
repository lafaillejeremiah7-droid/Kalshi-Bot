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
from .quality import INTERVAL_MINUTES
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
        research_interval: str = "15min",
    ) -> None:
        self.lab = lab
        self.min_confidence = min_confidence
        self.research_interval = research_interval
        self.regime_agent = RegimeAgent()
        self.selector = StrategySelectorAgent(
            min_probability=min_confidence,
            min_agreement=max(1, int(min_consensus)),
        )
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
        entry_price: float | None = None,
    ) -> TradeSignal | None:
        frame_map = self._normalize_frames(frames)
        # The strategy was researched on RESEARCH_INTERVAL, so live direction and
        # ATR risk geometry must use that same timeframe whenever available.
        core = self._pick_frame(frame_map, (self.research_interval, "15min"))
        execution = self._pick_frame(frame_map, ("5min", "1min"))
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

        if entry_price is None or not np.isfinite(float(entry_price)) or float(entry_price) <= 0:
            return None
        entry = float(entry_price)
        a = float(atr(core, 14).iloc[-1])
        if not np.isfinite(a) or a <= 0:
            return None

        # Match the exact stop and reward/risk assumptions used by the research
        # lifecycle backtester instead of changing geometry after selection.
        stop_mult = float(self.lab.backtester.stop_atr)
        rr = float(self.lab.backtester.reward_risk)
        risk = a * stop_mult
        if pick.direction.value == "BUY":
            sl, tp = entry - risk, entry + risk * rr
        else:
            sl, tp = entry + risk, entry - risk * rr

        candidate = pick.score
        wf_hit = candidate.walk_forward_hit_rate if candidate.walk_forward_hit_rate > 0 else candidate.valid_hit_rate
        oos_trades = sum(max(0, int(n)) for n in candidate.regime_trades.values())
        if oos_trades <= 0:
            oos_trades = int(candidate.trades)
        holding_bars = int(self.lab.HORIZONS.get(candidate.candidate.strategy_id, 4))
        research_minutes = int(INTERVAL_MINUTES.get(self.research_interval, 15))
        max_holding_minutes = holding_bars * research_minutes

        reasons = [
            f"Strategy Selector: {pick.label}",
            f"Walk-forward OOS hit rate: {wf_hit:.1%} over {candidate.folds} folds; stability {pick.walk_forward_stability:.0%}",
            f"Current {regime} history: {pick.regime_history:.1%} over {pick.regime_samples} OOS trades",
            f"Lifecycle: PF {candidate.profit_factor:.2f}, avg R {candidate.avg_r_multiple:+.2f}, max DD {candidate.max_drawdown_r:.2f}R, worst streak {candidate.max_loss_streak}",
            f"OOS net expectancy per trade: {candidate.expectancy:.4%}",
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
                "strategy_id": candidate.candidate.strategy_id,
                "train_hit_rate": candidate.train_hit_rate,
                "valid_hit_rate": candidate.valid_hit_rate,
                "walk_forward_hit_rate": candidate.walk_forward_hit_rate,
                "walk_forward_std": candidate.walk_forward_std,
                "folds": candidate.folds,
                "expectancy": candidate.expectancy,
                "profit_factor": candidate.profit_factor,
                "avg_r_multiple": candidate.avg_r_multiple,
                "max_drawdown_r": candidate.max_drawdown_r,
                "max_loss_streak": candidate.max_loss_streak,
                "backtest_model": candidate.backtest_model,
                "trades": candidate.trades,
                "oos_trades": oos_trades,
                "research_score": candidate.score,
                "regime_fit": pick.regime_fit,
                "regime_history": pick.regime_history,
                "regime_samples": pick.regime_samples,
                "timeframe_alignment": pick.timeframe_alignment,
                "macro_alignment": pick.macro_alignment,
                "lifecycle_quality": pick.lifecycle_quality,
                "research_interval": self.research_interval,
                "max_holding_bars": holding_bars,
                "max_holding_minutes": max_holding_minutes,
                "resolution_interval_minutes": 1,
                "stop_atr": stop_mult,
                "reward_risk": rr,
            },
        )
