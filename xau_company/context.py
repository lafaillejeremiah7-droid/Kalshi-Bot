from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .indicators import atr, ema, roc
from .models import AgentVote, Direction


class MultiTimeframeAgent:
    """Independent desks for execution, structure, trend and macro trend timeframes."""

    ROLES = {
        "1min": "1m Execution Desk",
        "5min": "5m Execution Desk",
        "15min": "15m Structure Desk",
        "1h": "1h Trend Desk",
        "4h": "4h Macro Trend Desk",
    }

    WEIGHTS = {"1min": 0.82, "5min": 0.90, "15min": 1.00, "1h": 1.08, "4h": 1.12}

    def vote_frame(self, timeframe: str, df: pd.DataFrame) -> AgentVote:
        name = self.ROLES.get(timeframe, f"{timeframe} Desk")
        if df is None or len(df) < 60:
            return AgentVote(name, Direction.HOLD, 0.30, "Insufficient timeframe history", {"timeframe": timeframe})

        close = df["close"]
        fast_span, slow_span = (9, 21) if timeframe in {"1min", "5min"} else (20, 50)
        fast, slow = ema(close, fast_span), ema(close, slow_span)
        momentum = float(roc(close, 5).iloc[-1])
        a = float(atr(df, 14).iloc[-1])
        separation = abs(float(fast.iloc[-1] - slow.iloc[-1])) / max(a, 1e-9)
        strength = float(np.clip(0.50 + separation * 0.12 + min(abs(momentum) * 30, 0.12), 0.50, 0.82))
        strength = float(np.clip(strength * self.WEIGHTS.get(timeframe, 1.0), 0.0, 0.90))

        bullish = fast.iloc[-1] > slow.iloc[-1] and momentum > 0
        bearish = fast.iloc[-1] < slow.iloc[-1] and momentum < 0
        metadata = {"timeframe": timeframe, "momentum": momentum, "ema_separation_atr": separation}
        if bullish:
            return AgentVote(name, Direction.BUY, strength, f"{fast_span}/{slow_span} EMA trend and momentum are bullish", metadata)
        if bearish:
            return AgentVote(name, Direction.SELL, strength, f"{fast_span}/{slow_span} EMA trend and momentum are bearish", metadata)
        return AgentVote(name, Direction.HOLD, 0.42, "Trend and momentum do not agree", metadata)

    def analyze(self, frames: dict[str, pd.DataFrame]) -> list[AgentVote]:
        return [self.vote_frame(tf, frames.get(tf)) for tf in self.ROLES if tf in frames]


class MacroContextAgent:
    """Turns USD and Treasury-yield movement into independent XAU/USD context votes."""

    def _inverse_vote(self, name: str, df: pd.DataFrame | None) -> AgentVote:
        if df is None or len(df) < 35:
            return AgentVote(name, Direction.HOLD, 0.30, "Macro feed unavailable or insufficient history")
        close = df["close"]
        short = ema(close, 10).iloc[-1]
        long = ema(close, 30).iloc[-1]
        move = float(roc(close, 5).iloc[-1])
        magnitude = min(abs(move) * 35, 0.15)
        conf = float(np.clip(0.52 + magnitude, 0.52, 0.70))
        # Gold commonly trades inversely to USD strength and nominal yields; this is
        # evidence, not a hard law, so these desks never have veto authority.
        if short > long and move > 0:
            return AgentVote(name, Direction.SELL, conf, f"Macro series strengthening ({move:+.2%}); headwind for gold")
        if short < long and move < 0:
            return AgentVote(name, Direction.BUY, conf, f"Macro series weakening ({move:+.2%}); tailwind for gold")
        return AgentVote(name, Direction.HOLD, 0.40, "Macro series is mixed")

    def analyze(self, dxy: pd.DataFrame | None, yield_df: pd.DataFrame | None) -> list[AgentVote]:
        return [
            self._inverse_vote("USD Strength Desk", dxy),
            self._inverse_vote("Treasury Yield Desk", yield_df),
        ]


class NewsRiskAgent:
    """Hard veto around configured high-impact event timestamps (UTC)."""

    name = "Economic News Risk Desk"

    def __init__(self, event_times_utc: tuple[datetime, ...] = (), block_minutes: int = 20) -> None:
        self.event_times_utc = event_times_utc
        self.block_minutes = max(0, block_minutes)

    @classmethod
    def from_csv(cls, raw: str, block_minutes: int = 20) -> "NewsRiskAgent":
        events: list[datetime] = []
        for item in (x.strip() for x in raw.split(",") if x.strip()):
            try:
                dt = datetime.fromisoformat(item.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                events.append(dt.astimezone(timezone.utc))
            except ValueError:
                continue
        return cls(tuple(events), block_minutes)

    def vote(self, now: datetime | None = None) -> AgentVote:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        nearest: tuple[float, datetime] | None = None
        for event in self.event_times_utc:
            minutes = abs((event - now).total_seconds()) / 60.0
            if nearest is None or minutes < nearest[0]:
                nearest = (minutes, event)
        if nearest and nearest[0] <= self.block_minutes:
            event = nearest[1].isoformat()
            return AgentVote(self.name, Direction.HOLD, 0.99, f"High-impact event blackout around {event}", {"veto": True})
        return AgentVote(self.name, Direction.HOLD, 0.55, "No configured high-impact event in blackout window", {"veto": False})
