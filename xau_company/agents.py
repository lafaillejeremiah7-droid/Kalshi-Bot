from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr, donchian, ema, roc, rsi, rolling_zscore
from .models import AgentVote, Direction
from .research import StrategyResearchAgent


class RegimeAgent:
    name = "Regime Analyst"

    def classify(self, df: pd.DataFrame) -> str:
        close = df["close"]
        e20, e50 = ema(close, 20), ema(close, 50)
        a = atr(df, 14)
        trend_strength = abs(e20.iloc[-1] - e50.iloc[-1]) / max(a.iloc[-1], 1e-9)
        vol_now = a.iloc[-1] / close.iloc[-1]
        vol_med = (a / close).rolling(120).median().iloc[-1]
        if vol_now > vol_med * 1.5:
            return "volatile"
        if trend_strength > 1.0:
            return "trend_up" if e20.iloc[-1] > e50.iloc[-1] else "trend_down"
        return "range"


class TrendAgent:
    name = "Trend Desk"
    family = "trend"

    def vote(self, df: pd.DataFrame, regime: str, lab: StrategyResearchAgent) -> AgentVote:
        close = df["close"]
        e20, e50, e100 = ema(close, 20), ema(close, 50), ema(close, 100)
        bullish = e20.iloc[-1] > e50.iloc[-1] > e100.iloc[-1]
        bearish = e20.iloc[-1] < e50.iloc[-1] < e100.iloc[-1]
        quality = lab.quality(self.family)
        regime_mult = 1.15 if regime.startswith("trend") else 0.80
        conf = float(np.clip(quality * regime_mult, 0.0, 0.95))
        if bullish:
            return AgentVote(self.name, Direction.BUY, conf, "EMA 20/50/100 bullish alignment")
        if bearish:
            return AgentVote(self.name, Direction.SELL, conf, "EMA 20/50/100 bearish alignment")
        return AgentVote(self.name, Direction.HOLD, 0.45, "No clean trend alignment")


class BreakoutAgent:
    name = "Breakout Desk"
    family = "breakout"

    def vote(self, df: pd.DataFrame, regime: str, lab: StrategyResearchAgent) -> AgentVote:
        close = df["close"]
        hi, lo = donchian(df, 20)
        m = roc(close, 5).iloc[-1]
        q = lab.quality(self.family)
        boost = 1.10 if regime in {"volatile", "trend_up", "trend_down"} else 0.85
        conf = float(np.clip(q * boost, 0, 0.95))
        if close.iloc[-1] > hi.iloc[-1] and m > 0:
            return AgentVote(self.name, Direction.BUY, conf, "20-bar upside breakout with positive momentum")
        if close.iloc[-1] < lo.iloc[-1] and m < 0:
            return AgentVote(self.name, Direction.SELL, conf, "20-bar downside breakout with negative momentum")
        return AgentVote(self.name, Direction.HOLD, 0.44, "No confirmed breakout")


class MeanReversionAgent:
    name = "Mean Reversion Desk"
    family = "mean_reversion"

    def vote(self, df: pd.DataFrame, regime: str, lab: StrategyResearchAgent) -> AgentVote:
        close = df["close"]
        rs = rsi(close, 14).iloc[-1]
        z = rolling_zscore(close, 30).iloc[-1]
        q = lab.quality(self.family)
        boost = 1.15 if regime == "range" else 0.65
        conf = float(np.clip(q * boost, 0, 0.92))
        if rs < 30 and z < -1.0:
            return AgentVote(self.name, Direction.BUY, conf, f"Oversold: RSI={rs:.1f}, z={z:.2f}")
        if rs > 70 and z > 1.0:
            return AgentVote(self.name, Direction.SELL, conf, f"Overbought: RSI={rs:.1f}, z={z:.2f}")
        return AgentVote(self.name, Direction.HOLD, 0.43, "No mean-reversion extreme")


class MomentumAgent:
    name = "Momentum Desk"
    family = "momentum"

    def vote(self, df: pd.DataFrame, regime: str, lab: StrategyResearchAgent) -> AgentVote:
        close = df["close"]
        m3, m10 = roc(close, 3).iloc[-1], roc(close, 10).iloc[-1]
        q = lab.quality(self.family)
        conf = float(np.clip(q * (1.08 if regime != "range" else 0.85), 0, 0.93))
        if m3 > 0 and m10 > 0:
            return AgentVote(self.name, Direction.BUY, conf, "Short and medium momentum agree upward")
        if m3 < 0 and m10 < 0:
            return AgentVote(self.name, Direction.SELL, conf, "Short and medium momentum agree downward")
        return AgentVote(self.name, Direction.HOLD, 0.42, "Momentum is mixed")


class PriceActionAgent:
    name = "Price Action Desk"

    def vote(self, df: pd.DataFrame, regime: str, lab: StrategyResearchAgent) -> AgentVote:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        body = abs(last.close - last.open)
        rng = max(last.high - last.low, 1e-9)
        body_ratio = body / rng
        bullish_engulf = last.close > last.open and prev.close < prev.open and last.close >= prev.open and last.open <= prev.close
        bearish_engulf = last.close < last.open and prev.close > prev.open and last.open >= prev.close and last.close <= prev.open
        if bullish_engulf and body_ratio > 0.55:
            return AgentVote(self.name, Direction.BUY, 0.62, "Bullish engulfing candle with strong body")
        if bearish_engulf and body_ratio > 0.55:
            return AgentVote(self.name, Direction.SELL, 0.62, "Bearish engulfing candle with strong body")
        return AgentVote(self.name, Direction.HOLD, 0.40, "No high-quality candle trigger")


class VolatilityGuardAgent:
    name = "Volatility Guard"

    def vote(self, df: pd.DataFrame, regime: str, lab: StrategyResearchAgent) -> AgentVote:
        a = atr(df, 14)
        ratio = float(a.iloc[-1] / df.close.iloc[-1])
        baseline = float((a / df.close).rolling(200).median().iloc[-1])
        if baseline and ratio > baseline * 2.2:
            return AgentVote(self.name, Direction.HOLD, 0.95, "Extreme ATR expansion; veto new trade", {"veto": True})
        return AgentVote(self.name, Direction.HOLD, 0.55, "Volatility within acceptable range", {"veto": False})


class StructureAgent:
    name = "Market Structure Desk"

    def vote(self, df: pd.DataFrame, regime: str, lab: StrategyResearchAgent) -> AgentVote:
        h = df.high.rolling(5).max()
        l = df.low.rolling(5).min()
        higher_highs = h.iloc[-1] > h.iloc[-6]
        higher_lows = l.iloc[-1] > l.iloc[-6]
        lower_highs = h.iloc[-1] < h.iloc[-6]
        lower_lows = l.iloc[-1] < l.iloc[-6]
        if higher_highs and higher_lows:
            return AgentVote(self.name, Direction.BUY, 0.61, "Higher-high/higher-low structure")
        if lower_highs and lower_lows:
            return AgentVote(self.name, Direction.SELL, 0.61, "Lower-high/lower-low structure")
        return AgentVote(self.name, Direction.HOLD, 0.42, "Structure is mixed")


class SessionAgent:
    name = "Session Desk"

    def vote(self, df: pd.DataFrame, regime: str, lab: StrategyResearchAgent) -> AgentVote:
        ts = df.datetime.iloc[-1]
        hour = int(ts.hour)
        liquid = 7 <= hour <= 17
        return AgentVote(
            self.name,
            Direction.HOLD,
            0.60 if liquid else 0.48,
            "Major London/NY liquidity window" if liquid else "Lower-liquidity time window",
            {"liquidity_ok": liquid},
        )
