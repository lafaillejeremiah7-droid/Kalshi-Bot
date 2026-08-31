from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeOutcome:
    signal_index: int
    entry_index: int
    exit_index: int
    direction: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    net_return: float
    r_multiple: float
    won: bool
    exit_reason: str


class TradeLifecycleBacktester:
    """Conservative OHLC trade simulator for strategy research.

    Rules are intentionally stricter than the previous future-close shortcut:
    - a signal is only actionable after its candle closes;
    - entry occurs at the next candle open;
    - spread and slippage are paid on both entry and exit;
    - stop and target are fixed from ATR known on the signal candle;
    - if a candle touches both stop and target, the stop is assumed first;
    - only one trade may be open at a time;
    - trades still open at the holding limit exit at that candle's close.
    """

    def __init__(
        self,
        spread_bps: float = 1.5,
        slippage_bps: float = 0.5,
        stop_atr: float = 1.20,
        reward_risk: float = 1.70,
    ) -> None:
        self.spread_bps = max(0.0, float(spread_bps))
        self.slippage_bps = max(0.0, float(slippage_bps))
        self.stop_atr = max(0.10, float(stop_atr))
        self.reward_risk = max(0.25, float(reward_risk))

    @property
    def side_cost(self) -> float:
        # spread_bps represents the full bid/ask spread; half is paid on each side.
        return (self.spread_bps / 2.0 + self.slippage_bps) / 10_000.0

    def _entry_fill(self, raw_open: float, direction: int) -> float:
        if direction > 0:
            return raw_open * (1.0 + self.side_cost)
        return raw_open * (1.0 - self.side_cost)

    def _exit_fill(self, raw_price: float, direction: int) -> float:
        if direction > 0:
            return raw_price * (1.0 - self.side_cost)
        return raw_price * (1.0 + self.side_cost)

    def simulate(
        self,
        df: pd.DataFrame,
        signal: pd.Series,
        atr_values: pd.Series,
        max_holding: int,
    ) -> list[TradeOutcome]:
        if len(df) < 3 or signal.empty:
            return []

        max_holding = max(1, int(max_holding))
        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        sig = signal.to_numpy(dtype=np.int8)
        atr_np = atr_values.to_numpy(dtype=float)

        trades: list[TradeOutcome] = []
        occupied_until = -1
        n = len(df)

        for signal_idx in np.flatnonzero(sig):
            signal_idx = int(signal_idx)
            if signal_idx < occupied_until or signal_idx + 1 >= n:
                continue

            direction = 1 if sig[signal_idx] > 0 else -1
            known_atr = float(atr_np[signal_idx])
            raw_open = float(opens[signal_idx + 1])
            if not np.isfinite(known_atr) or known_atr <= 0 or not np.isfinite(raw_open) or raw_open <= 0:
                continue

            entry_idx = signal_idx + 1
            entry = self._entry_fill(raw_open, direction)
            risk_distance = known_atr * self.stop_atr
            if direction > 0:
                stop = entry - risk_distance
                target = entry + risk_distance * self.reward_risk
            else:
                stop = entry + risk_distance
                target = entry - risk_distance * self.reward_risk

            last_idx = min(n - 1, entry_idx + max_holding - 1)
            exit_idx = last_idx
            raw_exit = float(closes[last_idx])
            reason = "timeout"

            for bar_idx in range(entry_idx, last_idx + 1):
                hi = float(highs[bar_idx])
                lo = float(lows[bar_idx])
                if not np.isfinite(hi) or not np.isfinite(lo):
                    continue

                if direction > 0:
                    stop_hit = lo <= stop
                    target_hit = hi >= target
                else:
                    stop_hit = hi >= stop
                    target_hit = lo <= target

                # OHLC data cannot reveal intrabar ordering. Assuming the stop was
                # hit first prevents optimistic backtests when both levels trade.
                if stop_hit:
                    exit_idx = bar_idx
                    raw_exit = stop
                    reason = "stop"
                    break
                if target_hit:
                    exit_idx = bar_idx
                    raw_exit = target
                    reason = "target"
                    break

            exit_fill = self._exit_fill(raw_exit, direction)
            pnl = (exit_fill - entry) * direction
            net_return = pnl / max(abs(entry), 1e-12)
            r_multiple = pnl / max(risk_distance, 1e-12)

            trades.append(
                TradeOutcome(
                    signal_index=signal_idx,
                    entry_index=entry_idx,
                    exit_index=exit_idx,
                    direction=direction,
                    entry_price=float(entry),
                    exit_price=float(exit_fill),
                    stop_price=float(stop),
                    target_price=float(target),
                    net_return=float(net_return),
                    r_multiple=float(r_multiple),
                    won=bool(pnl > 0),
                    exit_reason=reason,
                )
            )
            occupied_until = exit_idx

        return trades

    @staticmethod
    def profit_factor(trades: list[TradeOutcome]) -> float:
        if not trades:
            return 1.0
        returns = np.asarray([t.net_return for t in trades], dtype=float)
        gross_profit = float(returns[returns > 0].sum())
        gross_loss = float(-returns[returns < 0].sum())
        if gross_loss <= 1e-12:
            return 4.0 if gross_profit > 0 else 1.0
        return float(np.clip(gross_profit / gross_loss, 0.0, 4.0))

    @staticmethod
    def max_drawdown_r(trades: list[TradeOutcome]) -> float:
        if not trades:
            return 0.0
        equity = np.cumsum(np.asarray([t.r_multiple for t in trades], dtype=float))
        peaks = np.maximum.accumulate(np.concatenate(([0.0], equity)))
        path = np.concatenate(([0.0], equity))
        drawdowns = peaks - path
        return float(max(0.0, np.max(drawdowns)))

    @staticmethod
    def max_loss_streak(trades: list[TradeOutcome]) -> int:
        worst = 0
        current = 0
        for trade in trades:
            if trade.won:
                current = 0
            else:
                current += 1
                worst = max(worst, current)
        return worst
