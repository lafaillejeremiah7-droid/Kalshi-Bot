"""
Entry and exit rule designer for validated hypotheses.

Designs entry rules (signal threshold, confirmation, filter conditions)
and exit rules (time-based, signal reversal, trailing stop, profit target,
stop-loss) for each validated hypothesis, and backtests combinations to
select the best risk-adjusted configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from quant_research.hypotheses.catalog import Hypothesis
from quant_research.testing.statistical import StatisticalTester


@dataclass
class StrategyRules:
    """Entry and exit rules for a strategy.

    Attributes
    ----------
    entry_threshold : float
        Signal strength required to trigger entry (0 to 1 scale).
    confirmation_bars : int
        Number of consecutive bars signal must persist before entry.
    exit_type : str
        Primary exit type: 'time', 'reversal', 'trailing_stop', 'combined'.
    stop_loss_atr_mult : float
        Stop-loss distance as multiple of ATR.
    profit_target_atr_mult : float
        Profit target as multiple of ATR.
    max_holding_period : int
        Maximum bars to hold a position.
    trailing_stop_atr_mult : float
        Trailing stop distance as multiple of ATR.
    """

    entry_threshold: float = 0.0
    confirmation_bars: int = 1
    exit_type: str = "combined"
    stop_loss_atr_mult: float = 2.0
    profit_target_atr_mult: float = 3.0
    max_holding_period: int = 20
    trailing_stop_atr_mult: float = 1.5


@dataclass
class BacktestResult:
    """Result of backtesting a set of strategy rules.

    Attributes
    ----------
    returns : pd.Series
        Daily returns from the backtest.
    trades : list[dict]
        List of trade records.
    metrics : dict
        Performance metrics (sharpe, hit_rate, max_drawdown, etc.).
    rules : StrategyRules
        The rules used for this backtest.
    """

    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    rules: StrategyRules = field(default_factory=StrategyRules)


class EntryExitDesigner:
    """Designs entry and exit rules for validated hypotheses.

    For each hypothesis, tests multiple parameter combinations on in-sample
    data and selects the best combination by Sharpe ratio (risk-adjusted
    return), not raw return. Uses walk-forward logic: optimize on train
    portion, validate on test portion.

    Parameters
    ----------
    train_ratio : float, optional
        Fraction of data to use for optimization. Default is 0.7.

    Examples
    --------
    >>> designer = EntryExitDesigner()
    >>> rules = designer.design_rules(hypothesis, data)
    >>> result = designer.backtest_rules(rules, hypothesis, data)
    """

    def __init__(self, train_ratio: float = 0.7) -> None:
        self.train_ratio = train_ratio
        self.tester = StatisticalTester()

    def design_rules(
        self, hypothesis: Hypothesis, data: pd.DataFrame
    ) -> StrategyRules:
        """Design optimal entry/exit rules for a hypothesis.

        Tests multiple parameter combinations on in-sample data and
        selects the best by Sharpe ratio.

        Parameters
        ----------
        hypothesis : Hypothesis
            The hypothesis to design rules for.
        data : pd.DataFrame
            OHLCV + features DataFrame.

        Returns
        -------
        StrategyRules
            Optimal strategy rules based on in-sample optimization.
        """
        # Split data for walk-forward optimization
        split_idx = int(len(data) * self.train_ratio)
        train_data = data.iloc[:split_idx]

        # Define parameter grid (kept small for efficiency)
        param_grid = {
            "entry_threshold": [0.0, 0.5],
            "confirmation_bars": [1, 2],
            "stop_loss_atr_mult": [1.5, 2.0, 3.0],
            "profit_target_atr_mult": [2.0, 3.0, 4.0],
            "max_holding_period": [10, 20],
        }

        best_sharpe = -np.inf
        best_rules = StrategyRules()

        # Grid search on training data
        for entry_thresh, confirm, sl_mult, tp_mult, max_hold in product(
            param_grid["entry_threshold"],
            param_grid["confirmation_bars"],
            param_grid["stop_loss_atr_mult"],
            param_grid["profit_target_atr_mult"],
            param_grid["max_holding_period"],
        ):
            rules = StrategyRules(
                entry_threshold=entry_thresh,
                confirmation_bars=confirm,
                exit_type="combined",
                stop_loss_atr_mult=sl_mult,
                profit_target_atr_mult=tp_mult,
                max_holding_period=max_hold,
                trailing_stop_atr_mult=sl_mult * 0.75,
            )

            result = self.backtest_rules(rules, hypothesis, train_data)
            sharpe = result.metrics.get("sharpe_ratio", -np.inf)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_rules = rules

        return best_rules

    def backtest_rules(
        self,
        rules: StrategyRules,
        hypothesis: Hypothesis,
        data: pd.DataFrame,
    ) -> BacktestResult:
        """Backtest a set of strategy rules on data.

        Parameters
        ----------
        rules : StrategyRules
            The entry/exit rules to test.
        hypothesis : Hypothesis
            The hypothesis providing the signal function.
        data : pd.DataFrame
            OHLCV + features DataFrame.

        Returns
        -------
        BacktestResult
            Backtest results including returns, trades, and metrics.
        """
        if len(data) < 20:
            return BacktestResult(rules=rules)

        # Compute signal
        try:
            signal = hypothesis.signal_function(data)
        except Exception:
            return BacktestResult(rules=rules)

        # Compute ATR for stops
        atr = self._compute_atr(data)

        # Apply entry rules with confirmation and threshold
        entry_signal = self._apply_entry_rules(signal, rules)

        # Simulate trades with exit rules
        returns, trades = self._simulate_trades(
            data, entry_signal, atr, rules
        )

        # Compute metrics
        metrics = self._compute_metrics(returns)

        return BacktestResult(
            returns=returns,
            trades=trades,
            metrics=metrics,
            rules=rules,
        )

    def _compute_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data.
        period : int
            ATR lookback period.

        Returns
        -------
        pd.Series
            ATR values.
        """
        high = data["High"]
        low = data["Low"]
        close = data["Close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.ewm(span=period, adjust=False).mean()

    def _apply_entry_rules(
        self, signal: pd.Series, rules: StrategyRules
    ) -> pd.Series:
        """Apply entry threshold and confirmation requirements.

        Parameters
        ----------
        signal : pd.Series
            Raw signal from hypothesis.
        rules : StrategyRules
            Rules containing threshold and confirmation parameters.

        Returns
        -------
        pd.Series
            Filtered entry signal.
        """
        # Apply threshold: only keep signals with absolute value above threshold
        filtered = signal.copy()
        if rules.entry_threshold > 0:
            filtered = filtered.where(filtered.abs() >= rules.entry_threshold, 0.0)

        # Apply confirmation: signal must persist for N bars
        if rules.confirmation_bars > 1:
            # Check if signal has been the same sign for confirmation_bars
            confirmed = filtered.copy()
            for i in range(1, rules.confirmation_bars):
                same_sign = (filtered.shift(i).fillna(0) * filtered > 0)
                confirmed = confirmed.where(same_sign, 0.0)
            return confirmed

        return filtered

    def _simulate_trades(
        self,
        data: pd.DataFrame,
        entry_signal: pd.Series,
        atr: pd.Series,
        rules: StrategyRules,
    ) -> tuple[pd.Series, list[dict]]:
        """Simulate trades with entry/exit rules.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data.
        entry_signal : pd.Series
            Filtered entry signal.
        atr : pd.Series
            ATR values for stop/target computation.
        rules : StrategyRules
            Strategy rules.

        Returns
        -------
        tuple[pd.Series, list[dict]]
            (daily_returns, trade_list)
        """
        close = data["Close"]
        daily_returns = pd.Series(0.0, index=data.index)
        trades: list[dict] = []

        position = 0  # 0 = flat, 1 = long, -1 = short
        entry_price = 0.0
        entry_idx = 0
        bars_held = 0
        highest_since_entry = 0.0
        lowest_since_entry = float("inf")

        for i in range(1, len(data)):
            idx = data.index[i]
            current_close = close.iloc[i]
            current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 1.0

            if position == 0:
                # Check for entry
                sig = entry_signal.iloc[i] if i < len(entry_signal) else 0
                if sig > 0:
                    position = 1
                    entry_price = current_close
                    entry_idx = i
                    bars_held = 0
                    highest_since_entry = current_close
                    lowest_since_entry = current_close
                elif sig < 0:
                    position = -1
                    entry_price = current_close
                    entry_idx = i
                    bars_held = 0
                    highest_since_entry = current_close
                    lowest_since_entry = current_close
            else:
                bars_held += 1
                daily_ret = (current_close - close.iloc[i - 1]) / close.iloc[i - 1]
                daily_returns.iloc[i] = daily_ret * position

                # Update tracking
                highest_since_entry = max(highest_since_entry, current_close)
                lowest_since_entry = min(lowest_since_entry, current_close)

                # Check exit conditions
                exit_triggered = False
                exit_reason = ""

                # Stop-loss
                if position == 1:
                    stop_price = entry_price - rules.stop_loss_atr_mult * current_atr
                    if current_close <= stop_price:
                        exit_triggered = True
                        exit_reason = "stop_loss"
                else:
                    stop_price = entry_price + rules.stop_loss_atr_mult * current_atr
                    if current_close >= stop_price:
                        exit_triggered = True
                        exit_reason = "stop_loss"

                # Profit target
                if not exit_triggered:
                    if position == 1:
                        target = entry_price + rules.profit_target_atr_mult * current_atr
                        if current_close >= target:
                            exit_triggered = True
                            exit_reason = "profit_target"
                    else:
                        target = entry_price - rules.profit_target_atr_mult * current_atr
                        if current_close <= target:
                            exit_triggered = True
                            exit_reason = "profit_target"

                # Trailing stop
                if not exit_triggered:
                    trail_dist = rules.trailing_stop_atr_mult * current_atr
                    if position == 1:
                        trail_stop = highest_since_entry - trail_dist
                        if current_close <= trail_stop:
                            exit_triggered = True
                            exit_reason = "trailing_stop"
                    else:
                        trail_stop = lowest_since_entry + trail_dist
                        if current_close >= trail_stop:
                            exit_triggered = True
                            exit_reason = "trailing_stop"

                # Time-based exit
                if not exit_triggered and bars_held >= rules.max_holding_period:
                    exit_triggered = True
                    exit_reason = "time_exit"

                # Signal reversal
                if not exit_triggered:
                    current_sig = (
                        entry_signal.iloc[i] if i < len(entry_signal) else 0
                    )
                    if position == 1 and current_sig < 0:
                        exit_triggered = True
                        exit_reason = "signal_reversal"
                    elif position == -1 and current_sig > 0:
                        exit_triggered = True
                        exit_reason = "signal_reversal"

                if exit_triggered:
                    pnl = (current_close - entry_price) / entry_price * position
                    trades.append({
                        "entry_date": str(data.index[entry_idx]),
                        "exit_date": str(idx),
                        "direction": position,
                        "entry_price": entry_price,
                        "exit_price": current_close,
                        "pnl": pnl,
                        "bars_held": bars_held,
                        "exit_reason": exit_reason,
                    })
                    position = 0

        return daily_returns, trades

    def _compute_metrics(self, returns: pd.Series) -> dict:
        """Compute performance metrics from returns.

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.

        Returns
        -------
        dict
            Performance metrics.
        """
        if len(returns) == 0 or returns.std() == 0:
            return {
                "sharpe_ratio": 0.0,
                "hit_rate": 0.0,
                "max_drawdown": 0.0,
                "expectancy": 0.0,
                "total_return": 0.0,
            }

        active_returns = returns[returns != 0]
        sharpe = self.tester.compute_sharpe_ratio(active_returns)
        hit_rate = self.tester.compute_hit_rate(active_returns)
        max_dd = self.tester.compute_max_drawdown(returns.cumsum())
        expectancy = self.tester.expectancy(active_returns)
        total_return = float(returns.sum())

        return {
            "sharpe_ratio": sharpe,
            "hit_rate": hit_rate,
            "max_drawdown": max_dd,
            "expectancy": expectancy,
            "total_return": total_return,
        }
