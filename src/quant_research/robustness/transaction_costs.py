"""
Transaction cost modeling for strategy profitability assessment.

Models realistic market frictions including commission, bid-ask spread,
slippage (volume-dependent), and market impact. Used to determine whether
a trading edge survives after real-world costs are applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research.hypotheses.catalog import Hypothesis
from quant_research.testing.statistical import StatisticalTester


@dataclass
class CostAdjustedResult:
    """Result of transaction cost analysis for one hypothesis.

    Attributes
    ----------
    hypothesis_id : str
        ID of the hypothesis tested.
    survives_costs : bool
        Whether the edge survives after costs.
    gross_sharpe : float
        Sharpe ratio before costs.
    net_sharpe : float
        Sharpe ratio after costs.
    gross_expectancy : float
        Expectancy before costs.
    net_expectancy : float
        Expectancy after costs.
    gross_profit_factor : float
        Profit factor before costs.
    net_profit_factor : float
        Profit factor after costs.
    total_cost_per_trade : float
        Average total cost per round-trip trade.
    n_trades : int
        Number of trades.
    turnover_annual : float
        Estimated annual turnover (trades per year).
    """

    hypothesis_id: str
    survives_costs: bool
    gross_sharpe: float = 0.0
    net_sharpe: float = 0.0
    gross_expectancy: float = 0.0
    net_expectancy: float = 0.0
    gross_profit_factor: float = 0.0
    net_profit_factor: float = 0.0
    total_cost_per_trade: float = 0.0
    n_trades: int = 0
    turnover_annual: float = 0.0


class TransactionCostModel:
    """Models realistic transaction costs for strategy evaluation.

    Components:
    - Commission: fixed cost per share
    - Spread: bid-ask spread (varies by vol regime)
    - Slippage: proportional to sqrt(trade_size / avg_daily_volume) * spread
    - Market impact: square-root model for larger positions

    Parameters
    ----------
    commission_per_share : float, optional
        Commission per share in dollars. Default is 0.005.
    spread_low_vol : float, optional
        Spread as fraction in low-vol regime. Default is 0.0001 (1 bp).
    spread_normal : float, optional
        Spread as fraction in normal vol regime. Default is 0.0002 (2 bp).
    spread_high_vol : float, optional
        Spread as fraction in high-vol regime. Default is 0.0005 (5 bp).
    vol_threshold_low : float, optional
        Annualized vol below which is considered low-vol. Default is 0.15.
    vol_threshold_high : float, optional
        Annualized vol above which is considered high-vol. Default is 0.30.
    assumed_trade_size : float, optional
        Assumed trade size in shares for slippage. Default is 1000.
    min_net_sharpe : float, optional
        Minimum net Sharpe to consider edge surviving. Default is 0.2.

    Examples
    --------
    >>> cost_model = TransactionCostModel()
    >>> result = cost_model.evaluate(hypothesis, data)
    >>> print(result.survives_costs, result.net_sharpe)
    """

    def __init__(
        self,
        commission_per_share: float = 0.005,
        spread_low_vol: float = 0.0001,
        spread_normal: float = 0.0002,
        spread_high_vol: float = 0.0005,
        vol_threshold_low: float = 0.15,
        vol_threshold_high: float = 0.30,
        assumed_trade_size: float = 1000.0,
        min_net_sharpe: float = 0.2,
    ) -> None:
        self.commission_per_share = commission_per_share
        self.spread_low_vol = spread_low_vol
        self.spread_normal = spread_normal
        self.spread_high_vol = spread_high_vol
        self.vol_threshold_low = vol_threshold_low
        self.vol_threshold_high = vol_threshold_high
        self.assumed_trade_size = assumed_trade_size
        self.min_net_sharpe = min_net_sharpe
        self.tester = StatisticalTester()

    def _compute_spread(self, realized_vol: pd.Series) -> pd.Series:
        """Compute spread based on realized volatility regime.

        Parameters
        ----------
        realized_vol : pd.Series
            Annualized realized volatility.

        Returns
        -------
        pd.Series
            Spread as fraction for each day.
        """
        spread = pd.Series(self.spread_normal, index=realized_vol.index)
        spread[realized_vol < self.vol_threshold_low] = self.spread_low_vol
        spread[realized_vol > self.vol_threshold_high] = self.spread_high_vol
        return spread

    def _compute_slippage(
        self, spread: pd.Series, avg_volume: pd.Series
    ) -> pd.Series:
        """Compute slippage as function of trade size and volume.

        Slippage = sqrt(trade_size / avg_daily_volume) * spread

        Parameters
        ----------
        spread : pd.Series
            Current spread for each day.
        avg_volume : pd.Series
            Average daily volume.

        Returns
        -------
        pd.Series
            Slippage as fraction for each day.
        """
        # Avoid division by zero
        safe_volume = avg_volume.clip(lower=1.0)
        size_ratio = np.sqrt(self.assumed_trade_size / safe_volume)
        return size_ratio * spread

    def _compute_market_impact(
        self, avg_volume: pd.Series, avg_price: pd.Series
    ) -> pd.Series:
        """Compute market impact using square-root model.

        Impact = sigma * sqrt(trade_value / daily_dollar_volume)
        Simplified: impact proportional to sqrt(shares / avg_volume) * volatility_proxy

        Parameters
        ----------
        avg_volume : pd.Series
            Average daily volume.
        avg_price : pd.Series
            Average price.

        Returns
        -------
        pd.Series
            Market impact as fraction for each day.
        """
        # Daily dollar volume
        daily_dollar_vol = avg_volume * avg_price
        safe_ddv = daily_dollar_vol.clip(lower=1.0)

        # Trade value
        trade_value = self.assumed_trade_size * avg_price

        # Square-root impact model
        impact = 0.1 * np.sqrt(trade_value / safe_ddv)
        return impact.clip(lower=0.0, upper=0.01)  # Cap at 1%

    def _compute_commission_cost(self, avg_price: pd.Series) -> pd.Series:
        """Compute commission as fraction of trade value.

        Parameters
        ----------
        avg_price : pd.Series
            Average price per share.

        Returns
        -------
        pd.Series
            Commission as fraction of trade value.
        """
        safe_price = avg_price.clip(lower=0.01)
        return self.commission_per_share / safe_price

    def apply_costs(
        self,
        gross_returns: pd.Series,
        signals: pd.Series,
        avg_price: pd.Series,
        avg_volume: pd.Series,
        realized_vol: pd.Series | None = None,
    ) -> tuple[pd.Series, float]:
        """Apply transaction costs to gross returns.

        Costs are applied on each signal change (trade). A round-trip
        cost includes entry and exit costs.

        Parameters
        ----------
        gross_returns : pd.Series
            Gross signal-weighted returns.
        signals : pd.Series
            Signal values (used to determine when trades occur).
        avg_price : pd.Series
            Average price per share for each day.
        avg_volume : pd.Series
            Average daily volume.
        realized_vol : pd.Series or None, optional
            Realized volatility for spread regime. If None, computed from
            gross returns.

        Returns
        -------
        tuple[pd.Series, float]
            (net_returns, avg_cost_per_trade)
        """
        if len(gross_returns) == 0:
            return pd.Series(dtype=float), 0.0

        # Align all series
        common_idx = gross_returns.index.intersection(signals.index)
        common_idx = common_idx.intersection(avg_price.index)
        common_idx = common_idx.intersection(avg_volume.index)

        if len(common_idx) == 0:
            return pd.Series(dtype=float), 0.0

        gross_returns = gross_returns.loc[common_idx]
        signals = signals.loc[common_idx]
        avg_price = avg_price.loc[common_idx]
        avg_volume = avg_volume.loc[common_idx]

        # Compute realized vol if not provided
        if realized_vol is None:
            daily_vol = gross_returns.rolling(20, min_periods=5).std() * np.sqrt(252)
            realized_vol = daily_vol.fillna(0.20)  # Default 20% vol
        else:
            realized_vol = realized_vol.loc[common_idx]

        # Compute cost components
        spread = self._compute_spread(realized_vol)
        slippage = self._compute_slippage(spread, avg_volume)
        market_impact = self._compute_market_impact(avg_volume, avg_price)
        commission_frac = self._compute_commission_cost(avg_price)

        # Total one-way cost
        one_way_cost = spread / 2 + slippage + market_impact + commission_frac

        # Detect trades (signal changes)
        signal_changes = signals.diff().abs().fillna(0)
        # Normalize: a full reversal (e.g., -1 to +1) counts as 2 trades
        trade_indicator = signal_changes.clip(upper=2.0) / 2.0

        # Apply round-trip costs on each trade
        # Each trade incurs costs on entry; assume exit cost at next trade
        cost_per_day = trade_indicator * one_way_cost * 2  # Round-trip

        net_returns = gross_returns - cost_per_day

        # Average cost per trade
        total_trades = trade_indicator.sum()
        if total_trades > 0:
            avg_cost = float(cost_per_day.sum() / total_trades)
        else:
            avg_cost = 0.0

        return net_returns, avg_cost

    def evaluate(
        self, hypothesis: Hypothesis, data: pd.DataFrame
    ) -> CostAdjustedResult:
        """Evaluate whether a hypothesis survives transaction costs.

        Uses discretized positions (long/flat/short) derived from the signal
        to compute realistic trade events and holding-period returns, rather
        than applying costs to every raw signal change on 1-day returns.

        Parameters
        ----------
        hypothesis : Hypothesis
            The hypothesis to evaluate.
        data : pd.DataFrame
            Full OHLCV + features DataFrame.

        Returns
        -------
        CostAdjustedResult
            Comparison of gross vs net performance metrics.
        """
        # Compute signal and discretize into positions: +1, 0, -1
        try:
            raw_signal = hypothesis.signal_function(data)
        except Exception:
            raw_signal = pd.Series(dtype=float)

        if len(raw_signal) < 5:
            return CostAdjustedResult(
                hypothesis_id=hypothesis.id,
                survives_costs=False,
            )

        # Discretize signal into positions
        positions = pd.Series(0.0, index=raw_signal.index)
        positions[raw_signal > 0] = 1.0
        positions[raw_signal < 0] = -1.0

        # Compute position-weighted returns (holding-period returns)
        daily_ret = data["Close"].pct_change().fillna(0.0)
        gross_returns = (positions.shift(1) * daily_ret).dropna()

        if len(gross_returns) < 5:
            return CostAdjustedResult(
                hypothesis_id=hypothesis.id,
                survives_costs=False,
            )

        # Get price and volume for cost computation
        avg_price = data["Close"].rolling(20, min_periods=1).mean()
        avg_volume = data["Volume"].astype(float).rolling(20, min_periods=1).mean()

        # Compute realized vol
        log_ret = np.log(data["Close"] / data["Close"].shift(1))
        realized_vol = log_ret.rolling(20, min_periods=5).std() * np.sqrt(252)
        realized_vol = realized_vol.fillna(0.20)

        # Apply costs using discretized positions (trade events = position changes)
        net_returns, avg_cost = self.apply_costs(
            gross_returns, positions, avg_price, avg_volume, realized_vol
        )

        # Compute metrics
        gross_sharpe = self.tester.compute_sharpe_ratio(gross_returns)
        gross_expectancy = self.tester.expectancy(gross_returns)
        gross_pf = self.tester.profit_factor(gross_returns)

        if len(net_returns) > 0:
            net_sharpe = self.tester.compute_sharpe_ratio(net_returns)
            net_expectancy = self.tester.expectancy(net_returns)
            net_pf = self.tester.profit_factor(net_returns)
        else:
            net_sharpe = 0.0
            net_expectancy = 0.0
            net_pf = 0.0

        # Determine survival
        survives = net_sharpe > self.min_net_sharpe and net_expectancy > 0

        # Estimate annual turnover from position changes
        position_changes = positions.diff().abs().fillna(0)
        trades_per_day = float(position_changes[position_changes > 0].count()) / max(len(positions), 1)
        turnover_annual = trades_per_day * 252

        return CostAdjustedResult(
            hypothesis_id=hypothesis.id,
            survives_costs=survives,
            gross_sharpe=gross_sharpe,
            net_sharpe=net_sharpe,
            gross_expectancy=gross_expectancy,
            net_expectancy=net_expectancy,
            gross_profit_factor=gross_pf,
            net_profit_factor=net_pf,
            total_cost_per_trade=avg_cost,
            n_trades=int(position_changes[position_changes > 0].sum() / 2),
            turnover_annual=turnover_annual,
        )

    def evaluate_batch(
        self, hypotheses: list[Hypothesis], data: pd.DataFrame
    ) -> list[CostAdjustedResult]:
        """Evaluate multiple hypotheses for cost survival.

        Parameters
        ----------
        hypotheses : list[Hypothesis]
            List of hypotheses to evaluate.
        data : pd.DataFrame
            Full dataset.

        Returns
        -------
        list[CostAdjustedResult]
            Cost analysis results for each hypothesis.
        """
        return [self.evaluate(h, data) for h in hypotheses]
