"""
Statistical testing framework for hypothesis evaluation.

Provides methods for computing signal returns, t-tests, bootstrap confidence
intervals, permutation tests, and various performance metrics.

All tests assume daily return series derived from OHLCV data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quant_research.hypotheses.catalog import Hypothesis


class StatisticalTester:
    """Statistical testing of market hypotheses.

    Evaluates hypotheses by computing signal-conditional returns and
    running various statistical significance tests.

    Examples
    --------
    >>> tester = StatisticalTester()
    >>> returns = tester.compute_signal_returns(hypothesis, data)
    >>> t_stat, p_val = tester.t_test_mean_return(returns)
    """

    def compute_signal_returns(
        self,
        hypothesis: Hypothesis,
        data: pd.DataFrame,
        forward_period: int = 1,
    ) -> pd.Series:
        """Compute forward returns conditional on the signal.

        Parameters
        ----------
        hypothesis : Hypothesis
            The hypothesis to test (contains signal_function).
        data : pd.DataFrame
            OHLCV + features DataFrame.
        forward_period : int, optional
            Number of days forward for return computation. Default is 1.

        Returns
        -------
        pd.Series
            Forward returns on days when signal is non-zero,
            multiplied by signal direction.
        """
        signal = hypothesis.signal_function(data)
        forward_ret = np.log(
            data["Close"].shift(-forward_period) / data["Close"]
        )
        # Returns when signal is active, signed by signal direction
        active = signal != 0
        signal_returns = (forward_ret * signal)[active]
        return signal_returns.dropna()

    def t_test_mean_return(
        self, returns: pd.Series
    ) -> tuple[float, float]:
        """One-sample t-test for mean return being different from zero.

        Parameters
        ----------
        returns : pd.Series
            Series of returns to test.

        Returns
        -------
        tuple[float, float]
            (t_statistic, p_value) for two-sided test.
        """
        if len(returns) < 2:
            return 0.0, 1.0
        t_stat, p_value = stats.ttest_1samp(returns.dropna(), 0)
        return float(t_stat), float(p_value)

    def bootstrap_test(
        self,
        returns: pd.Series,
        n_iterations: int = 10000,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for mean return.

        Parameters
        ----------
        returns : pd.Series
            Series of returns.
        n_iterations : int, optional
            Number of bootstrap iterations. Default is 10000.

        Returns
        -------
        tuple[float, float]
            (lower_ci, upper_ci) at 95% confidence level.
        """
        returns_arr = returns.dropna().values
        if len(returns_arr) < 2:
            return 0.0, 0.0
        rng = np.random.default_rng(42)
        boot_means = np.array([
            rng.choice(returns_arr, size=len(returns_arr), replace=True).mean()
            for _ in range(n_iterations)
        ])
        lower = float(np.percentile(boot_means, 2.5))
        upper = float(np.percentile(boot_means, 97.5))
        return lower, upper

    def permutation_test(
        self,
        signal: pd.Series,
        returns: pd.Series,
        n_permutations: int = 5000,
    ) -> float:
        """Permutation test for signal-return relationship.

        Shuffles the signal and recomputes mean signal-weighted return
        to build null distribution.

        Parameters
        ----------
        signal : pd.Series
            Signal values (aligned with returns).
        returns : pd.Series
            Return values (aligned with signal).
        n_permutations : int, optional
            Number of permutations. Default is 5000.

        Returns
        -------
        float
            P-value (fraction of permuted stats >= observed).
        """
        # Align and drop NaN
        combined = pd.concat([signal, returns], axis=1).dropna()
        if len(combined) < 5:
            return 1.0
        sig = combined.iloc[:, 0].values
        ret = combined.iloc[:, 1].values
        observed = np.mean(sig * ret)
        rng = np.random.default_rng(42)
        count = 0
        for _ in range(n_permutations):
            perm_sig = rng.permutation(sig)
            perm_stat = np.mean(perm_sig * ret)
            if abs(perm_stat) >= abs(observed):
                count += 1
        return count / n_permutations

    def compute_sharpe_ratio(
        self, returns: pd.Series, risk_free: float = 0.0
    ) -> float:
        """Compute annualized Sharpe ratio.

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.
        risk_free : float, optional
            Daily risk-free rate. Default is 0.0.

        Returns
        -------
        float
            Annualized Sharpe ratio.
        """
        excess = returns - risk_free
        if len(excess) < 2 or excess.std() == 0:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(252))

    def compute_information_ratio(
        self,
        signal_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Compute information ratio (active return / tracking error).

        Parameters
        ----------
        signal_returns : pd.Series
            Strategy returns.
        benchmark_returns : pd.Series
            Benchmark returns.

        Returns
        -------
        float
            Annualized information ratio.
        """
        active = signal_returns - benchmark_returns
        if len(active) < 2 or active.std() == 0:
            return 0.0
        return float(active.mean() / active.std() * np.sqrt(252))

    def compute_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Compute maximum drawdown from cumulative returns.

        Parameters
        ----------
        cumulative_returns : pd.Series
            Cumulative return series (e.g., cumsum of log returns).

        Returns
        -------
        float
            Maximum drawdown as a negative fraction.
        """
        if len(cumulative_returns) < 2:
            return 0.0
        wealth = np.exp(cumulative_returns)
        peak = wealth.cummax()
        drawdown = (wealth - peak) / peak
        return float(drawdown.min())

    def profit_factor(self, returns: pd.Series) -> float:
        """Compute profit factor (gross profit / gross loss).

        Parameters
        ----------
        returns : pd.Series
            Return series.

        Returns
        -------
        float
            Profit factor. Returns inf if no losses, 0 if no profits.
        """
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return float("inf") if profits > 0 else 0.0
        return float(profits / losses)

    def expectancy(self, returns: pd.Series) -> float:
        """Compute expectancy (expected value per trade).

        Expectancy = avg_win * win_rate - avg_loss * loss_rate

        Parameters
        ----------
        returns : pd.Series
            Return series.

        Returns
        -------
        float
            Expectancy per trade.
        """
        if len(returns) == 0:
            return 0.0
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = len(wins) / len(returns)
        loss_rate = len(losses) / len(returns)
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        return float(avg_win * win_rate - avg_loss * loss_rate)

    def compute_hit_rate(self, returns: pd.Series) -> float:
        """Compute hit rate (fraction of positive returns).

        Parameters
        ----------
        returns : pd.Series
            Return series.

        Returns
        -------
        float
            Hit rate (0 to 1).
        """
        if len(returns) == 0:
            return 0.0
        return float((returns > 0).sum() / len(returns))

