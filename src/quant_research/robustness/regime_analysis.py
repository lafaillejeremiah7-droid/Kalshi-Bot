"""
Market regime identification and regime-conditional hypothesis testing.

Identifies distinct market regimes (bull trending, bear trending, sideways,
high-vol crisis) from price data and evaluates whether trading hypotheses
are robust across multiple regimes or regime-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from quant_research.hypotheses.catalog import Hypothesis
from quant_research.testing.statistical import StatisticalTester


class MarketRegime(Enum):
    """Market regime classifications."""

    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS = "sideways"
    HIGH_VOL_CRISIS = "high_vol_crisis"


@dataclass
class RegimeMetrics:
    """Performance metrics for a hypothesis within a specific regime.

    Attributes
    ----------
    regime : MarketRegime
        The market regime.
    n_days : int
        Number of days in this regime.
    n_trades : int
        Number of active signal days (trades) in this regime.
    sharpe_ratio : float
        Annualized Sharpe ratio in this regime.
    expectancy : float
        Expected return per trade in this regime.
    hit_rate : float
        Fraction of positive trades in this regime.
    """

    regime: MarketRegime
    n_days: int = 0
    n_trades: int = 0
    sharpe_ratio: float = 0.0
    expectancy: float = 0.0
    hit_rate: float = 0.0


@dataclass
class RegimeResult:
    """Result of regime analysis for one hypothesis.

    Attributes
    ----------
    hypothesis_id : str
        ID of the hypothesis tested.
    regime_robust : bool
        True if hypothesis has positive expectancy in at least 2 regimes.
    regime_metrics : list[RegimeMetrics]
        Per-regime performance metrics.
    dominant_regime : MarketRegime | None
        The regime where the hypothesis performs best (if regime-dependent).
    regimes_with_positive_expectancy : int
        Count of regimes with positive expectancy.
    """

    hypothesis_id: str
    regime_robust: bool
    regime_metrics: list[RegimeMetrics] = field(default_factory=list)
    dominant_regime: MarketRegime | None = None
    regimes_with_positive_expectancy: int = 0


class RegimeAnalyzer:
    """Identifies market regimes and tests hypothesis performance per regime.

    Uses a rule-based classification approach:
    - Bull trending: 60-day annualized return > 10% AND realized vol below
      1.5x long-term average
    - Bear trending: 60-day annualized return < -10% (any vol level)
    - Sideways/choppy: absolute 60-day annualized return < 10% AND realized
      vol below 1.5x long-term average
    - High-vol crisis: realized vol > 1.5x long-term average (regardless
      of direction)

    Optionally uses Hidden Markov Model (hmmlearn) if available; otherwise
    falls back to the rule-based approach.

    Parameters
    ----------
    use_hmm : bool, optional
        Whether to attempt HMM-based regime detection. Default is False.
    return_lookback : int, optional
        Lookback period for return computation (days). Default is 60.
    vol_lookback : int, optional
        Lookback period for volatility computation (days). Default is 60.
    vol_threshold_multiplier : float, optional
        Multiplier of long-term vol to trigger high-vol regime. Default is 1.5.
    trend_threshold : float, optional
        Annualized return threshold for bull/bear classification. Default is 0.10.
    min_regimes_for_robust : int, optional
        Minimum number of regimes with positive expectancy to be considered
        robust. Default is 2.

    Examples
    --------
    >>> analyzer = RegimeAnalyzer()
    >>> regimes = analyzer.identify_regimes(data)
    >>> result = analyzer.analyze(hypothesis, data)
    >>> print(result.regime_robust)
    """

    def __init__(
        self,
        use_hmm: bool = False,
        return_lookback: int = 60,
        vol_lookback: int = 60,
        vol_threshold_multiplier: float = 1.5,
        trend_threshold: float = 0.10,
        min_regimes_for_robust: int = 2,
    ) -> None:
        self.use_hmm = use_hmm
        self.return_lookback = return_lookback
        self.vol_lookback = vol_lookback
        self.vol_threshold_multiplier = vol_threshold_multiplier
        self.trend_threshold = trend_threshold
        self.min_regimes_for_robust = min_regimes_for_robust
        self.tester = StatisticalTester()

    def identify_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Identify market regime for each day in the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame with 'Close' column.

        Returns
        -------
        pd.Series
            Series of MarketRegime values indexed by date.
        """
        if self.use_hmm:
            try:
                return self._identify_regimes_hmm(data)
            except (ImportError, Exception):
                pass  # Fall back to rule-based

        return self._identify_regimes_rules(data)

    def _identify_regimes_rules(self, data: pd.DataFrame) -> pd.Series:
        """Rule-based regime identification.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame with 'Close' column.

        Returns
        -------
        pd.Series
            Series of MarketRegime values.
        """
        close = data["Close"]

        # Compute 60-day log return (annualized)
        ret_60d = np.log(close / close.shift(self.return_lookback))
        # Annualize: multiply by 252/lookback
        annualized_ret = ret_60d * (252 / self.return_lookback)

        # Compute realized volatility (annualized)
        daily_ret = np.log(close / close.shift(1))
        realized_vol = daily_ret.rolling(self.vol_lookback).std() * np.sqrt(252)

        # Long-term average vol (use full available history)
        long_term_vol = daily_ret.expanding(min_periods=self.vol_lookback).std() * np.sqrt(252)

        # Vol threshold
        vol_threshold = long_term_vol * self.vol_threshold_multiplier

        # Classify each day
        regimes = pd.Series(index=data.index, dtype=object)

        for i in range(len(data)):
            if pd.isna(annualized_ret.iloc[i]) or pd.isna(realized_vol.iloc[i]):
                regimes.iloc[i] = MarketRegime.SIDEWAYS  # Default for warmup
                continue

            is_high_vol = realized_vol.iloc[i] > vol_threshold.iloc[i]
            ret = annualized_ret.iloc[i]

            if is_high_vol:
                regimes.iloc[i] = MarketRegime.HIGH_VOL_CRISIS
            elif ret > self.trend_threshold:
                regimes.iloc[i] = MarketRegime.BULL_TRENDING
            elif ret < -self.trend_threshold:
                regimes.iloc[i] = MarketRegime.BEAR_TRENDING
            else:
                regimes.iloc[i] = MarketRegime.SIDEWAYS

        return regimes

    def _identify_regimes_hmm(self, data: pd.DataFrame) -> pd.Series:
        """HMM-based regime identification using hmmlearn.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame.

        Returns
        -------
        pd.Series
            Series of MarketRegime values.

        Raises
        ------
        ImportError
            If hmmlearn is not installed.
        """
        from hmmlearn.hmm import GaussianHMM

        close = data["Close"]
        daily_ret = np.log(close / close.shift(1)).dropna()
        realized_vol = daily_ret.rolling(20).std().dropna()

        # Align series
        aligned = pd.concat([daily_ret, realized_vol], axis=1).dropna()
        aligned.columns = ["return", "vol"]

        # Fit HMM with 4 states
        model = GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
        model.fit(aligned.values)
        hidden_states = model.predict(aligned.values)

        # Map states to regimes based on mean return and vol
        state_means = {}
        for state in range(4):
            mask = hidden_states == state
            state_means[state] = {
                "return": aligned["return"].values[mask].mean(),
                "vol": aligned["vol"].values[mask].mean(),
            }

        # Sort states by characteristics to map to regimes
        regime_mapping = self._map_hmm_states_to_regimes(state_means)

        # Create full series
        regimes = pd.Series(MarketRegime.SIDEWAYS, index=data.index)
        for i, idx in enumerate(aligned.index):
            regimes[idx] = regime_mapping[hidden_states[i]]

        return regimes

    @staticmethod
    def _map_hmm_states_to_regimes(
        state_means: dict,
    ) -> dict[int, MarketRegime]:
        """Map HMM hidden states to named market regimes.

        Parameters
        ----------
        state_means : dict
            Dictionary mapping state index to mean return and vol.

        Returns
        -------
        dict[int, MarketRegime]
            Mapping from state index to MarketRegime.
        """
        # Sort states by volatility
        vol_sorted = sorted(state_means.keys(), key=lambda s: state_means[s]["vol"])
        # Highest vol state is crisis
        crisis_state = vol_sorted[-1]

        # Among remaining, sort by return
        remaining = [s for s in vol_sorted if s != crisis_state]
        ret_sorted = sorted(remaining, key=lambda s: state_means[s]["return"])

        mapping: dict[int, MarketRegime] = {}
        mapping[crisis_state] = MarketRegime.HIGH_VOL_CRISIS

        if len(ret_sorted) >= 3:
            mapping[ret_sorted[0]] = MarketRegime.BEAR_TRENDING
            mapping[ret_sorted[1]] = MarketRegime.SIDEWAYS
            mapping[ret_sorted[2]] = MarketRegime.BULL_TRENDING
        elif len(ret_sorted) == 2:
            mapping[ret_sorted[0]] = MarketRegime.BEAR_TRENDING
            mapping[ret_sorted[1]] = MarketRegime.BULL_TRENDING
        elif len(ret_sorted) == 1:
            mapping[ret_sorted[0]] = MarketRegime.SIDEWAYS

        return mapping

    def analyze(
        self, hypothesis: Hypothesis, data: pd.DataFrame
    ) -> RegimeResult:
        """Analyze hypothesis performance across market regimes.

        Parameters
        ----------
        hypothesis : Hypothesis
            The hypothesis to analyze.
        data : pd.DataFrame
            Full dataset with OHLCV + features.

        Returns
        -------
        RegimeResult
            Per-regime metrics and robustness assessment.
        """
        regimes = self.identify_regimes(data)
        regime_metrics: list[RegimeMetrics] = []
        positive_count = 0
        best_sharpe = -np.inf
        dominant_regime: MarketRegime | None = None

        for regime in MarketRegime:
            mask = regimes == regime
            regime_data = data[mask]

            if len(regime_data) < 10:
                # Not enough data in this regime
                regime_metrics.append(
                    RegimeMetrics(regime=regime, n_days=len(regime_data))
                )
                continue

            # Compute signal returns in this regime
            try:
                returns = self.tester.compute_signal_returns(
                    hypothesis, regime_data
                )
            except Exception:
                returns = pd.Series(dtype=float)

            if len(returns) < 3:
                regime_metrics.append(
                    RegimeMetrics(
                        regime=regime,
                        n_days=len(regime_data),
                        n_trades=len(returns),
                    )
                )
                continue

            sharpe = self.tester.compute_sharpe_ratio(returns)
            expectancy = self.tester.expectancy(returns)
            hit_rate = self.tester.compute_hit_rate(returns)

            metrics = RegimeMetrics(
                regime=regime,
                n_days=len(regime_data),
                n_trades=len(returns),
                sharpe_ratio=sharpe,
                expectancy=expectancy,
                hit_rate=hit_rate,
            )
            regime_metrics.append(metrics)

            if expectancy > 0:
                positive_count += 1

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                dominant_regime = regime

        regime_robust = positive_count >= self.min_regimes_for_robust

        return RegimeResult(
            hypothesis_id=hypothesis.id,
            regime_robust=regime_robust,
            regime_metrics=regime_metrics,
            dominant_regime=dominant_regime,
            regimes_with_positive_expectancy=positive_count,
        )

    def analyze_batch(
        self, hypotheses: list[Hypothesis], data: pd.DataFrame
    ) -> list[RegimeResult]:
        """Analyze multiple hypotheses across market regimes.

        Parameters
        ----------
        hypotheses : list[Hypothesis]
            List of hypotheses to analyze.
        data : pd.DataFrame
            Full dataset.

        Returns
        -------
        list[RegimeResult]
            Regime analysis results for each hypothesis.
        """
        return [self.analyze(h, data) for h in hypotheses]
