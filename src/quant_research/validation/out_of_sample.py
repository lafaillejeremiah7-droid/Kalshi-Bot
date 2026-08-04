"""
Out-of-sample validation with strict holdout testing.

Reserves a final portion of data that is never used during hypothesis
development, walk-forward validation, or any prior step. This provides
the ultimate test of whether an edge is real or an artifact of overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from quant_research.hypotheses.catalog import Hypothesis
from quant_research.testing.statistical import StatisticalTester


@dataclass
class OOSResult:
    """Result of out-of-sample validation for one hypothesis.

    Attributes
    ----------
    hypothesis_id : str
        ID of the hypothesis tested.
    passed : bool
        Whether the hypothesis passed OOS validation.
    rejection_reasons : list[str]
        Reasons for failure (empty if passed).
    sharpe_ratio : float
        Annualized Sharpe ratio on holdout.
    hit_rate : float
        Hit rate on holdout data.
    expectancy : float
        Expectancy on holdout data.
    max_drawdown : float
        Maximum drawdown on holdout data.
    profit_factor : float
        Profit factor on holdout data.
    p_value : float
        Statistical significance on holdout.
    n_trades : int
        Number of trades in holdout period.
    edge_decay : float
        Ratio of second-half performance to first-half performance.
    first_half_sharpe : float
        Sharpe in first half of holdout.
    second_half_sharpe : float
        Sharpe in second half of holdout.
    """

    hypothesis_id: str
    passed: bool
    rejection_reasons: list[str] = field(default_factory=list)
    sharpe_ratio: float = 0.0
    hit_rate: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    p_value: float = 1.0
    n_trades: int = 0
    edge_decay: float = 0.0
    first_half_sharpe: float = 0.0
    second_half_sharpe: float = 0.0


class OutOfSampleValidator:
    """Validates hypotheses on a pure holdout dataset.

    Applies stricter thresholds than in-sample testing to guard against
    overfitting. Also performs edge decay analysis to detect signals
    that are weakening over time.

    Parameters
    ----------
    holdout_fraction : float, optional
        Fraction of data reserved as holdout (from the end). Default is 0.2.
    min_p_value : float, optional
        Maximum allowed p-value on holdout. Default is 0.01.
    min_sharpe : float, optional
        Minimum Sharpe ratio on holdout. Default is 0.4.
    min_expectancy : float, optional
        Minimum expectancy on holdout. Default is 0.0 (must be positive).
    max_edge_decay : float, optional
        Maximum allowed edge decay (second half vs first half ratio below
        this triggers a warning but not rejection). Default is 0.3.

    Examples
    --------
    >>> validator = OutOfSampleValidator(holdout_fraction=0.2)
    >>> result = validator.validate(hypothesis, full_data)
    >>> print(result.passed, result.sharpe_ratio)
    """

    def __init__(
        self,
        holdout_fraction: float = 0.2,
        min_p_value: float = 0.01,
        min_sharpe: float = 0.4,
        min_expectancy: float = 0.0,
        max_edge_decay: float = 0.3,
    ) -> None:
        self.holdout_fraction = holdout_fraction
        self.min_p_value = min_p_value
        self.min_sharpe = min_sharpe
        self.min_expectancy = min_expectancy
        self.max_edge_decay = max_edge_decay
        self.tester = StatisticalTester()

    def split_holdout(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into development and holdout portions.

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (development_data, holdout_data)
        """
        n = len(data)
        holdout_start = int(n * (1 - self.holdout_fraction))
        return data.iloc[:holdout_start], data.iloc[holdout_start:]

    def _compute_edge_decay(
        self, returns: pd.Series
    ) -> tuple[float, float, float]:
        """Analyze edge decay by comparing first half vs second half.

        Parameters
        ----------
        returns : pd.Series
            Signal returns on holdout data.

        Returns
        -------
        tuple[float, float, float]
            (edge_decay_ratio, first_half_sharpe, second_half_sharpe)
            edge_decay_ratio: (first - second) / first if first > 0, else 0
        """
        if len(returns) < 4:
            return 0.0, 0.0, 0.0

        mid = len(returns) // 2
        first_half = returns.iloc[:mid]
        second_half = returns.iloc[mid:]

        first_sharpe = self.tester.compute_sharpe_ratio(first_half)
        second_sharpe = self.tester.compute_sharpe_ratio(second_half)

        if first_sharpe > 0:
            decay = (first_sharpe - second_sharpe) / first_sharpe
        else:
            decay = 0.0

        return decay, first_sharpe, second_sharpe

    def validate(
        self, hypothesis: Hypothesis, data: pd.DataFrame
    ) -> OOSResult:
        """Validate a hypothesis on holdout data.

        Parameters
        ----------
        hypothesis : Hypothesis
            The hypothesis to validate.
        data : pd.DataFrame
            Full dataset (holdout will be extracted from the end).

        Returns
        -------
        OOSResult
            Validation result with metrics and pass/fail.
        """
        _, holdout_data = self.split_holdout(data)

        # Compute signal returns on holdout
        try:
            returns = self.tester.compute_signal_returns(hypothesis, holdout_data)
        except Exception:
            returns = pd.Series(dtype=float)

        if len(returns) < 5:
            return OOSResult(
                hypothesis_id=hypothesis.id,
                passed=False,
                rejection_reasons=["Insufficient trades in holdout period"],
                n_trades=len(returns),
            )

        # Compute metrics
        sharpe = self.tester.compute_sharpe_ratio(returns)
        hit_rate = self.tester.compute_hit_rate(returns)
        expectancy = self.tester.expectancy(returns)
        max_dd = self.tester.compute_max_drawdown(returns.cumsum())
        pf = self.tester.profit_factor(returns)

        # Statistical significance
        _, p_value = self.tester.t_test_mean_return(returns)

        # Edge decay analysis
        edge_decay, first_half_sharpe, second_half_sharpe = (
            self._compute_edge_decay(returns)
        )

        n_trades = len(returns)

        # Apply stricter thresholds
        rejection_reasons: list[str] = []

        if p_value > self.min_p_value:
            rejection_reasons.append(
                f"OOS p-value {p_value:.4f} > {self.min_p_value}"
            )

        if sharpe < self.min_sharpe:
            rejection_reasons.append(
                f"OOS Sharpe {sharpe:.3f} < {self.min_sharpe}"
            )

        if expectancy <= self.min_expectancy:
            rejection_reasons.append(
                f"OOS expectancy {expectancy:.6f} <= {self.min_expectancy}"
            )

        passed = len(rejection_reasons) == 0

        return OOSResult(
            hypothesis_id=hypothesis.id,
            passed=passed,
            rejection_reasons=rejection_reasons,
            sharpe_ratio=sharpe,
            hit_rate=hit_rate,
            expectancy=expectancy,
            max_drawdown=max_dd,
            profit_factor=pf,
            p_value=p_value,
            n_trades=n_trades,
            edge_decay=edge_decay,
            first_half_sharpe=first_half_sharpe,
            second_half_sharpe=second_half_sharpe,
        )

    def validate_batch(
        self, hypotheses: list[Hypothesis], data: pd.DataFrame
    ) -> list[OOSResult]:
        """Validate multiple hypotheses on holdout data.

        Parameters
        ----------
        hypotheses : list[Hypothesis]
            List of hypotheses to validate.
        data : pd.DataFrame
            Full dataset.

        Returns
        -------
        list[OOSResult]
            Validation results for each hypothesis.
        """
        return [self.validate(h, data) for h in hypotheses]
