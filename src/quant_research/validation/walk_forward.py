"""
Walk-forward validation for hypothesis testing.

Implements expanding-window and rolling-window walk-forward analysis
to assess the out-of-sample stability of trading hypotheses. This is
the primary defense against overfitting: a hypothesis must perform
consistently across multiple non-overlapping test periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_research.hypotheses.catalog import Hypothesis
from quant_research.testing.statistical import StatisticalTester


@dataclass
class FoldResult:
    """Result metrics for a single walk-forward fold.

    Attributes
    ----------
    fold_index : int
        Zero-based fold index.
    train_start : str
        Start date of training window.
    train_end : str
        End date of training window.
    test_start : str
        Start date of test window.
    test_end : str
        End date of test window.
    in_sample_sharpe : float
        Sharpe ratio on training data.
    out_of_sample_sharpe : float
        Sharpe ratio on test data.
    in_sample_expectancy : float
        Expectancy on training data.
    out_of_sample_expectancy : float
        Expectancy on test data.
    hit_rate : float
        Hit rate on test data.
    n_trades : int
        Number of trades (active signals) in test period.
    """

    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0
    in_sample_expectancy: float = 0.0
    out_of_sample_expectancy: float = 0.0
    hit_rate: float = 0.0
    n_trades: int = 0


@dataclass
class WalkForwardResult:
    """Aggregated result of walk-forward validation for one hypothesis.

    Attributes
    ----------
    hypothesis_id : str
        ID of the hypothesis tested.
    passed : bool
        Whether the hypothesis passed walk-forward validation.
    rejection_reasons : list[str]
        Reasons for failure (empty if passed).
    fold_results : list[FoldResult]
        Per-fold detailed results.
    consistency_ratio : float
        Fraction of folds with positive out-of-sample expectancy.
    avg_in_sample_sharpe : float
        Average in-sample Sharpe across folds.
    avg_out_of_sample_sharpe : float
        Average out-of-sample Sharpe across folds.
    performance_degradation : float
        Ratio of OOS performance drop vs IS performance.
    """

    hypothesis_id: str
    passed: bool
    rejection_reasons: list[str] = field(default_factory=list)
    fold_results: list[FoldResult] = field(default_factory=list)
    consistency_ratio: float = 0.0
    avg_in_sample_sharpe: float = 0.0
    avg_out_of_sample_sharpe: float = 0.0
    performance_degradation: float = 0.0


class WalkForwardValidator:
    """Walk-forward validation of trading hypotheses.

    Splits data into sequential train/test folds and evaluates signal
    performance on each test fold. Supports both expanding-window and
    rolling-window approaches.

    Parameters
    ----------
    n_folds : int, optional
        Number of walk-forward folds. Default is 5.
    train_ratio : float, optional
        Fraction of each fold used for training (expanding mode initial).
        Default is 0.8.
    method : str, optional
        Either "expanding" or "rolling". Default is "expanding".
    min_consistency_ratio : float, optional
        Minimum fraction of folds with positive expectancy. Default is 0.6.
    max_degradation : float, optional
        Maximum allowed performance degradation (OOS vs IS). Default is 0.5.

    Examples
    --------
    >>> validator = WalkForwardValidator(n_folds=5, method="expanding")
    >>> result = validator.validate(hypothesis, data)
    >>> print(result.passed, result.consistency_ratio)
    """

    def __init__(
        self,
        n_folds: int = 5,
        train_ratio: float = 0.8,
        method: str = "expanding",
        min_consistency_ratio: float = 0.6,
        max_degradation: float = 0.5,
    ) -> None:
        if method not in ("expanding", "rolling"):
            raise ValueError(f"method must be 'expanding' or 'rolling', got '{method}'")
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.method = method
        self.min_consistency_ratio = min_consistency_ratio
        self.max_degradation = max_degradation
        self.tester = StatisticalTester()

    def _generate_folds(
        self, data: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test fold splits from the data.

        For expanding window: training start is always the beginning of data,
        training end advances forward, test is the next segment.

        For rolling window: training window has fixed size that slides forward.

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset to split into folds.

        Returns
        -------
        list[tuple[pd.DataFrame, pd.DataFrame]]
            List of (train_data, test_data) tuples.
        """
        n = len(data)
        folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

        if self.method == "expanding":
            # Divide data into n_folds + 1 segments
            # First segment is the initial training set, subsequent are test folds
            initial_train_size = int(n * self.train_ratio / (1 + self.n_folds * (1 - self.train_ratio) / self.train_ratio))
            # Simpler approach: divide the test portion into n_folds
            test_portion = n - initial_train_size
            fold_size = max(1, test_portion // self.n_folds)

            for i in range(self.n_folds):
                # Training: from start to initial_train_size + i * fold_size
                train_end_idx = initial_train_size + i * fold_size
                test_start_idx = train_end_idx
                test_end_idx = min(test_start_idx + fold_size, n)

                if test_start_idx >= n or test_end_idx <= test_start_idx:
                    break

                train_data = data.iloc[:train_end_idx]
                test_data = data.iloc[test_start_idx:test_end_idx]
                folds.append((train_data, test_data))

        else:  # rolling
            # Fixed-size training window that slides
            fold_size = n // (self.n_folds + 1)
            train_size = int(fold_size * self.n_folds * self.train_ratio)
            train_size = min(train_size, int(n * self.train_ratio))
            test_size = max(1, (n - train_size) // self.n_folds)

            for i in range(self.n_folds):
                test_start_idx = train_size + i * test_size
                test_end_idx = min(test_start_idx + test_size, n)

                if test_start_idx >= n or test_end_idx <= test_start_idx:
                    break

                # Rolling: train window of fixed size ending at test_start
                train_start_idx = max(0, test_start_idx - train_size)
                train_data = data.iloc[train_start_idx:test_start_idx]
                test_data = data.iloc[test_start_idx:test_end_idx]
                folds.append((train_data, test_data))

        return folds

    def _compute_fold_metrics(
        self,
        hypothesis: Hypothesis,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        fold_index: int,
    ) -> FoldResult:
        """Compute performance metrics for a single fold.

        Parameters
        ----------
        hypothesis : Hypothesis
            The hypothesis to evaluate.
        train_data : pd.DataFrame
            Training portion of the data.
        test_data : pd.DataFrame
            Test portion of the data.
        fold_index : int
            Index of this fold.

        Returns
        -------
        FoldResult
            Metrics for this fold.
        """
        # Compute in-sample metrics
        try:
            is_returns = self.tester.compute_signal_returns(hypothesis, train_data)
        except Exception:
            is_returns = pd.Series(dtype=float)

        is_sharpe = self.tester.compute_sharpe_ratio(is_returns)
        is_expectancy = self.tester.expectancy(is_returns)

        # Compute out-of-sample metrics
        try:
            oos_returns = self.tester.compute_signal_returns(hypothesis, test_data)
        except Exception:
            oos_returns = pd.Series(dtype=float)

        oos_sharpe = self.tester.compute_sharpe_ratio(oos_returns)
        oos_expectancy = self.tester.expectancy(oos_returns)
        oos_hit_rate = self.tester.compute_hit_rate(oos_returns)
        n_trades = len(oos_returns)

        # Get date range strings
        train_start = str(train_data.index[0].date()) if len(train_data) > 0 else ""
        train_end = str(train_data.index[-1].date()) if len(train_data) > 0 else ""
        test_start = str(test_data.index[0].date()) if len(test_data) > 0 else ""
        test_end = str(test_data.index[-1].date()) if len(test_data) > 0 else ""

        return FoldResult(
            fold_index=fold_index,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            in_sample_sharpe=is_sharpe,
            out_of_sample_sharpe=oos_sharpe,
            in_sample_expectancy=is_expectancy,
            out_of_sample_expectancy=oos_expectancy,
            hit_rate=oos_hit_rate,
            n_trades=n_trades,
        )

    def validate(
        self, hypothesis: Hypothesis, data: pd.DataFrame
    ) -> WalkForwardResult:
        """Run walk-forward validation on a single hypothesis.

        Parameters
        ----------
        hypothesis : Hypothesis
            The hypothesis to validate.
        data : pd.DataFrame
            Full dataset with OHLCV + features.

        Returns
        -------
        WalkForwardResult
            Complete validation result with per-fold details.
        """
        folds = self._generate_folds(data)
        fold_results: list[FoldResult] = []

        for i, (train_data, test_data) in enumerate(folds):
            fold_result = self._compute_fold_metrics(
                hypothesis, train_data, test_data, i
            )
            fold_results.append(fold_result)

        # Compute aggregate metrics
        if not fold_results:
            return WalkForwardResult(
                hypothesis_id=hypothesis.id,
                passed=False,
                rejection_reasons=["No valid folds generated"],
                fold_results=[],
                consistency_ratio=0.0,
            )

        # Consistency ratio: fraction of folds with positive OOS expectancy
        positive_folds = sum(
            1 for f in fold_results if f.out_of_sample_expectancy > 0
        )
        consistency_ratio = positive_folds / len(fold_results)

        # Average Sharpe ratios
        avg_is_sharpe = np.mean([f.in_sample_sharpe for f in fold_results])
        avg_oos_sharpe = np.mean([f.out_of_sample_sharpe for f in fold_results])

        # Performance degradation
        avg_is_expectancy = np.mean(
            [f.in_sample_expectancy for f in fold_results]
        )
        avg_oos_expectancy = np.mean(
            [f.out_of_sample_expectancy for f in fold_results]
        )

        if avg_is_expectancy > 0:
            degradation = 1.0 - (avg_oos_expectancy / avg_is_expectancy)
        else:
            # If in-sample is not positive, degradation is not meaningful
            degradation = 0.0

        # Apply rejection criteria
        rejection_reasons: list[str] = []

        if consistency_ratio < self.min_consistency_ratio:
            rejection_reasons.append(
                f"Consistency ratio {consistency_ratio:.2f} < {self.min_consistency_ratio}"
            )

        if degradation > self.max_degradation and avg_is_expectancy > 0:
            rejection_reasons.append(
                f"Performance degradation {degradation:.2f} > {self.max_degradation} "
                f"(IS expectancy: {avg_is_expectancy:.6f}, OOS: {avg_oos_expectancy:.6f})"
            )

        passed = len(rejection_reasons) == 0

        return WalkForwardResult(
            hypothesis_id=hypothesis.id,
            passed=passed,
            rejection_reasons=rejection_reasons,
            fold_results=fold_results,
            consistency_ratio=consistency_ratio,
            avg_in_sample_sharpe=float(avg_is_sharpe),
            avg_out_of_sample_sharpe=float(avg_oos_sharpe),
            performance_degradation=float(degradation),
        )

    def validate_batch(
        self, hypotheses: list[Hypothesis], data: pd.DataFrame
    ) -> list[WalkForwardResult]:
        """Run walk-forward validation on multiple hypotheses.

        Parameters
        ----------
        hypotheses : list[Hypothesis]
            List of hypotheses to validate.
        data : pd.DataFrame
            Full dataset with OHLCV + features.

        Returns
        -------
        list[WalkForwardResult]
            Validation results for each hypothesis.
        """
        return [self.validate(h, data) for h in hypotheses]
