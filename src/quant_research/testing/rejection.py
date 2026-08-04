"""
Hypothesis rejection framework with multiple-testing correction.

Implements Benjamini-Hochberg FDR correction and multi-criteria rejection
to filter hypotheses that do not show statistically significant or
economically meaningful results.

With 100+ hypotheses tested simultaneously, naive p-values are meaningless
due to multiple comparisons. This module addresses that critical issue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from quant_research.hypotheses.catalog import Hypothesis, HypothesisCategory
from quant_research.testing.statistical import StatisticalTester


@dataclass
class RejectionResult:
    """Result of hypothesis rejection analysis.

    Attributes
    ----------
    hypothesis_id : str
        ID of the hypothesis.
    rejected : bool
        Whether the hypothesis was rejected.
    reasons : list[str]
        List of rejection reasons (empty if not rejected).
    p_value_raw : float
        Raw (unadjusted) p-value.
    p_value_adjusted : float
        Benjamini-Hochberg adjusted p-value.
    sharpe_ratio : float
        Annualized Sharpe ratio.
    n_observations : int
        Number of signal observations.
    effect_size : float
        Cohen's d effect size.
    confidence_flag : str
        Either "normal", "limited_data_confidence", or "rejected".
    """

    hypothesis_id: str
    rejected: bool
    reasons: list[str] = field(default_factory=list)
    p_value_raw: float = 1.0
    p_value_adjusted: float = 1.0
    sharpe_ratio: float = 0.0
    n_observations: int = 0
    effect_size: float = 0.0
    confidence_flag: str = "normal"


class HypothesisRejector:
    """Applies multiple-testing correction and multi-criteria rejection.

    Rejection criteria:
    1. Benjamini-Hochberg adjusted p-value > 0.05
    2. Sharpe ratio < 0.3 (pre-costs)
    3. Fewer than 30 signal observations
    4. For ORDER_FLOW_PROXY: flags as "limited_data_confidence" instead
       of outright rejection, with note about data limitations

    Parameters
    ----------
    alpha : float, optional
        Significance level for BH correction. Default is 0.05.
    min_sharpe : float, optional
        Minimum Sharpe ratio threshold. Default is 0.3.
    min_observations : int, optional
        Minimum number of signal observations. Default is 30.

    Examples
    --------
    >>> rejector = HypothesisRejector()
    >>> results = rejector.evaluate_all(hypotheses, data)
    >>> surviving = [r for r in results if not r.rejected]
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_sharpe: float = 0.3,
        min_observations: int = 30,
    ) -> None:
        self.alpha = alpha
        self.min_sharpe = min_sharpe
        self.min_observations = min_observations
        self.tester = StatisticalTester()

    def benjamini_hochberg(
        self, p_values: list[float]
    ) -> list[float]:
        """Apply Benjamini-Hochberg FDR correction.

        Parameters
        ----------
        p_values : list[float]
            Raw p-values from multiple tests.

        Returns
        -------
        list[float]
            Adjusted p-values controlling FDR at specified alpha.
        """
        n = len(p_values)
        if n == 0:
            return []
        # Sort p-values and track original indices
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted = [0.0] * n
        # BH procedure: p_adj[i] = min(p[i] * n / rank, 1.0)
        prev_adj = 1.0
        for rank_idx in range(n - 1, -1, -1):
            orig_idx, p_val = indexed[rank_idx]
            rank = rank_idx + 1  # 1-indexed rank
            adj = min(p_val * n / rank, 1.0)
            adj = min(adj, prev_adj)  # Enforce monotonicity
            adjusted[orig_idx] = adj
            prev_adj = adj
        return adjusted

    def compute_cohens_d(self, returns: pd.Series) -> float:
        """Compute Cohen's d effect size.

        Parameters
        ----------
        returns : pd.Series
            Return series.

        Returns
        -------
        float
            Cohen's d (mean / std).
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std())

    def evaluate_all(
        self,
        hypotheses: list[Hypothesis],
        data: pd.DataFrame,
        forward_period: int = 1,
    ) -> list[RejectionResult]:
        """Evaluate all hypotheses and apply rejection criteria.

        Parameters
        ----------
        hypotheses : list[Hypothesis]
            List of hypotheses to evaluate.
        data : pd.DataFrame
            OHLCV + features DataFrame.
        forward_period : int, optional
            Forward return period in days. Default is 1.

        Returns
        -------
        list[RejectionResult]
            List of rejection results for each hypothesis.
        """
        # First pass: compute raw statistics for each hypothesis
        raw_results: list[dict] = []
        for hyp in hypotheses:
            try:
                returns = self.tester.compute_signal_returns(
                    hyp, data, forward_period
                )
            except Exception:
                returns = pd.Series(dtype=float)
            t_stat, p_value = self.tester.t_test_mean_return(returns)
            sharpe = self.tester.compute_sharpe_ratio(returns)
            effect_size = self.compute_cohens_d(returns)
            raw_results.append({
                "hypothesis": hyp,
                "returns": returns,
                "p_value": p_value,
                "sharpe": sharpe,
                "n_obs": len(returns),
                "effect_size": effect_size,
            })

        # Apply Benjamini-Hochberg correction
        raw_p_values = [r["p_value"] for r in raw_results]
        adjusted_p_values = self.benjamini_hochberg(raw_p_values)

        # Second pass: apply rejection criteria
        results: list[RejectionResult] = []
        for i, raw in enumerate(raw_results):
            hyp = raw["hypothesis"]
            reasons: list[str] = []
            rejected = False
            confidence_flag = "normal"

            # Criterion 1: Minimum observations
            if raw["n_obs"] < self.min_observations:
                reasons.append(
                    f"Insufficient observations: {raw['n_obs']} < {self.min_observations}"
                )
                rejected = True

            # Criterion 2: BH-adjusted p-value
            if adjusted_p_values[i] > self.alpha:
                reasons.append(
                    f"BH-adjusted p-value {adjusted_p_values[i]:.4f} > {self.alpha}"
                )
                rejected = True

            # Criterion 3: Minimum Sharpe ratio
            if raw["sharpe"] < self.min_sharpe:
                reasons.append(
                    f"Sharpe ratio {raw['sharpe']:.3f} < {self.min_sharpe}"
                )
                rejected = True

            # Special handling for ORDER_FLOW_PROXY category
            # These hypotheses are still rejected if they fail criteria,
            # but flagged as limited_data_confidence for reporting purposes.
            if hyp.category == HypothesisCategory.ORDER_FLOW_PROXY:
                if rejected:
                    confidence_flag = "limited_data_confidence"
                    reasons.append(
                        "ORDER_FLOW_PROXY: flagged as limited_data_confidence. "
                        "These signals are inferred from OHLCV and cannot represent "
                        "true order book data."
                    )

            result = RejectionResult(
                hypothesis_id=hyp.id,
                rejected=rejected,
                reasons=reasons,
                p_value_raw=raw["p_value"],
                p_value_adjusted=adjusted_p_values[i],
                sharpe_ratio=raw["sharpe"],
                n_observations=raw["n_obs"],
                effect_size=raw["effect_size"],
                confidence_flag=confidence_flag,
            )
            results.append(result)

        return results

    def get_surviving(
        self, results: list[RejectionResult]
    ) -> list[RejectionResult]:
        """Get hypotheses that survived rejection.

        Parameters
        ----------
        results : list[RejectionResult]
            Full list of rejection results.

        Returns
        -------
        list[RejectionResult]
            Only non-rejected results.
        """
        return [r for r in results if not r.rejected]

    def get_rejected(
        self, results: list[RejectionResult]
    ) -> list[RejectionResult]:
        """Get hypotheses that were rejected with reasons.

        Parameters
        ----------
        results : list[RejectionResult]
            Full list of rejection results.

        Returns
        -------
        list[RejectionResult]
            Only rejected results.
        """
        return [r for r in results if r.rejected]

