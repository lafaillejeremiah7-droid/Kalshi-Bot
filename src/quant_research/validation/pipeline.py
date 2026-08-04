"""
Validation pipeline orchestrating all validation and robustness stages.

Chains walk-forward validation, out-of-sample testing, regime analysis,
and transaction cost modeling into a single pipeline that takes hypotheses
that passed initial statistical testing and produces a final list of
validated, robust, cost-adjusted survivors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from quant_research.hypotheses.catalog import Hypothesis
from quant_research.robustness.regime_analysis import RegimeAnalyzer, RegimeResult
from quant_research.robustness.transaction_costs import (
    CostAdjustedResult,
    TransactionCostModel,
)
from quant_research.validation.out_of_sample import OOSResult, OutOfSampleValidator
from quant_research.validation.walk_forward import WalkForwardResult, WalkForwardValidator

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class RejectionFunnel:
    """Tracks how many hypotheses are rejected at each pipeline stage.

    Attributes
    ----------
    initial_count : int
        Number of hypotheses entering the pipeline.
    walk_forward_rejected : int
        Number rejected by walk-forward validation.
    walk_forward_survivors : int
        Number surviving walk-forward.
    oos_rejected : int
        Number rejected by out-of-sample testing.
    oos_survivors : int
        Number surviving OOS.
    regime_flagged : int
        Number flagged as regime-dependent (not robust).
    cost_rejected : int
        Number rejected by transaction cost analysis.
    final_survivors : int
        Number of hypotheses surviving all stages.
    """

    initial_count: int = 0
    walk_forward_rejected: int = 0
    walk_forward_survivors: int = 0
    oos_rejected: int = 0
    oos_survivors: int = 0
    regime_flagged: int = 0
    cost_rejected: int = 0
    final_survivors: int = 0


@dataclass
class ValidatedHypothesis:
    """A hypothesis that passed all validation stages with full metrics.

    Attributes
    ----------
    hypothesis : Hypothesis
        The original hypothesis object.
    walk_forward_result : WalkForwardResult
        Walk-forward validation result.
    oos_result : OOSResult
        Out-of-sample validation result.
    regime_result : RegimeResult
        Regime analysis result.
    cost_result : CostAdjustedResult
        Transaction cost analysis result.
    """

    hypothesis: Hypothesis
    walk_forward_result: WalkForwardResult
    oos_result: OOSResult
    regime_result: RegimeResult
    cost_result: CostAdjustedResult


@dataclass
class ValidationReport:
    """Complete validation pipeline report.

    Attributes
    ----------
    validated_hypotheses : list[ValidatedHypothesis]
        Hypotheses that survived all stages.
    rejection_funnel : RejectionFunnel
        Counts at each stage.
    walk_forward_results : list[WalkForwardResult]
        All walk-forward results.
    oos_results : list[OOSResult]
        All OOS results.
    regime_results : list[RegimeResult]
        All regime analysis results.
    cost_results : list[CostAdjustedResult]
        All cost analysis results.
    """

    validated_hypotheses: list[ValidatedHypothesis] = field(default_factory=list)
    rejection_funnel: RejectionFunnel = field(default_factory=RejectionFunnel)
    walk_forward_results: list[WalkForwardResult] = field(default_factory=list)
    oos_results: list[OOSResult] = field(default_factory=list)
    regime_results: list[RegimeResult] = field(default_factory=list)
    cost_results: list[CostAdjustedResult] = field(default_factory=list)


class ValidationPipeline:
    """Orchestrates the full validation flow for tested hypotheses.

    Pipeline stages:
    1. Walk-forward validation: reject hypotheses that are inconsistent OOS
    2. Out-of-sample validation: reject on strict holdout thresholds
    3. Regime analysis: flag regime-dependent hypotheses
    4. Transaction costs: reject if edge disappears after costs

    Parameters
    ----------
    walk_forward_validator : WalkForwardValidator or None, optional
        Custom walk-forward validator. If None, uses defaults.
    oos_validator : OutOfSampleValidator or None, optional
        Custom OOS validator. If None, uses defaults.
    regime_analyzer : RegimeAnalyzer or None, optional
        Custom regime analyzer. If None, uses defaults.
    cost_model : TransactionCostModel or None, optional
        Custom transaction cost model. If None, uses defaults.

    Examples
    --------
    >>> pipeline = ValidationPipeline()
    >>> report = pipeline.run(hypotheses, data)
    >>> print(f"{report.rejection_funnel.final_survivors} hypotheses survived")
    """

    def __init__(
        self,
        walk_forward_validator: WalkForwardValidator | None = None,
        oos_validator: OutOfSampleValidator | None = None,
        regime_analyzer: RegimeAnalyzer | None = None,
        cost_model: TransactionCostModel | None = None,
    ) -> None:
        self.wf_validator = walk_forward_validator or WalkForwardValidator()
        self.oos_validator = oos_validator or OutOfSampleValidator()
        self.regime_analyzer = regime_analyzer or RegimeAnalyzer()
        self.cost_model = cost_model or TransactionCostModel()

    def run(
        self, hypotheses: list[Hypothesis], data: pd.DataFrame
    ) -> ValidationReport:
        """Run the full validation pipeline.

        Parameters
        ----------
        hypotheses : list[Hypothesis]
            Hypotheses that passed initial statistical testing.
        data : pd.DataFrame
            Full OHLCV + features DataFrame.

        Returns
        -------
        ValidationReport
            Complete report with surviving hypotheses and rejection funnel.
        """
        funnel = RejectionFunnel(initial_count=len(hypotheses))
        all_wf_results: list[WalkForwardResult] = []
        all_oos_results: list[OOSResult] = []
        all_regime_results: list[RegimeResult] = []
        all_cost_results: list[CostAdjustedResult] = []

        logger.info(
            "Starting validation pipeline with %d hypotheses", len(hypotheses)
        )

        # Stage 1: Walk-forward validation
        logger.info("Stage 1: Walk-forward validation")
        wf_survivors: list[Hypothesis] = []
        for hyp in hypotheses:
            result = self.wf_validator.validate(hyp, data)
            all_wf_results.append(result)
            if result.passed:
                wf_survivors.append(hyp)

        funnel.walk_forward_rejected = len(hypotheses) - len(wf_survivors)
        funnel.walk_forward_survivors = len(wf_survivors)
        logger.info(
            "Walk-forward: %d/%d survived",
            len(wf_survivors),
            len(hypotheses),
        )

        # Stage 2: Out-of-sample validation
        logger.info("Stage 2: Out-of-sample validation")
        oos_survivors: list[Hypothesis] = []
        for hyp in wf_survivors:
            result = self.oos_validator.validate(hyp, data)
            all_oos_results.append(result)
            if result.passed:
                oos_survivors.append(hyp)

        funnel.oos_rejected = len(wf_survivors) - len(oos_survivors)
        funnel.oos_survivors = len(oos_survivors)
        logger.info(
            "OOS validation: %d/%d survived",
            len(oos_survivors),
            len(wf_survivors),
        )

        # Stage 3: Regime analysis (flags but does not reject)
        logger.info("Stage 3: Regime analysis")
        regime_flagged_count = 0
        for hyp in oos_survivors:
            result = self.regime_analyzer.analyze(hyp, data)
            all_regime_results.append(result)
            if not result.regime_robust:
                regime_flagged_count += 1

        funnel.regime_flagged = regime_flagged_count
        logger.info(
            "Regime analysis: %d/%d flagged as regime-dependent",
            regime_flagged_count,
            len(oos_survivors),
        )

        # Stage 4: Transaction cost analysis
        logger.info("Stage 4: Transaction cost analysis")
        cost_survivors: list[Hypothesis] = []
        for hyp in oos_survivors:
            result = self.cost_model.evaluate(hyp, data)
            all_cost_results.append(result)
            if result.survives_costs:
                cost_survivors.append(hyp)

        funnel.cost_rejected = len(oos_survivors) - len(cost_survivors)
        funnel.final_survivors = len(cost_survivors)
        logger.info(
            "Cost analysis: %d/%d survived",
            len(cost_survivors),
            len(oos_survivors),
        )

        # Build validated hypothesis objects
        validated: list[ValidatedHypothesis] = []
        for hyp in cost_survivors:
            # Find the results for this hypothesis
            wf_res = next(
                (r for r in all_wf_results if r.hypothesis_id == hyp.id), None
            )
            oos_res = next(
                (r for r in all_oos_results if r.hypothesis_id == hyp.id), None
            )
            regime_res = next(
                (r for r in all_regime_results if r.hypothesis_id == hyp.id), None
            )
            cost_res = next(
                (r for r in all_cost_results if r.hypothesis_id == hyp.id), None
            )

            if wf_res and oos_res and regime_res and cost_res:
                validated.append(
                    ValidatedHypothesis(
                        hypothesis=hyp,
                        walk_forward_result=wf_res,
                        oos_result=oos_res,
                        regime_result=regime_res,
                        cost_result=cost_res,
                    )
                )

        logger.info(
            "Pipeline complete: %d hypotheses validated out of %d initial",
            len(validated),
            len(hypotheses),
        )

        return ValidationReport(
            validated_hypotheses=validated,
            rejection_funnel=funnel,
            walk_forward_results=all_wf_results,
            oos_results=all_oos_results,
            regime_results=all_regime_results,
            cost_results=all_cost_results,
        )
