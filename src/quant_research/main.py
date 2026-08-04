"""
Main pipeline orchestrator for the quantitative research framework.

Runs the complete research process end-to-end: data fetching, feature
engineering, hypothesis generation, statistical testing, validation,
strategy design, and reporting.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quant_research.data.fetcher import DataFetcher
from quant_research.data.features import FeatureEngine
from quant_research.hypotheses.catalog import Hypothesis
from quant_research.hypotheses.generator import HypothesisGenerator
from quant_research.reporting.results import ResearchReporter
from quant_research.robustness.regime_analysis import RegimeAnalyzer
from quant_research.robustness.transaction_costs import TransactionCostModel
from quant_research.strategy.entries_exits import EntryExitDesigner
from quant_research.strategy.position_sizing import PositionSizer
from quant_research.strategy.risk_controls import PortfolioState, RiskController
from quant_research.testing.rejection import HypothesisRejector
from quant_research.testing.statistical import StatisticalTester
from quant_research.validation.out_of_sample import OutOfSampleValidator
from quant_research.validation.pipeline import ValidationPipeline
from quant_research.validation.walk_forward import WalkForwardValidator

logger = logging.getLogger(__name__)

DATA_LIMITATIONS_DISCLAIMER = """
================================================================================
                        DATA LIMITATIONS DISCLAIMER
================================================================================

This pipeline operates on OHLCV (Open, High, Low, Close, Volume) data ONLY.

It does NOT have access to:
  - Level II / order book data
  - Bid-ask spread information
  - Trade-level (tick) data or Time & Sales
  - True order flow or market microstructure data

Order flow proxy hypotheses are INFERRED from price-volume relationships
and CANNOT capture true order flow dynamics.

All results should be interpreted with these constraints in mind.
================================================================================
"""


@dataclass
class PipelineResult:
    """Result of the full pipeline execution.

    Attributes
    ----------
    all_hypotheses : list[Hypothesis]
        All generated hypotheses.
    statistical_survivors : list[Hypothesis]
        Hypotheses surviving statistical testing.
    validated_survivors : list[Hypothesis]
        Hypotheses surviving full validation pipeline.
    strategies : list[dict]
        Strategy details for each validated survivor.
    report_path : str
        Path to the generated report file.
    """

    all_hypotheses: list[Hypothesis] = field(default_factory=list)
    statistical_survivors: list[Hypothesis] = field(default_factory=list)
    validated_survivors: list[Hypothesis] = field(default_factory=list)
    strategies: list[dict] = field(default_factory=list)
    report_path: str = ""


def run_pipeline(config: dict | None = None) -> PipelineResult:
    """Run the complete quantitative research pipeline.

    Pipeline steps:
    1. Print data limitations disclaimer
    2. Fetch data via DataFetcher
    3. Compute features via FeatureEngine
    4. Generate all hypotheses
    5. Run statistical tests on training data
    6. Apply rejection criteria with FDR correction
    7. Run validation pipeline (walk-forward + OOS + regime + costs)
    8. Design entry/exit rules for survivors
    9. Apply position sizing (half-Kelly)
    10. Apply risk controls
    11. Generate and save report

    Parameters
    ----------
    config : dict or None, optional
        Configuration overrides. Supported keys:
        - ticker: str (default "QQQ")
        - years: int (default 10)
        - output_dir: str (default "results")
        - data: pd.DataFrame (pre-supplied data, bypasses fetch)

    Returns
    -------
    PipelineResult
        Complete pipeline results.
    """
    config = config or {}
    ticker = config.get("ticker", "QQQ")
    years = config.get("years", 10)
    output_dir = config.get("output_dir", "results")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    result = PipelineResult()

    # Step 1: Print DATA LIMITATIONS DISCLAIMER
    print(DATA_LIMITATIONS_DISCLAIMER)

    # Step 2: Fetch data
    logger.info("Step 2: Fetching data for %s (%d years)", ticker, years)
    try:
        if "data" in config:
            data = config["data"]
            logger.info("Using pre-supplied data: %d rows", len(data))
        else:
            fetcher = DataFetcher(tickers=[ticker], years=years)
            data = fetcher.fetch(ticker)
            logger.info("Fetched %d rows for %s", len(data), ticker)
    except Exception as e:
        logger.error("Data fetch failed: %s", e)
        return result

    # Step 3: Compute features
    logger.info("Step 3: Computing features")
    try:
        engine = FeatureEngine()
        features = engine.compute_all(data)
        # Merge features with OHLCV data
        full_data = data.join(features)
        logger.info("Computed %d features", len(features.columns))
    except Exception as e:
        logger.error("Feature computation failed: %s", e)
        return result

    # Step 4: Split into train/validation/holdout (60%/20%/20%)
    logger.info("Step 4: Splitting data (60/20/20)")
    n = len(full_data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    train_data = full_data.iloc[:train_end]
    # validation_data used implicitly in walk-forward
    # holdout_data reserved for OOS
    logger.info(
        "Train: %d rows, Validation: %d rows, Holdout: %d rows",
        train_end, val_end - train_end, n - val_end,
    )

    # Step 5: Generate hypotheses
    logger.info("Step 5: Generating hypotheses")
    try:
        generator = HypothesisGenerator()
        all_hypotheses = generator.generate_all()
        result.all_hypotheses = all_hypotheses
        logger.info("Generated %d hypotheses", len(all_hypotheses))
    except Exception as e:
        logger.error("Hypothesis generation failed: %s", e)
        return result

    # Step 6: Run initial statistical tests on training data
    logger.info("Step 6: Running statistical tests on training data")
    try:
        rejector = HypothesisRejector()
        rejection_results = rejector.evaluate_all(all_hypotheses, train_data)
        surviving_results = rejector.get_surviving(rejection_results)
        surviving_ids = {r.hypothesis_id for r in surviving_results}
        statistical_survivors = [
            h for h in all_hypotheses if h.id in surviving_ids
        ]
        result.statistical_survivors = statistical_survivors
        logger.info(
            "Statistical testing: %d/%d survived (FDR-corrected)",
            len(statistical_survivors),
            len(all_hypotheses),
        )
    except Exception as e:
        logger.error("Statistical testing failed: %s", e)
        return result

    # Step 7: Run validation pipeline on survivors
    # Use only train+validation data (first 80%) for walk-forward to prevent
    # overlap with the OOS holdout (last 20-25%).
    logger.info("Step 7: Running validation pipeline")
    validation_report = None
    try:
        if not statistical_survivors:
            logger.warning("No hypotheses survived statistical testing")
        else:
            oos_holdout_fraction = 0.25
            wf_data = full_data.iloc[:int(len(full_data) * (1 - oos_holdout_fraction))]
            pipeline = ValidationPipeline(
                walk_forward_validator=WalkForwardValidator(n_folds=3),
                oos_validator=OutOfSampleValidator(holdout_fraction=oos_holdout_fraction),
                regime_analyzer=RegimeAnalyzer(),
                cost_model=TransactionCostModel(),
            )
            validation_report = pipeline.run(
                statistical_survivors, full_data, wf_data=wf_data
            )
            validated_hyps = [
                vh.hypothesis
                for vh in validation_report.validated_hypotheses
            ]
            result.validated_survivors = validated_hyps
            logger.info(
                "Validation pipeline: %d/%d survived all stages",
                len(validated_hyps),
                len(statistical_survivors),
            )
    except Exception as e:
        logger.error("Validation pipeline failed: %s", e)
        raise

    # Step 8: Design entry/exit rules for final survivors
    # Backtest only on train_data to avoid leaking holdout information.
    logger.info("Step 8: Designing strategy rules")
    strategy_results: list[dict] = []
    try:
        if result.validated_survivors:
            designer = EntryExitDesigner()
            for hyp in result.validated_survivors:
                rules = designer.design_rules(hyp, train_data)
                bt_result = designer.backtest_rules(rules, hyp, train_data)
                strategy_results.append({
                    "hypothesis_id": hyp.id,
                    "rules": {
                        "entry_threshold": rules.entry_threshold,
                        "confirmation_bars": rules.confirmation_bars,
                        "exit_type": rules.exit_type,
                        "stop_loss_atr_mult": rules.stop_loss_atr_mult,
                        "profit_target_atr_mult": rules.profit_target_atr_mult,
                        "max_holding_period": rules.max_holding_period,
                        "trailing_stop_atr_mult": rules.trailing_stop_atr_mult,
                    },
                    "backtest_metrics": bt_result.metrics,
                    "n_trades": len(bt_result.trades),
                })
            result.strategies = strategy_results
            logger.info("Designed strategies for %d hypotheses", len(strategy_results))
    except Exception as e:
        logger.error("Strategy design failed: %s", e)
        raise

    # Step 9: Apply position sizing (half-Kelly)
    # Use per-trade average win/loss magnitudes for Kelly criterion,
    # not expectancy (which combines wins and losses) or max_drawdown
    # (which is a path-dependent statistic, not per-trade loss).
    logger.info("Step 9: Computing position sizes")
    try:
        sizer = PositionSizer()
        for sr in strategy_results:
            metrics = sr.get("backtest_metrics", {})
            hit_rate = metrics.get("hit_rate", 0.5)
            # Compute avg_win and avg_loss from trade-level data
            # avg_win: average magnitude of winning trades
            # avg_loss: average magnitude of losing trades
            avg_win = max(metrics.get("avg_win", 0.001), 0.001)
            avg_loss = max(abs(metrics.get("avg_loss", 0.01)), 0.001)
            position_size = sizer.kelly_criterion(
                win_rate=hit_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                half_kelly=True,
            )
            sr["position_size"] = position_size
        logger.info("Position sizes computed")
    except Exception as e:
        logger.error("Position sizing failed: %s", e)
        raise

    # Step 10: Apply risk controls
    logger.info("Step 10: Applying risk controls")
    try:
        controller = RiskController()
        proposed_positions = [
            sr.get("position_size", 0.0) for sr in strategy_results
        ]
        state = PortfolioState(
            equity=100000.0,
            current_drawdown=0.0,
            daily_pnl=0.0,
            current_regime="normal",
            position_correlations=[],
        )
        risk_result = controller.apply_controls(proposed_positions, state)
        for i, sr in enumerate(strategy_results):
            if i < len(risk_result.adjusted_positions):
                sr["adjusted_position_size"] = risk_result.adjusted_positions[i]
        logger.info(
            "Risk controls applied (triggered: %s)",
            risk_result.triggered_controls or "none",
        )
    except Exception as e:
        logger.error("Risk controls failed: %s", e)
        raise

    # Step 11: Generate report
    logger.info("Step 11: Generating report")
    try:
        reporter = ResearchReporter()
        data_info = {
            "Ticker": ticker,
            "Period": f"{years} years",
            "Total rows": len(data),
            "Date range": f"{data.index[0].date()} to {data.index[-1].date()}",
            "Features computed": len(features.columns) if "features" in dir() else 0,
        }
        report = reporter.generate_report(
            validation_report=validation_report,
            strategy_results=strategy_results,
            data_info=data_info,
            initial_count=len(all_hypotheses),
            statistical_survivors_count=len(statistical_survivors),
        )
        md_path, _ = reporter.save_report(report, output_dir)
        result.report_path = str(md_path)
        logger.info("Report saved to %s", md_path)
    except Exception as e:
        logger.error("Report generation failed: %s", e)

    logger.info("Pipeline complete")
    return result


if __name__ == "__main__":
    run_pipeline()
