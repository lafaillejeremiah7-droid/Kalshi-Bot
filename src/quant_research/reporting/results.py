"""
Research reporting module for generating comprehensive pipeline results.

Generates markdown reports with executive summaries, methodology descriptions,
detailed results for each surviving hypothesis, comparison tables, and
data limitation disclaimers.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from quant_research.strategy.entries_exits import BacktestResult, StrategyRules
from quant_research.validation.pipeline import ValidationReport, ValidatedHypothesis

logger = logging.getLogger(__name__)


DATA_LIMITATIONS_DISCLAIMER = """
## DATA LIMITATIONS DISCLAIMER

**IMPORTANT: All analysis in this report uses OHLCV (Open, High, Low, Close, Volume) data ONLY.**

This pipeline does NOT have access to:
- Level II / order book data
- Bid-ask spread information
- Trade-level (tick) data or Time & Sales
- True order flow or market microstructure data
- Dark pool activity or hidden liquidity

**Order flow proxy hypotheses** are inferred from price-volume relationships and
CANNOT capture true order flow dynamics. These signals proxy order flow behavior
but cannot represent bid-ask dynamics, queue position, or hidden liquidity.

Volume data reflects exchange-reported daily totals only and may not include
all trading venues. VWAP calculations are approximations using typical price,
not true tick-level VWAP.

All results should be interpreted with these limitations in mind.
"""


class ResearchReporter:
    """Generates comprehensive research reports from pipeline results.

    Produces markdown reports with all required sections including executive
    summary, methodology, results, and prominently displayed data limitations.

    Examples
    --------
    >>> reporter = ResearchReporter()
    >>> report = reporter.generate_report(validation_results, strategy_results, data_info)
    >>> reporter.save_report(report, "results/")
    """

    def generate_report(
        self,
        validation_report: ValidationReport | None = None,
        strategy_results: list[dict] | None = None,
        data_info: dict | None = None,
        initial_count: int = 0,
        statistical_survivors_count: int = 0,
    ) -> str:
        """Generate a full markdown research report.

        Parameters
        ----------
        validation_report : ValidationReport or None
            Results from the validation pipeline.
        strategy_results : list[dict] or None
            Strategy design results for each survivor.
        data_info : dict or None
            Information about the data used (ticker, period, rows, etc.).
        initial_count : int
            Total number of hypotheses generated.
        statistical_survivors_count : int
            Number surviving initial statistical testing.

        Returns
        -------
        str
            Complete markdown report.
        """
        sections: list[str] = []

        # Title
        sections.append("# Quantitative Research Pipeline Report\n")

        # Data Limitations (prominent)
        sections.append(DATA_LIMITATIONS_DISCLAIMER)

        # Executive Summary
        sections.append(self._executive_summary(
            validation_report, initial_count, statistical_survivors_count
        ))

        # Methodology
        sections.append(self._methodology_section())

        # Data Information
        if data_info:
            sections.append(self._data_info_section(data_info))

        # Results for each surviving hypothesis
        if validation_report and validation_report.validated_hypotheses:
            sections.append(self._results_section(
                validation_report, strategy_results
            ))

            # Comparison table
            sections.append(self._comparison_table(
                validation_report, strategy_results
            ))

        # Rejection Funnel
        if validation_report:
            sections.append(self._rejection_funnel(
                validation_report, initial_count, statistical_survivors_count
            ))

        # Regime Analysis Summary
        if validation_report and validation_report.regime_results:
            sections.append(self._regime_summary(validation_report))

        # Limitations and Future Work
        sections.append(self._limitations_section())

        return "\n".join(sections)

    def save_report(
        self, report: str, output_dir: str | Path
    ) -> tuple[Path, Path]:
        """Save report as markdown and summary CSV.

        Parameters
        ----------
        report : str
            Markdown report content.
        output_dir : str or Path
            Directory to save output files.

        Returns
        -------
        tuple[Path, Path]
            Paths to (markdown_file, csv_file).
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        md_path = output_path / "research_report.md"
        csv_path = output_path / "summary.csv"

        md_path.write_text(report, encoding="utf-8")
        logger.info(f"Report saved to {md_path}")

        # Generate summary CSV
        self._save_summary_csv(csv_path, report)
        logger.info(f"Summary saved to {csv_path}")

        return md_path, csv_path

    def _executive_summary(
        self,
        validation_report: ValidationReport | None,
        initial_count: int,
        statistical_survivors_count: int,
    ) -> str:
        """Generate executive summary section."""
        lines = [
            "## Executive Summary\n",
            f"- **Hypotheses generated:** {initial_count}",
            f"- **Survived initial statistical testing (FDR-corrected):** {statistical_survivors_count}",
        ]

        if validation_report:
            funnel = validation_report.rejection_funnel
            lines.extend([
                f"- **Entered validation pipeline:** {funnel.initial_count}",
                f"- **Survived walk-forward validation:** {funnel.walk_forward_survivors}",
                f"- **Survived out-of-sample testing:** {funnel.oos_survivors}",
                f"- **Flagged as regime-dependent:** {funnel.regime_flagged}",
                f"- **Survived transaction cost analysis:** {funnel.final_survivors}",
                f"- **Final validated strategies:** {len(validation_report.validated_hypotheses)}",
            ])
        else:
            lines.append("- **Validation:** Not completed (insufficient survivors)")

        lines.append("")
        return "\n".join(lines)

    def _methodology_section(self) -> str:
        """Generate methodology description."""
        return """## Methodology

The pipeline follows a rigorous multi-stage process:

1. **Hypothesis Generation:** Programmatic generation of 100+ market behavior
   hypotheses across categories (momentum, mean reversion, volatility, gaps,
   session effects, order flow proxy, regime, microstructure proxy).

2. **Statistical Testing:** Each hypothesis is tested for statistical significance
   using t-tests on signal-conditional returns. Raw p-values are computed for
   all hypotheses simultaneously.

3. **FDR Correction:** Benjamini-Hochberg False Discovery Rate correction is
   applied to control for multiple comparisons. Only hypotheses with adjusted
   p-values below 0.05 survive, along with minimum Sharpe and observation
   count thresholds.

4. **Walk-Forward Validation:** Expanding-window walk-forward analysis across
   multiple folds tests out-of-sample consistency. Hypotheses must show
   positive expectancy in at least 60% of folds.

5. **Out-of-Sample Testing:** Pure holdout validation with stricter thresholds
   (lower p-value, higher Sharpe minimum) on data never seen during development.

6. **Regime Analysis:** Performance is evaluated across market regimes
   (bull trending, bear trending, sideways, high-vol crisis). Hypotheses
   performing in fewer than 2 regimes are flagged as regime-dependent.

7. **Transaction Cost Analysis:** Realistic costs (commission, spread, slippage,
   market impact) are applied. Only hypotheses maintaining positive edge
   after costs survive.

8. **Strategy Design:** Entry/exit rules are optimized on training data using
   risk-adjusted metrics (Sharpe ratio), with position sizing (half-Kelly)
   and portfolio risk controls applied.

"""

    def _data_info_section(self, data_info: dict) -> str:
        """Generate data information section."""
        lines = ["## Data Information\n"]
        for key, value in data_info.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")
        return "\n".join(lines)

    def _results_section(
        self,
        validation_report: ValidationReport,
        strategy_results: list[dict] | None,
    ) -> str:
        """Generate detailed results for each surviving hypothesis."""
        lines = ["## Detailed Results\n"]

        strategy_map: dict[str, dict] = {}
        if strategy_results:
            for sr in strategy_results:
                hyp_id = sr.get("hypothesis_id", "")
                strategy_map[hyp_id] = sr

        for vh in validation_report.validated_hypotheses:
            hyp = vh.hypothesis
            lines.append(f"### {hyp.name} ({hyp.id})\n")
            lines.append(f"**Category:** {hyp.category.value}")
            lines.append(f"**Description:** {hyp.description}")
            lines.append(f"**Economic Rationale:** {hyp.economic_rationale}")
            lines.append(f"**Data Limitations:** {hyp.data_limitations}")
            lines.append("")

            # Walk-forward metrics
            wf = vh.walk_forward_result
            lines.append("**Walk-Forward Results:**")
            lines.append(f"- Consistency ratio: {wf.consistency_ratio:.2f}")
            lines.append(f"- Avg IS Sharpe: {wf.avg_in_sample_sharpe:.3f}")
            lines.append(f"- Avg OOS Sharpe: {wf.avg_out_of_sample_sharpe:.3f}")
            lines.append(f"- Performance degradation: {wf.performance_degradation:.2f}")
            lines.append("")

            # OOS metrics
            oos = vh.oos_result
            lines.append("**Out-of-Sample Results:**")
            lines.append(f"- Sharpe ratio: {oos.sharpe_ratio:.3f}")
            lines.append(f"- Hit rate: {oos.hit_rate:.2%}")
            lines.append(f"- Expectancy: {oos.expectancy:.6f}")
            lines.append(f"- Max drawdown: {oos.max_drawdown:.2%}")
            lines.append(f"- Profit factor: {oos.profit_factor:.2f}")
            lines.append(f"- Number of trades: {oos.n_trades}")
            lines.append("")

            # Regime results
            regime = vh.regime_result
            lines.append("**Regime Analysis:**")
            lines.append(f"- Regime robust: {'Yes' if regime.regime_robust else 'No'}")
            lines.append(f"- Regimes with positive expectancy: {regime.regimes_with_positive_expectancy}")
            if regime.dominant_regime:
                lines.append(f"- Dominant regime: {regime.dominant_regime.value}")
            lines.append("")

            # Cost-adjusted metrics
            cost = vh.cost_result
            lines.append("**Net-of-Cost Performance:**")
            lines.append(f"- Gross Sharpe: {cost.gross_sharpe:.3f}")
            lines.append(f"- Net Sharpe: {cost.net_sharpe:.3f}")
            lines.append(f"- Net expectancy: {cost.net_expectancy:.6f}")
            lines.append(f"- Cost per trade: {cost.total_cost_per_trade:.6f}")
            lines.append("")

            # Strategy rules if available
            sr = strategy_map.get(hyp.id)
            if sr and "rules" in sr:
                rules = sr["rules"]
                lines.append("**Strategy Rules:**")
                lines.append(f"- Entry threshold: {rules.get('entry_threshold', 'N/A')}")
                lines.append(f"- Confirmation bars: {rules.get('confirmation_bars', 'N/A')}")
                lines.append(f"- Stop-loss (ATR mult): {rules.get('stop_loss_atr_mult', 'N/A')}")
                lines.append(f"- Profit target (ATR mult): {rules.get('profit_target_atr_mult', 'N/A')}")
                lines.append(f"- Max holding period: {rules.get('max_holding_period', 'N/A')}")
                lines.append(f"- Trailing stop (ATR mult): {rules.get('trailing_stop_atr_mult', 'N/A')}")
                lines.append("")

            lines.append("---\n")

        return "\n".join(lines)

    def _comparison_table(
        self,
        validation_report: ValidationReport,
        strategy_results: list[dict] | None,
    ) -> str:
        """Generate comparison table ranked by net Sharpe."""
        lines = ["## Strategy Comparison (Ranked by Net Sharpe)\n"]
        lines.append("| Rank | ID | Name | Net Sharpe | OOS Sharpe | Hit Rate | Regime Robust |")
        lines.append("|------|-------|------|------------|------------|----------|---------------|")

        # Sort by net Sharpe
        survivors = sorted(
            validation_report.validated_hypotheses,
            key=lambda vh: vh.cost_result.net_sharpe,
            reverse=True,
        )

        for rank, vh in enumerate(survivors, 1):
            hyp = vh.hypothesis
            robust = "Yes" if vh.regime_result.regime_robust else "No"
            lines.append(
                f"| {rank} | {hyp.id} | {hyp.name} | "
                f"{vh.cost_result.net_sharpe:.3f} | "
                f"{vh.oos_result.sharpe_ratio:.3f} | "
                f"{vh.oos_result.hit_rate:.2%} | "
                f"{robust} |"
            )

        lines.append("")
        return "\n".join(lines)

    def _rejection_funnel(
        self,
        validation_report: ValidationReport,
        initial_count: int,
        statistical_survivors_count: int,
    ) -> str:
        """Generate rejection funnel visualization."""
        funnel = validation_report.rejection_funnel
        lines = ["## Rejection Funnel\n"]
        lines.append("```")
        lines.append(f"Hypotheses Generated:        {initial_count:>4}")
        lines.append(f"  |")
        lines.append(f"  v  Statistical Testing + FDR")
        lines.append(f"Survived Statistics:          {statistical_survivors_count:>4}  "
                     f"(rejected: {initial_count - statistical_survivors_count})")
        lines.append(f"  |")
        lines.append(f"  v  Walk-Forward Validation")
        lines.append(f"Survived Walk-Forward:        {funnel.walk_forward_survivors:>4}  "
                     f"(rejected: {funnel.walk_forward_rejected})")
        lines.append(f"  |")
        lines.append(f"  v  Out-of-Sample Testing")
        lines.append(f"Survived OOS:                 {funnel.oos_survivors:>4}  "
                     f"(rejected: {funnel.oos_rejected})")
        lines.append(f"  |")
        lines.append(f"  v  Transaction Cost Analysis")
        lines.append(f"Final Survivors:              {funnel.final_survivors:>4}  "
                     f"(rejected: {funnel.cost_rejected})")
        lines.append(f"  |")
        lines.append(f"  (Regime-dependent flagged: {funnel.regime_flagged})")
        lines.append("```\n")
        return "\n".join(lines)

    def _regime_summary(self, validation_report: ValidationReport) -> str:
        """Generate regime analysis summary."""
        lines = ["## Regime Analysis Summary\n"]

        robust_list: list[str] = []
        dependent_list: list[str] = []

        for vh in validation_report.validated_hypotheses:
            name = f"{vh.hypothesis.name} ({vh.hypothesis.id})"
            if vh.regime_result.regime_robust:
                robust_list.append(name)
            else:
                dependent_list.append(name)

        lines.append("**Regime-Robust Hypotheses:**")
        if robust_list:
            for name in robust_list:
                lines.append(f"- {name}")
        else:
            lines.append("- None")
        lines.append("")

        lines.append("**Regime-Dependent Hypotheses:**")
        if dependent_list:
            for name in dependent_list:
                lines.append(f"- {name}")
        else:
            lines.append("- None")
        lines.append("")

        return "\n".join(lines)

    def _limitations_section(self) -> str:
        """Generate limitations and future work section."""
        return """## Limitations and Future Work

### Current Limitations

1. **OHLCV-Only Data:** All signals are derived from daily OHLCV bars. True order
   flow, market microstructure, and intraday dynamics cannot be captured.

2. **Single Asset:** Analysis is conducted on a single ETF (QQQ). Cross-sectional
   momentum and relative value signals are not available.

3. **Transaction Cost Assumptions:** Cost model uses estimates for spread, slippage,
   and market impact. Actual execution costs depend on order size, timing, and
   market conditions.

4. **Regime Classification:** Rule-based regime identification may lag actual
   regime transitions.

5. **Survivorship Bias:** Using QQQ (a successful ETF) introduces mild survivorship
   bias in long-term analysis.

### Future Work

- Incorporate intraday data for higher-frequency signals
- Add cross-asset analysis (SPY, IWM, sector ETFs)
- Implement adaptive parameter optimization
- Add options-derived signals (VIX, skew, term structure)
- Build live paper-trading framework for real-time validation
- Incorporate alternative data sources (sentiment, flows, positioning)

"""

    def _save_summary_csv(self, csv_path: Path, report: str) -> None:
        """Save a summary CSV with key metrics."""
        # Parse basic info from report - write a minimal summary
        rows = [["Section", "Content"]]
        lines = report.split("\n")
        current_section = ""
        for line in lines:
            if line.startswith("## "):
                current_section = line.replace("## ", "").strip()
            elif line.startswith("- **") and current_section:
                rows.append([current_section, line.strip("- ")])

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
