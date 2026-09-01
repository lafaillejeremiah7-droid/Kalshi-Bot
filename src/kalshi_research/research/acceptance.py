from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResearchMetrics:
    test_brier: float
    baseline_brier: float
    test_log_loss: float
    baseline_log_loss: float
    calibration_error: float
    net_pnl: float
    gross_pnl: float
    max_drawdown: float
    trade_count: int
    profitable_walkforward_windows: int
    total_walkforward_windows: int
    latency_stress_net_pnl: float
    cost_stress_net_pnl: float


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    accepted: bool
    reasons: tuple[str, ...]


def evaluate_for_probability_stage(metrics: ResearchMetrics) -> PromotionDecision:
    """Conservative gates before advancing research toward a Probability Bot.

    These are defaults, not guarantees. They deliberately demand both predictive
    improvement and executable economics rather than accepting in-sample P&L.
    """
    failures: list[str] = []
    if metrics.trade_count < 500:
        failures.append("fewer than 500 out-of-sample executable decisions")
    if metrics.test_brier >= metrics.baseline_brier * 0.99:
        failures.append("Brier score improvement is under 1% versus baseline")
    if metrics.test_log_loss >= metrics.baseline_log_loss * 0.99:
        failures.append("log-loss improvement is under 1% versus baseline")
    if metrics.calibration_error > 0.03:
        failures.append("expected calibration error exceeds 3 percentage points")
    if metrics.net_pnl <= 0:
        failures.append("net out-of-sample P&L is non-positive")
    if metrics.gross_pnl > 0 and metrics.net_pnl / metrics.gross_pnl < 0.25:
        failures.append("costs consume more than 75% of gross edge")
    if metrics.total_walkforward_windows <= 0 or (
        metrics.profitable_walkforward_windows / metrics.total_walkforward_windows < 0.60
    ):
        failures.append("fewer than 60% of walk-forward windows are profitable after costs")
    if metrics.latency_stress_net_pnl <= 0:
        failures.append("edge fails latency stress")
    if metrics.cost_stress_net_pnl <= 0:
        failures.append("edge fails transaction-cost stress")
    return PromotionDecision(not failures, tuple(failures))
