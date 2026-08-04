"""
Portfolio-level risk controls for strategy management.

Implements risk constraints including maximum portfolio heat, correlation
adjustments, drawdown circuit breakers, regime scaling, and daily loss limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PortfolioState:
    """Current state of the portfolio for risk assessment.

    Attributes
    ----------
    equity : float
        Current portfolio equity.
    current_drawdown : float
        Current drawdown from peak (as positive fraction, e.g., 0.05 = 5%).
    daily_pnl : float
        Today's P&L as fraction of equity.
    current_regime : str
        Current market regime ('low_vol', 'normal', 'high_vol').
    position_correlations : list[float]
        Pairwise correlations between active positions.
    """

    equity: float = 100000.0
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    current_regime: str = "normal"
    position_correlations: list[float] = field(default_factory=list)


@dataclass
class RiskControlResult:
    """Result of applying risk controls.

    Attributes
    ----------
    adjusted_positions : list[float]
        Position sizes after risk controls are applied.
    triggered_controls : list[str]
        Names of controls that were triggered.
    scaling_factor : float
        Overall scaling factor applied (1.0 = no change).
    """

    adjusted_positions: list[float] = field(default_factory=list)
    triggered_controls: list[str] = field(default_factory=list)
    scaling_factor: float = 1.0


class RiskController:
    """Portfolio-level risk management controller.

    Applies multiple risk constraints to proposed position sizes:
    - Maximum portfolio heat (total capital at risk)
    - Correlation adjustment (reduce when correlated)
    - Drawdown circuit breaker (reduce after drawdown)
    - Regime scaling (reduce in high-vol regimes)
    - Daily loss limit (stop trading after daily loss)

    Parameters
    ----------
    max_portfolio_heat : float, optional
        Maximum total capital at risk across all positions.
        Default is 0.06 (6%).
    correlation_threshold : float, optional
        Correlation above which positions are reduced.
        Default is 0.5.
    drawdown_threshold : float, optional
        Drawdown level that triggers circuit breaker.
        Default is 0.10 (10%).
    drawdown_reduction : float, optional
        Factor to reduce positions by when circuit breaker triggers.
        Default is 0.5 (reduce by 50%).
    daily_loss_limit : float, optional
        Maximum daily loss before stopping trading.
        Default is 0.02 (2%).
    high_vol_regime_scale : float, optional
        Scaling factor for high-vol regime. Default is 0.5.

    Examples
    --------
    >>> controller = RiskController()
    >>> state = PortfolioState(equity=100000, current_drawdown=0.12)
    >>> result = controller.apply_controls([0.10, 0.08, 0.05], state)
    >>> result.triggered_controls
    ['drawdown_circuit_breaker']
    """

    def __init__(
        self,
        max_portfolio_heat: float = 0.06,
        correlation_threshold: float = 0.5,
        drawdown_threshold: float = 0.10,
        drawdown_reduction: float = 0.5,
        daily_loss_limit: float = 0.02,
        high_vol_regime_scale: float = 0.5,
    ) -> None:
        self.max_portfolio_heat = max_portfolio_heat
        self.correlation_threshold = correlation_threshold
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_reduction = drawdown_reduction
        self.daily_loss_limit = daily_loss_limit
        self.high_vol_regime_scale = high_vol_regime_scale

    def apply_controls(
        self,
        proposed_positions: list[float],
        portfolio_state: PortfolioState,
    ) -> RiskControlResult:
        """Apply all risk controls to proposed position sizes.

        Parameters
        ----------
        proposed_positions : list[float]
            Proposed position sizes as fractions of equity.
        portfolio_state : PortfolioState
            Current portfolio state.

        Returns
        -------
        RiskControlResult
            Adjusted positions and triggered controls.
        """
        if not proposed_positions:
            return RiskControlResult()

        adjusted = list(proposed_positions)
        triggered: list[str] = []
        overall_scale = 1.0

        # 1. Daily loss limit check
        if portfolio_state.daily_pnl < -self.daily_loss_limit:
            adjusted = [0.0] * len(adjusted)
            triggered.append("daily_loss_limit")
            return RiskControlResult(
                adjusted_positions=adjusted,
                triggered_controls=triggered,
                scaling_factor=0.0,
            )

        # 2. Drawdown circuit breaker
        if portfolio_state.current_drawdown > self.drawdown_threshold:
            scale = self.drawdown_reduction
            adjusted = [pos * scale for pos in adjusted]
            overall_scale *= scale
            triggered.append("drawdown_circuit_breaker")

        # 3. Regime scaling
        if portfolio_state.current_regime == "high_vol":
            scale = self.high_vol_regime_scale
            adjusted = [pos * scale for pos in adjusted]
            overall_scale *= scale
            triggered.append("regime_scaling")

        # 4. Correlation adjustment
        correlations = portfolio_state.position_correlations
        if correlations:
            avg_corr = float(np.mean([
                abs(c) for c in correlations
            ]))
            if avg_corr > self.correlation_threshold:
                corr_scale = 1.0 - (avg_corr - self.correlation_threshold)
                corr_scale = max(corr_scale, 0.3)  # Floor at 30%
                adjusted = [pos * corr_scale for pos in adjusted]
                overall_scale *= corr_scale
                triggered.append("correlation_adjustment")

        # 5. Maximum portfolio heat
        total_heat = sum(abs(pos) for pos in adjusted)
        if total_heat > self.max_portfolio_heat:
            heat_scale = self.max_portfolio_heat / total_heat
            adjusted = [pos * heat_scale for pos in adjusted]
            overall_scale *= heat_scale
            triggered.append("max_portfolio_heat")

        return RiskControlResult(
            adjusted_positions=adjusted,
            triggered_controls=triggered,
            scaling_factor=overall_scale,
        )
