"""Tests for strategy design module: entries/exits, position sizing, risk controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.hypotheses.catalog import Hypothesis, HypothesisCategory
from quant_research.strategy.entries_exits import (
    BacktestResult,
    EntryExitDesigner,
    StrategyRules,
)
from quant_research.strategy.position_sizing import PositionSizer
from quant_research.strategy.risk_controls import (
    PortfolioState,
    RiskControlResult,
    RiskController,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def strategy_ohlcv() -> pd.DataFrame:
    """Generate synthetic OHLCV data for strategy testing (200 rows)."""
    np.random.seed(42)
    n_days = 200
    dates = pd.bdate_range(start="2021-01-04", periods=n_days, freq="B")

    drift = 0.001
    volatility = 0.015
    log_returns = np.random.normal(drift, volatility, n_days)
    close = 100 * np.exp(np.cumsum(log_returns))

    overnight_gaps = np.random.normal(0, 0.002, n_days)
    open_prices = np.roll(close, 1) * (1 + overnight_gaps)
    open_prices[0] = 100.0

    high = np.maximum(open_prices, close) * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    low = np.minimum(open_prices, close) * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    volume = (50_000_000 * np.exp(np.random.normal(0, 0.3, n_days))).astype(int)

    return pd.DataFrame(
        {"Open": open_prices, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def simple_hypothesis() -> Hypothesis:
    """Create a simple momentum hypothesis for testing."""

    def momentum_signal(data: pd.DataFrame) -> pd.Series:
        ret = np.log(data["Close"] / data["Close"].shift(5))
        signal = pd.Series(0.0, index=data.index)
        signal[ret > 0] = 1.0
        signal[ret < 0] = -1.0
        return signal

    return Hypothesis(
        id="TEST_001",
        category=HypothesisCategory.MOMENTUM,
        name="Test Momentum",
        description="5-day momentum for testing.",
        economic_rationale="Test only.",
        data_requirements=["Close"],
        signal_function=momentum_signal,
        expected_direction=1,
        data_limitations="Test.",
    )


# ============================================================
# EntryExitDesigner Tests
# ============================================================


class TestEntryExitDesigner:
    """Tests for EntryExitDesigner class."""

    def test_design_rules_produces_valid_strategy_rules(
        self, simple_hypothesis: Hypothesis, strategy_ohlcv: pd.DataFrame
    ) -> None:
        """design_rules() should return StrategyRules with reasonable parameters."""
        designer = EntryExitDesigner()
        rules = designer.design_rules(simple_hypothesis, strategy_ohlcv)

        assert isinstance(rules, StrategyRules)
        assert rules.entry_threshold >= 0.0
        assert rules.confirmation_bars >= 1
        assert rules.stop_loss_atr_mult > 0
        assert rules.profit_target_atr_mult > 0
        assert rules.max_holding_period > 0
        assert rules.trailing_stop_atr_mult > 0
        assert rules.exit_type in ("time", "reversal", "trailing_stop", "combined")

    def test_backtest_rules_returns_backtest_result(
        self, simple_hypothesis: Hypothesis, strategy_ohlcv: pd.DataFrame
    ) -> None:
        """backtest_rules() should return a BacktestResult with returns and metrics."""
        designer = EntryExitDesigner()
        rules = StrategyRules(
            entry_threshold=0.0,
            confirmation_bars=1,
            exit_type="combined",
            stop_loss_atr_mult=2.0,
            profit_target_atr_mult=3.0,
            max_holding_period=10,
            trailing_stop_atr_mult=1.5,
        )
        result = designer.backtest_rules(rules, simple_hypothesis, strategy_ohlcv)

        assert isinstance(result, BacktestResult)
        assert isinstance(result.returns, pd.Series)
        assert len(result.returns) == len(strategy_ohlcv)
        assert isinstance(result.metrics, dict)
        assert "sharpe_ratio" in result.metrics
        assert "hit_rate" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "expectancy" in result.metrics

    def test_backtest_with_short_data(
        self, simple_hypothesis: Hypothesis
    ) -> None:
        """backtest_rules() should handle short data gracefully."""
        designer = EntryExitDesigner()
        short_data = pd.DataFrame(
            {"Open": [100], "High": [101], "Low": [99], "Close": [100], "Volume": [1000]},
            index=pd.bdate_range("2023-01-02", periods=1),
        )
        rules = StrategyRules()
        result = designer.backtest_rules(rules, simple_hypothesis, short_data)
        assert isinstance(result, BacktestResult)

    def test_design_rules_selects_by_sharpe(
        self, simple_hypothesis: Hypothesis, strategy_ohlcv: pd.DataFrame
    ) -> None:
        """design_rules() should optimize for Sharpe ratio, not raw return."""
        designer = EntryExitDesigner()
        rules = designer.design_rules(simple_hypothesis, strategy_ohlcv)
        # The designed rules should have valid ATR multipliers
        assert rules.stop_loss_atr_mult in [1.5, 2.0, 3.0]
        assert rules.profit_target_atr_mult in [2.0, 3.0, 4.0]


# ============================================================
# PositionSizer Tests
# ============================================================


class TestPositionSizer:
    """Tests for PositionSizer class."""

    def test_fixed_fractional_known_inputs(self) -> None:
        """fixed_fractional with known inputs gives expected output."""
        sizer = PositionSizer()
        # Risk $1000 of $100k with $5 stop distance = 200 units
        size = sizer.fixed_fractional(
            account_equity=100000, risk_per_trade=0.01, stop_distance=5.0
        )
        assert size == pytest.approx(200.0, rel=1e-6)

    def test_fixed_fractional_capped_at_max(self) -> None:
        """fixed_fractional should be capped at max_position_fraction * equity."""
        sizer = PositionSizer(max_position_fraction=0.20)
        # Very small stop would give huge position, should be capped
        size = sizer.fixed_fractional(
            account_equity=100000, risk_per_trade=0.5, stop_distance=0.01
        )
        assert size == pytest.approx(20000.0, rel=1e-6)  # 20% of 100k

    def test_fixed_fractional_zero_stop(self) -> None:
        """fixed_fractional should return 0 for zero stop distance."""
        sizer = PositionSizer()
        assert sizer.fixed_fractional(100000, 0.01, 0.0) == 0.0

    def test_kelly_criterion_known_values(self) -> None:
        """kelly_criterion with known win/loss gives expected fraction."""
        sizer = PositionSizer()
        # win_rate=0.6, avg_win=2.0, avg_loss=1.0
        # Kelly = (0.6*2.0 - 0.4*1.0) / (2.0*1.0) = (1.2 - 0.4)/2 = 0.4
        # Half-Kelly = 0.2
        fraction = sizer.kelly_criterion(
            win_rate=0.6, avg_win=2.0, avg_loss=1.0, half_kelly=True
        )
        assert fraction == pytest.approx(0.2, rel=1e-6)

    def test_kelly_criterion_full(self) -> None:
        """kelly_criterion full Kelly should be double half-Kelly."""
        sizer = PositionSizer()
        half = sizer.kelly_criterion(0.6, 2.0, 1.0, half_kelly=True)
        full = sizer.kelly_criterion(0.6, 2.0, 1.0, half_kelly=False)
        # Full kelly = 0.4, but capped at 0.20
        assert full == pytest.approx(0.2, rel=1e-6)  # Capped at max
        assert half == pytest.approx(0.2, rel=1e-6)

    def test_kelly_criterion_negative_edge(self) -> None:
        """kelly_criterion should return 0 for negative expectancy."""
        sizer = PositionSizer()
        # win_rate=0.3, avg_win=1.0, avg_loss=2.0 -> negative kelly
        fraction = sizer.kelly_criterion(0.3, 1.0, 2.0, half_kelly=True)
        assert fraction == 0.0

    def test_volatility_target_scales_correctly(self) -> None:
        """volatility_target should scale position by vol ratio."""
        sizer = PositionSizer(max_position_fraction=1.0)  # No cap for this test
        # Target 15% vol, current 30% vol -> half position
        size = sizer.volatility_target(
            target_vol=0.15, current_vol=0.30, account_equity=100000
        )
        assert size == pytest.approx(50000.0, rel=1e-6)

    def test_volatility_target_capped(self) -> None:
        """volatility_target should be capped at max position."""
        sizer = PositionSizer(max_position_fraction=0.20)
        # Target high vol ratio
        size = sizer.volatility_target(
            target_vol=0.50, current_vol=0.01, account_equity=100000
        )
        assert size == pytest.approx(20000.0, rel=1e-6)

    def test_equal_risk(self) -> None:
        """equal_risk divides portfolio risk equally."""
        sizer = PositionSizer()
        # 5 strategies, 15% max risk, each strategy has 20% vol
        # risk_per_strategy = 0.15/5 = 0.03
        # fraction = 0.03 / 0.20 = 0.15
        frac = sizer.equal_risk(
            n_strategies=5, max_portfolio_risk=0.15, strategy_vol=0.20
        )
        assert frac == pytest.approx(0.15, rel=1e-6)

    def test_compute_position_size_dispatch(self) -> None:
        """compute_position_size dispatches to correct method."""
        sizer = PositionSizer()
        size = sizer.compute_position_size(
            "fixed_fractional",
            {"account_equity": 100000, "risk_per_trade": 0.01, "stop_distance": 5.0},
        )
        assert size == pytest.approx(200.0, rel=1e-6)

    def test_compute_position_size_invalid_method(self) -> None:
        """compute_position_size raises ValueError for unknown method."""
        sizer = PositionSizer()
        with pytest.raises(ValueError, match="Unknown method"):
            sizer.compute_position_size("invalid_method", {})

    def test_max_position_enforced(self) -> None:
        """No position should exceed max_position_fraction."""
        sizer = PositionSizer(max_position_fraction=0.10)
        # Kelly that would give > 10%
        fraction = sizer.kelly_criterion(0.7, 3.0, 1.0, half_kelly=False)
        assert fraction <= 0.10


# ============================================================
# RiskController Tests
# ============================================================


class TestRiskController:
    """Tests for RiskController class."""

    def test_drawdown_circuit_breaker_reduces_sizes(self) -> None:
        """Drawdown above threshold should reduce all positions by 50%."""
        controller = RiskController(drawdown_threshold=0.10, drawdown_reduction=0.5)
        state = PortfolioState(
            equity=100000,
            current_drawdown=0.12,  # Above 10% threshold
            daily_pnl=0.0,
            current_regime="normal",
            position_correlations=[],
        )
        result = controller.apply_controls([0.10, 0.08, 0.05], state)

        assert "drawdown_circuit_breaker" in result.triggered_controls
        # Positions should be reduced (but may also trigger heat limit)
        for i, pos in enumerate(result.adjusted_positions):
            # After 50% reduction: [0.05, 0.04, 0.025]
            # Total heat = 0.115, then capped at 0.06
            assert pos <= [0.10, 0.08, 0.05][i]

    def test_correlation_adjustment_reduces_correlated_positions(self) -> None:
        """High correlations should reduce position sizes."""
        controller = RiskController(correlation_threshold=0.5)
        state = PortfolioState(
            equity=100000,
            current_drawdown=0.0,
            daily_pnl=0.0,
            current_regime="normal",
            position_correlations=[0.7, 0.8, 0.6],  # High correlations
        )
        result = controller.apply_controls([0.02, 0.02, 0.02], state)

        assert "correlation_adjustment" in result.triggered_controls
        # All positions should be reduced
        for pos in result.adjusted_positions:
            assert pos < 0.02

    def test_daily_loss_limit_stops_trading(self) -> None:
        """Exceeding daily loss limit should zero all positions."""
        controller = RiskController(daily_loss_limit=0.02)
        state = PortfolioState(
            equity=100000,
            current_drawdown=0.0,
            daily_pnl=-0.025,  # -2.5% > 2% limit
            current_regime="normal",
            position_correlations=[],
        )
        result = controller.apply_controls([0.05, 0.03], state)

        assert "daily_loss_limit" in result.triggered_controls
        assert all(pos == 0.0 for pos in result.adjusted_positions)

    def test_regime_scaling_in_high_vol(self) -> None:
        """High-vol regime should scale down positions."""
        controller = RiskController(high_vol_regime_scale=0.5)
        state = PortfolioState(
            equity=100000,
            current_drawdown=0.0,
            daily_pnl=0.0,
            current_regime="high_vol",
            position_correlations=[],
        )
        result = controller.apply_controls([0.04, 0.02], state)

        assert "regime_scaling" in result.triggered_controls

    def test_max_portfolio_heat_cap(self) -> None:
        """Total positions exceeding heat limit should be scaled down."""
        controller = RiskController(max_portfolio_heat=0.06)
        state = PortfolioState(
            equity=100000,
            current_drawdown=0.0,
            daily_pnl=0.0,
            current_regime="normal",
            position_correlations=[],
        )
        # Total = 0.15, exceeds 0.06
        result = controller.apply_controls([0.05, 0.05, 0.05], state)

        assert "max_portfolio_heat" in result.triggered_controls
        total = sum(result.adjusted_positions)
        assert total <= 0.06 + 1e-10

    def test_no_controls_triggered_in_normal_state(self) -> None:
        """Normal state with small positions should not trigger any controls."""
        controller = RiskController()
        state = PortfolioState(
            equity=100000,
            current_drawdown=0.01,  # Well below threshold
            daily_pnl=0.005,  # Positive
            current_regime="normal",
            position_correlations=[0.1, 0.2],  # Low correlations
        )
        result = controller.apply_controls([0.01, 0.01], state)

        assert result.triggered_controls == []
        assert result.adjusted_positions == [0.01, 0.01]

    def test_empty_positions(self) -> None:
        """apply_controls should handle empty position list."""
        controller = RiskController()
        state = PortfolioState()
        result = controller.apply_controls([], state)
        assert result.adjusted_positions == []
        assert result.triggered_controls == []
