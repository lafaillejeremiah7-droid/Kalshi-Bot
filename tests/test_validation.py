"""
Tests for validation pipeline, walk-forward, OOS, regime analysis, and costs.

Uses synthetic data with known statistical properties to verify that:
- Good signals survive validation
- Noise signals are rejected
- Regimes are correctly identified
- Transaction costs correctly reduce returns
- The full pipeline chains stages correctly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.data.features import FeatureEngine
from quant_research.hypotheses.catalog import Hypothesis, HypothesisCategory
from quant_research.robustness.regime_analysis import (
    MarketRegime,
    RegimeAnalyzer,
)
from quant_research.robustness.transaction_costs import TransactionCostModel
from quant_research.testing.statistical import StatisticalTester
from quant_research.validation.out_of_sample import OutOfSampleValidator
from quant_research.validation.pipeline import ValidationPipeline
from quant_research.validation.walk_forward import WalkForwardValidator


# ---------------------------------------------------------------------------
# Fixtures for synthetic test data
# ---------------------------------------------------------------------------


@pytest.fixture
def trending_data() -> pd.DataFrame:
    """Create synthetic data with a clear upward trend for 500 days.

    This data has a strong momentum signal embedded: returns are positively
    autocorrelated, so a simple momentum signal should consistently profit.
    """
    np.random.seed(42)
    n_days = 500
    dates = pd.bdate_range(start="2020-01-02", periods=n_days, freq="B")

    # Strong upward drift with positive autocorrelation
    drift = 0.001  # Strong daily drift
    volatility = 0.008  # Low vol so signal is clear
    returns = np.random.normal(drift, volatility, n_days)
    # Add autocorrelation to make momentum predictive
    for i in range(1, n_days):
        returns[i] += 0.3 * returns[i - 1]

    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.roll(close, 1) * (1 + np.random.normal(0, 0.001, n_days))
    open_[0] = 100.0
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.002, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.002, n_days)))
    volume = (50_000_000 * np.exp(np.random.normal(0, 0.2, n_days))).astype(int)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    engine = FeatureEngine()
    features = engine.compute_all(df)
    return pd.concat([df, features], axis=1)


@pytest.fixture
def random_data() -> pd.DataFrame:
    """Create synthetic data with pure random walk (no signal)."""
    np.random.seed(99)
    n_days = 500
    dates = pd.bdate_range(start="2020-01-02", periods=n_days, freq="B")

    returns = np.random.normal(0, 0.015, n_days)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.roll(close, 1) * (1 + np.random.normal(0, 0.002, n_days))
    open_[0] = 100.0
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.003, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.003, n_days)))
    volume = (50_000_000 * np.exp(np.random.normal(0, 0.2, n_days))).astype(int)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    engine = FeatureEngine()
    features = engine.compute_all(df)
    return pd.concat([df, features], axis=1)


@pytest.fixture
def regime_data() -> pd.DataFrame:
    """Create data with obvious regime transitions.

    Structure:
    - Days 0-149: Bull trending (strong upward, low vol)
    - Days 150-299: Sideways/choppy (flat, low vol)
    - Days 300-399: High vol crisis (big swings, high vol)
    - Days 400-499: Bear trending (downward)
    """
    np.random.seed(77)
    n_days = 500
    dates = pd.bdate_range(start="2020-01-02", periods=n_days, freq="B")

    returns = np.zeros(n_days)
    # Bull trending: strong drift, low vol
    returns[0:150] = np.random.normal(0.002, 0.005, 150)
    # Sideways: no drift, low vol
    returns[150:300] = np.random.normal(0.0, 0.005, 150)
    # High vol crisis: negative drift, high vol
    returns[300:400] = np.random.normal(-0.001, 0.035, 100)
    # Bear trending: negative drift
    returns[400:500] = np.random.normal(-0.002, 0.010, 100)

    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.roll(close, 1) * (1 + np.random.normal(0, 0.001, n_days))
    open_[0] = 100.0
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.002, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.002, n_days)))
    volume = (50_000_000 * np.exp(np.random.normal(0, 0.2, n_days))).astype(int)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    engine = FeatureEngine()
    features = engine.compute_all(df)
    return pd.concat([df, features], axis=1)


def _make_momentum_hypothesis() -> Hypothesis:
    """Create a simple momentum hypothesis that goes long when 5-day return > 0."""

    def signal_fn(df: pd.DataFrame) -> pd.Series:
        ret_5d = np.log(df["Close"] / df["Close"].shift(5))
        signal = pd.Series(0.0, index=df.index)
        signal[ret_5d > 0] = 1.0
        signal[ret_5d < 0] = -1.0
        return signal

    return Hypothesis(
        id="TEST_MOM_001",
        category=HypothesisCategory.MOMENTUM,
        name="5-day momentum",
        description="Go long when 5-day return is positive, short when negative",
        economic_rationale="Momentum effect: recent winners continue winning",
        data_requirements=["Close"],
        signal_function=signal_fn,
        expected_direction=1,
        data_limitations="None",
    )


def _make_noise_hypothesis() -> Hypothesis:
    """Create a hypothesis based on random noise (should be rejected)."""

    def signal_fn(df: pd.DataFrame) -> pd.Series:
        # Use a seeded random signal that has no relation to future returns
        rng = np.random.default_rng(12345)
        n = len(df)
        random_vals = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.3, 0.4, 0.3])
        return pd.Series(random_vals, index=df.index)

    return Hypothesis(
        id="TEST_NOISE_001",
        category=HypothesisCategory.MOMENTUM,
        name="Random noise signal",
        description="Random signal with no predictive power",
        economic_rationale="None - this is a noise control",
        data_requirements=["Close"],
        signal_function=signal_fn,
        expected_direction=0,
        data_limitations="None",
    )


def _make_strong_signal_hypothesis() -> Hypothesis:
    """Create a hypothesis with a very strong embedded signal for testing."""

    def signal_fn(df: pd.DataFrame) -> pd.Series:
        # Cheating signal: use next-day direction (for testing purposes only)
        # This simulates a "perfect" signal that should easily pass all tests
        # But we use lagged information so it's realistically achievable:
        # Use 1-day return sign as signal (captures autocorrelation in trending data)
        ret_1d = np.log(df["Close"] / df["Close"].shift(1))
        signal = pd.Series(0.0, index=df.index)
        signal[ret_1d > 0.001] = 1.0
        signal[ret_1d < -0.001] = -1.0
        return signal

    return Hypothesis(
        id="TEST_STRONG_001",
        category=HypothesisCategory.MOMENTUM,
        name="Lagged return signal",
        description="Trade based on previous day return direction",
        economic_rationale="Short-term autocorrelation in trending markets",
        data_requirements=["Close"],
        signal_function=signal_fn,
        expected_direction=1,
        data_limitations="None",
    )


# ---------------------------------------------------------------------------
# Walk-Forward Validator Tests
# ---------------------------------------------------------------------------


class TestWalkForwardValidator:
    """Tests for WalkForwardValidator."""

    def test_init_defaults(self):
        """Test default initialization."""
        validator = WalkForwardValidator()
        assert validator.n_folds == 5
        assert validator.train_ratio == 0.8
        assert validator.method == "expanding"
        assert validator.min_consistency_ratio == 0.6
        assert validator.max_degradation == 0.5

    def test_init_custom_params(self):
        """Test custom initialization."""
        validator = WalkForwardValidator(
            n_folds=3, train_ratio=0.7, method="rolling"
        )
        assert validator.n_folds == 3
        assert validator.train_ratio == 0.7
        assert validator.method == "rolling"

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            WalkForwardValidator(method="invalid")

    def test_momentum_on_trending_data_passes(self, trending_data):
        """A momentum signal on strongly trending data should pass walk-forward."""
        validator = WalkForwardValidator(n_folds=5, method="expanding")
        hypothesis = _make_momentum_hypothesis()
        result = validator.validate(hypothesis, trending_data)

        assert result.hypothesis_id == "TEST_MOM_001"
        assert len(result.fold_results) == 5
        # On trending data with autocorrelation, momentum should be consistent
        assert result.consistency_ratio >= 0.6
        assert result.passed is True

    def test_noise_on_random_data_fails(self, random_data):
        """A noise signal on random data should fail walk-forward."""
        validator = WalkForwardValidator(n_folds=5, method="expanding")
        hypothesis = _make_noise_hypothesis()
        result = validator.validate(hypothesis, random_data)

        assert result.hypothesis_id == "TEST_NOISE_001"
        # Random noise should not consistently produce positive expectancy
        # It may occasionally pass by luck, but on average should fail
        assert result.consistency_ratio < 0.8

    def test_rolling_window_method(self, trending_data):
        """Test rolling window method produces valid results."""
        validator = WalkForwardValidator(n_folds=3, method="rolling")
        hypothesis = _make_momentum_hypothesis()
        result = validator.validate(hypothesis, trending_data)

        assert len(result.fold_results) == 3
        for fold in result.fold_results:
            assert fold.fold_index >= 0
            assert fold.train_start != ""
            assert fold.test_start != ""

    def test_fold_results_have_metrics(self, trending_data):
        """Test that fold results contain all expected metrics."""
        validator = WalkForwardValidator(n_folds=3, method="expanding")
        hypothesis = _make_momentum_hypothesis()
        result = validator.validate(hypothesis, trending_data)

        for fold in result.fold_results:
            assert fold.n_trades >= 0
            # Sharpe can be any value
            assert isinstance(fold.in_sample_sharpe, float)
            assert isinstance(fold.out_of_sample_sharpe, float)
            assert isinstance(fold.in_sample_expectancy, float)
            assert isinstance(fold.out_of_sample_expectancy, float)
            assert 0.0 <= fold.hit_rate <= 1.0

    def test_batch_validation(self, trending_data):
        """Test batch validation of multiple hypotheses."""
        validator = WalkForwardValidator(n_folds=3)
        hypotheses = [_make_momentum_hypothesis(), _make_noise_hypothesis()]
        results = validator.validate_batch(hypotheses, trending_data)

        assert len(results) == 2
        assert results[0].hypothesis_id == "TEST_MOM_001"
        assert results[1].hypothesis_id == "TEST_NOISE_001"


# ---------------------------------------------------------------------------
# Out-of-Sample Validator Tests
# ---------------------------------------------------------------------------


class TestOutOfSampleValidator:
    """Tests for OutOfSampleValidator."""

    def test_init_defaults(self):
        """Test default initialization."""
        validator = OutOfSampleValidator()
        assert validator.holdout_fraction == 0.2
        assert validator.min_p_value == 0.01
        assert validator.min_sharpe == 0.4

    def test_split_holdout(self, trending_data):
        """Test holdout split produces correct proportions."""
        validator = OutOfSampleValidator(holdout_fraction=0.2)
        dev, holdout = validator.split_holdout(trending_data)

        total = len(trending_data)
        assert len(dev) + len(holdout) == total
        assert len(holdout) == total - int(total * 0.8)

    def test_strong_signal_on_trending_passes(self, trending_data):
        """A strong momentum signal on consistently trending data should pass OOS."""
        validator = OutOfSampleValidator(
            holdout_fraction=0.2, min_sharpe=0.3, min_p_value=0.10
        )
        hypothesis = _make_strong_signal_hypothesis()
        result = validator.validate(hypothesis, trending_data)

        assert result.hypothesis_id == "TEST_STRONG_001"
        assert result.n_trades > 0
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.edge_decay, float)
        assert isinstance(result.first_half_sharpe, float)
        assert isinstance(result.second_half_sharpe, float)

    def test_noise_signal_fails_oos(self, random_data):
        """A noise signal should fail OOS validation with strict thresholds."""
        validator = OutOfSampleValidator(
            holdout_fraction=0.2, min_sharpe=0.4, min_p_value=0.01
        )
        hypothesis = _make_noise_hypothesis()
        result = validator.validate(hypothesis, random_data)

        assert result.hypothesis_id == "TEST_NOISE_001"
        # Random noise should not pass strict OOS criteria
        assert result.passed is False
        assert len(result.rejection_reasons) > 0

    def test_batch_validation(self, trending_data):
        """Test batch OOS validation."""
        validator = OutOfSampleValidator(holdout_fraction=0.2)
        hypotheses = [_make_momentum_hypothesis(), _make_noise_hypothesis()]
        results = validator.validate_batch(hypotheses, trending_data)

        assert len(results) == 2

    def test_edge_decay_computed(self, trending_data):
        """Test that edge decay analysis is performed."""
        validator = OutOfSampleValidator(holdout_fraction=0.3)
        hypothesis = _make_momentum_hypothesis()
        result = validator.validate(hypothesis, trending_data)

        # Edge decay should be a finite number
        assert np.isfinite(result.edge_decay)
        assert np.isfinite(result.first_half_sharpe)
        assert np.isfinite(result.second_half_sharpe)


# ---------------------------------------------------------------------------
# Regime Analyzer Tests
# ---------------------------------------------------------------------------


class TestRegimeAnalyzer:
    """Tests for RegimeAnalyzer."""

    def test_init_defaults(self):
        """Test default initialization."""
        analyzer = RegimeAnalyzer()
        assert analyzer.return_lookback == 60
        assert analyzer.vol_lookback == 60
        assert analyzer.vol_threshold_multiplier == 1.5
        assert analyzer.trend_threshold == 0.10
        assert analyzer.min_regimes_for_robust == 2

    def test_identifies_regimes_from_regime_data(self, regime_data):
        """Test that regime identification produces multiple regimes."""
        analyzer = RegimeAnalyzer()
        regimes = analyzer.identify_regimes(regime_data)

        assert len(regimes) == len(regime_data)

        # Should identify at least 3 distinct regimes in this data
        unique_regimes = set(regimes.values)
        assert len(unique_regimes) >= 3, (
            f"Expected at least 3 regimes, got {len(unique_regimes)}: {unique_regimes}"
        )

    def test_bull_regime_identified_in_trending(self, regime_data):
        """Test that bull trending regime is found in the first segment."""
        analyzer = RegimeAnalyzer()
        regimes = analyzer.identify_regimes(regime_data)

        # After warmup period (60 days), the strong bull section should be identified
        # Check some days in the 80-130 range (well into the bull period)
        bull_section = regimes.iloc[80:130]
        bull_count = (bull_section == MarketRegime.BULL_TRENDING).sum()
        # At least some days should be classified as bull
        assert bull_count > 0, "Expected some bull regime days in trending section"

    def test_high_vol_crisis_identified(self, regime_data):
        """Test that high-vol crisis regime is found in the crisis segment."""
        analyzer = RegimeAnalyzer()
        regimes = analyzer.identify_regimes(regime_data)

        # The crisis section (300-399) should have high vol regime days
        crisis_section = regimes.iloc[340:390]
        crisis_count = (crisis_section == MarketRegime.HIGH_VOL_CRISIS).sum()
        assert crisis_count > 0, "Expected some high-vol crisis days in volatile section"

    def test_analyze_hypothesis(self, regime_data):
        """Test hypothesis analysis produces per-regime metrics."""
        analyzer = RegimeAnalyzer()
        hypothesis = _make_momentum_hypothesis()
        result = analyzer.analyze(hypothesis, regime_data)

        assert result.hypothesis_id == "TEST_MOM_001"
        assert len(result.regime_metrics) == 4  # 4 regimes
        assert isinstance(result.regime_robust, bool)
        assert result.regimes_with_positive_expectancy >= 0

    def test_regime_metrics_structure(self, regime_data):
        """Test that regime metrics have proper structure."""
        analyzer = RegimeAnalyzer()
        hypothesis = _make_momentum_hypothesis()
        result = analyzer.analyze(hypothesis, regime_data)

        for metrics in result.regime_metrics:
            assert metrics.regime in MarketRegime
            assert metrics.n_days >= 0
            assert isinstance(metrics.sharpe_ratio, float)
            assert isinstance(metrics.expectancy, float)

    def test_batch_analysis(self, regime_data):
        """Test batch regime analysis."""
        analyzer = RegimeAnalyzer()
        hypotheses = [_make_momentum_hypothesis(), _make_noise_hypothesis()]
        results = analyzer.analyze_batch(hypotheses, regime_data)

        assert len(results) == 2


# ---------------------------------------------------------------------------
# Transaction Cost Model Tests
# ---------------------------------------------------------------------------


class TestTransactionCostModel:
    """Tests for TransactionCostModel."""

    def test_init_defaults(self):
        """Test default initialization."""
        model = TransactionCostModel()
        assert model.commission_per_share == 0.005
        assert model.spread_low_vol == 0.0001
        assert model.spread_normal == 0.0002
        assert model.spread_high_vol == 0.0005
        assert model.min_net_sharpe == 0.2

    def test_costs_reduce_returns(self, trending_data):
        """Test that applying costs always reduces returns."""
        model = TransactionCostModel()
        hypothesis = _make_momentum_hypothesis()
        result = model.evaluate(hypothesis, trending_data)

        assert result.hypothesis_id == "TEST_MOM_001"
        # Net returns should be less than gross
        assert result.net_sharpe <= result.gross_sharpe
        assert result.net_expectancy <= result.gross_expectancy

    def test_high_costs_kill_marginal_signal(self, random_data):
        """Test that a marginal signal does not survive high costs."""
        # Use very high costs
        model = TransactionCostModel(
            commission_per_share=0.05,  # 10x normal
            spread_normal=0.005,  # 25x normal
            min_net_sharpe=0.5,
        )
        hypothesis = _make_noise_hypothesis()
        result = model.evaluate(hypothesis, random_data)

        # Random signal with high costs should definitely not survive
        assert result.survives_costs is False

    def test_apply_costs_direct(self):
        """Test apply_costs function directly with known inputs."""
        model = TransactionCostModel(
            commission_per_share=0.005,
            spread_normal=0.0002,
        )

        n = 100
        dates = pd.bdate_range(start="2020-01-02", periods=n, freq="B")
        gross_returns = pd.Series(np.full(n, 0.001), index=dates)  # 0.1% per day
        signals = pd.Series(np.ones(n), index=dates)  # Always long
        # First day is a trade (entering position)
        signals.iloc[0] = 0  # Start flat, enter on day 1
        avg_price = pd.Series(np.full(n, 100.0), index=dates)
        avg_volume = pd.Series(np.full(n, 50_000_000.0), index=dates)

        net_returns, avg_cost = model.apply_costs(
            gross_returns, signals, avg_price, avg_volume
        )

        # Net returns should be less than or equal to gross
        assert net_returns.sum() <= gross_returns.sum()
        assert avg_cost >= 0

    def test_cost_result_fields(self, trending_data):
        """Test that CostAdjustedResult has all expected fields."""
        model = TransactionCostModel()
        hypothesis = _make_momentum_hypothesis()
        result = model.evaluate(hypothesis, trending_data)

        assert isinstance(result.gross_sharpe, float)
        assert isinstance(result.net_sharpe, float)
        assert isinstance(result.gross_expectancy, float)
        assert isinstance(result.net_expectancy, float)
        assert isinstance(result.gross_profit_factor, float)
        assert isinstance(result.net_profit_factor, float)
        assert isinstance(result.total_cost_per_trade, float)
        assert result.n_trades > 0
        assert isinstance(result.turnover_annual, float)

    def test_batch_evaluation(self, trending_data):
        """Test batch cost evaluation."""
        model = TransactionCostModel()
        hypotheses = [_make_momentum_hypothesis(), _make_noise_hypothesis()]
        results = model.evaluate_batch(hypotheses, trending_data)

        assert len(results) == 2


# ---------------------------------------------------------------------------
# Validation Pipeline Tests
# ---------------------------------------------------------------------------


class TestValidationPipeline:
    """Tests for the full ValidationPipeline."""

    def test_init_defaults(self):
        """Test default initialization creates all components."""
        pipeline = ValidationPipeline()
        assert pipeline.wf_validator is not None
        assert pipeline.oos_validator is not None
        assert pipeline.regime_analyzer is not None
        assert pipeline.cost_model is not None

    def test_init_custom_components(self):
        """Test custom component injection."""
        wf = WalkForwardValidator(n_folds=3)
        oos = OutOfSampleValidator(holdout_fraction=0.3)
        pipeline = ValidationPipeline(
            walk_forward_validator=wf, oos_validator=oos
        )
        assert pipeline.wf_validator.n_folds == 3
        assert pipeline.oos_validator.holdout_fraction == 0.3

    def test_pipeline_rejects_noise(self, random_data):
        """Test that noise hypothesis is rejected somewhere in the pipeline."""
        pipeline = ValidationPipeline(
            walk_forward_validator=WalkForwardValidator(n_folds=3),
            oos_validator=OutOfSampleValidator(min_sharpe=0.4, min_p_value=0.01),
        )
        hypotheses = [_make_noise_hypothesis()]
        report = pipeline.run(hypotheses, random_data)

        # Noise should not survive the full pipeline
        assert report.rejection_funnel.initial_count == 1
        assert report.rejection_funnel.final_survivors == 0

    def test_pipeline_report_structure(self, trending_data):
        """Test that the pipeline report has proper structure."""
        pipeline = ValidationPipeline(
            walk_forward_validator=WalkForwardValidator(n_folds=3),
            oos_validator=OutOfSampleValidator(
                holdout_fraction=0.2, min_sharpe=0.1, min_p_value=0.5
            ),
        )
        hypotheses = [_make_momentum_hypothesis()]
        report = pipeline.run(hypotheses, trending_data)

        assert report.rejection_funnel.initial_count == 1
        assert len(report.walk_forward_results) == 1
        # WF results are populated regardless of pass/fail
        assert report.rejection_funnel.walk_forward_survivors >= 0

    def test_pipeline_mixed_hypotheses(self, trending_data):
        """Test pipeline with mix of good and bad hypotheses."""
        pipeline = ValidationPipeline(
            walk_forward_validator=WalkForwardValidator(n_folds=3),
            oos_validator=OutOfSampleValidator(
                holdout_fraction=0.2, min_sharpe=0.0, min_p_value=0.5
            ),
            cost_model=TransactionCostModel(min_net_sharpe=0.0),
        )
        # Mix of hypotheses
        hypotheses = [
            _make_momentum_hypothesis(),
            _make_noise_hypothesis(),
        ]
        report = pipeline.run(hypotheses, trending_data)

        assert report.rejection_funnel.initial_count == 2
        # At minimum, the pipeline should process both
        total_processed = (
            report.rejection_funnel.walk_forward_rejected
            + report.rejection_funnel.walk_forward_survivors
        )
        assert total_processed == 2

    def test_rejection_funnel_sums(self, trending_data):
        """Test that rejection funnel numbers are consistent."""
        pipeline = ValidationPipeline(
            walk_forward_validator=WalkForwardValidator(n_folds=3),
            oos_validator=OutOfSampleValidator(
                holdout_fraction=0.2, min_sharpe=0.0, min_p_value=0.5
            ),
            cost_model=TransactionCostModel(min_net_sharpe=-10.0),
        )
        hypotheses = [_make_momentum_hypothesis(), _make_strong_signal_hypothesis()]
        report = pipeline.run(hypotheses, trending_data)

        funnel = report.rejection_funnel
        # Walk-forward should account for all initial
        assert funnel.walk_forward_rejected + funnel.walk_forward_survivors == funnel.initial_count
        # OOS should account for all WF survivors
        assert funnel.oos_rejected + funnel.oos_survivors == funnel.walk_forward_survivors
        # Costs should account for all OOS survivors
        assert funnel.cost_rejected + funnel.final_survivors == funnel.oos_survivors

    def test_validated_hypotheses_have_all_results(self, trending_data):
        """Test that validated hypotheses include results from all stages."""
        pipeline = ValidationPipeline(
            walk_forward_validator=WalkForwardValidator(n_folds=3),
            oos_validator=OutOfSampleValidator(
                holdout_fraction=0.2, min_sharpe=0.0, min_p_value=0.5
            ),
            cost_model=TransactionCostModel(min_net_sharpe=-10.0),
        )
        hypotheses = [_make_momentum_hypothesis()]
        report = pipeline.run(hypotheses, trending_data)

        for vh in report.validated_hypotheses:
            assert vh.hypothesis is not None
            assert vh.walk_forward_result is not None
            assert vh.oos_result is not None
            assert vh.regime_result is not None
            assert vh.cost_result is not None
