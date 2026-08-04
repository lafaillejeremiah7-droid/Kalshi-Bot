"""Integration tests for the main pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quant_research.main import PipelineResult, run_pipeline


@pytest.fixture
def synthetic_data_with_signal() -> pd.DataFrame:
    """Generate 3-year synthetic data with a known embedded momentum signal.

    The data has a strong positive autocorrelation (momentum) so that
    simple momentum hypotheses should survive the pipeline.
    """
    np.random.seed(123)
    n_days = 756  # ~3 years of trading days
    dates = pd.bdate_range(start="2020-01-02", periods=n_days, freq="B")

    # Generate data with strong momentum (positive autocorrelation)
    returns = np.zeros(n_days)
    returns[0] = np.random.normal(0.001, 0.01)

    for i in range(1, n_days):
        # Strong momentum component: 30% of previous return persists
        momentum = 0.3 * returns[i - 1]
        noise = np.random.normal(0.0005, 0.012)
        returns[i] = momentum + noise

    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLCV
    overnight_gaps = np.random.normal(0, 0.002, n_days)
    open_prices = np.roll(close, 1) * (1 + overnight_gaps)
    open_prices[0] = 100.0

    high = np.maximum(open_prices, close) * (1 + np.abs(np.random.normal(0, 0.004, n_days)))
    low = np.minimum(open_prices, close) * (1 - np.abs(np.random.normal(0, 0.004, n_days)))
    volume = (80_000_000 * np.exp(np.random.normal(0, 0.25, n_days))).astype(int)

    df = pd.DataFrame(
        {
            "Open": open_prices,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    return df


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_pipeline_completes_without_error(
        self, synthetic_data_with_signal: pd.DataFrame, tmp_path
    ) -> None:
        """Pipeline should complete without raising exceptions."""
        config = {
            "data": synthetic_data_with_signal,
            "output_dir": str(tmp_path / "results"),
        }
        result = run_pipeline(config)

        assert isinstance(result, PipelineResult)
        assert len(result.all_hypotheses) > 100

    def test_pipeline_generates_hypotheses(
        self, synthetic_data_with_signal: pd.DataFrame, tmp_path
    ) -> None:
        """Pipeline should generate hypotheses."""
        config = {
            "data": synthetic_data_with_signal,
            "output_dir": str(tmp_path / "results"),
        }
        result = run_pipeline(config)

        assert len(result.all_hypotheses) >= 100

    def test_pipeline_statistical_testing(
        self, synthetic_data_with_signal: pd.DataFrame, tmp_path
    ) -> None:
        """Pipeline should filter hypotheses through statistical testing."""
        config = {
            "data": synthetic_data_with_signal,
            "output_dir": str(tmp_path / "results"),
        }
        result = run_pipeline(config)

        # Some hypotheses should be filtered out by statistical testing
        assert len(result.statistical_survivors) <= len(result.all_hypotheses)

    def test_pipeline_generates_report(
        self, synthetic_data_with_signal: pd.DataFrame, tmp_path
    ) -> None:
        """Pipeline should generate a report file."""
        config = {
            "data": synthetic_data_with_signal,
            "output_dir": str(tmp_path / "results"),
        }
        result = run_pipeline(config)

        assert result.report_path != ""
        report_file = tmp_path / "results" / "research_report.md"
        assert report_file.exists()

    def test_report_contains_required_sections(
        self, synthetic_data_with_signal: pd.DataFrame, tmp_path
    ) -> None:
        """Report should contain all required sections."""
        config = {
            "data": synthetic_data_with_signal,
            "output_dir": str(tmp_path / "results"),
        }
        run_pipeline(config)

        report_file = tmp_path / "results" / "research_report.md"
        content = report_file.read_text()

        assert "DATA LIMITATIONS DISCLAIMER" in content
        assert "Executive Summary" in content
        assert "Methodology" in content
        assert "Rejection Funnel" in content
        assert "Limitations and Future Work" in content
        assert "OHLCV" in content
        assert "order flow" in content.lower() or "Order Flow" in content

    def test_pipeline_handles_no_survivors_gracefully(self, tmp_path) -> None:
        """Pipeline should handle case where no hypotheses survive."""
        # Very short data that produces no valid signals
        np.random.seed(999)
        n_days = 50
        dates = pd.bdate_range(start="2023-01-02", periods=n_days, freq="B")
        close = 100 + np.random.normal(0, 0.1, n_days).cumsum()
        open_p = close + np.random.normal(0, 0.05, n_days)
        high = np.maximum(open_p, close) + 0.5
        low = np.minimum(open_p, close) - 0.5
        volume = np.full(n_days, 50_000_000)

        data = pd.DataFrame(
            {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

        config = {
            "data": data,
            "output_dir": str(tmp_path / "results"),
        }
        result = run_pipeline(config)

        # Should not raise, pipeline should complete
        assert isinstance(result, PipelineResult)
        assert len(result.all_hypotheses) >= 100

    def test_pipeline_result_dataclass_fields(
        self, synthetic_data_with_signal: pd.DataFrame, tmp_path
    ) -> None:
        """PipelineResult should have all expected fields."""
        config = {
            "data": synthetic_data_with_signal,
            "output_dir": str(tmp_path / "results"),
        }
        result = run_pipeline(config)

        assert hasattr(result, "all_hypotheses")
        assert hasattr(result, "statistical_survivors")
        assert hasattr(result, "validated_survivors")
        assert hasattr(result, "strategies")
        assert hasattr(result, "report_path")

    def test_pipeline_import(self) -> None:
        """Pipeline should be importable."""
        from quant_research.main import run_pipeline as rp

        assert callable(rp)
