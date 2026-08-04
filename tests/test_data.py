"""Tests for data fetching, caching, and feature engineering modules."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_research.data.features import FeatureEngine
from quant_research.data.fetcher import DataFetcher


class TestDataFetcher:
    """Tests for the DataFetcher class."""

    def test_init_default_tickers(self) -> None:
        """DataFetcher initializes with default QQQ and ^IXIC tickers."""
        fetcher = DataFetcher()
        assert "QQQ" in fetcher.tickers
        assert "^IXIC" in fetcher.tickers

    def test_init_custom_tickers(self) -> None:
        """DataFetcher accepts custom ticker list."""
        fetcher = DataFetcher(tickers=["SPY", "AAPL"])
        assert fetcher.tickers == ["SPY", "AAPL"]

    def test_init_creates_cache_dir(self, tmp_path: Path) -> None:
        """DataFetcher creates cache directory on init."""
        cache_dir = tmp_path / "my_cache"
        fetcher = DataFetcher(cache_dir=cache_dir)
        assert cache_dir.exists()
        assert fetcher.cache_dir == cache_dir

    @patch("quant_research.data.fetcher.yf.download")
    def test_fetch_downloads_data(
        self, mock_download: MagicMock, sample_ohlcv: pd.DataFrame, tmp_path: Path
    ) -> None:
        """DataFetcher downloads data from yfinance when cache is empty."""
        mock_download.return_value = sample_ohlcv.copy()

        fetcher = DataFetcher(cache_dir=tmp_path / "cache")
        result = fetcher.fetch("QQQ")

        mock_download.assert_called_once()
        assert not result.empty
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    @patch("quant_research.data.fetcher.yf.download")
    def test_fetch_caches_to_parquet(
        self, mock_download: MagicMock, sample_ohlcv: pd.DataFrame, tmp_path: Path
    ) -> None:
        """DataFetcher caches downloaded data to parquet file."""
        mock_download.return_value = sample_ohlcv.copy()

        cache_dir = tmp_path / "cache"
        fetcher = DataFetcher(cache_dir=cache_dir)
        fetcher.fetch("QQQ")

        cache_file = cache_dir / "QQQ.parquet"
        assert cache_file.exists()

    @patch("quant_research.data.fetcher.yf.download")
    def test_fetch_uses_cache(
        self, mock_download: MagicMock, sample_ohlcv: pd.DataFrame, tmp_path: Path
    ) -> None:
        """DataFetcher reads from cache on second call without re-downloading."""
        mock_download.return_value = sample_ohlcv.copy()

        cache_dir = tmp_path / "cache"
        fetcher = DataFetcher(cache_dir=cache_dir)

        # First call downloads
        result1 = fetcher.fetch("QQQ")
        assert mock_download.call_count == 1

        # Second call uses cache
        result2 = fetcher.fetch("QQQ")
        assert mock_download.call_count == 1  # Not called again

        pd.testing.assert_frame_equal(result1, result2, check_freq=False)

    @patch("quant_research.data.fetcher.yf.download")
    def test_fetch_force_download_bypasses_cache(
        self, mock_download: MagicMock, sample_ohlcv: pd.DataFrame, tmp_path: Path
    ) -> None:
        """DataFetcher re-downloads when force_download=True."""
        mock_download.return_value = sample_ohlcv.copy()

        cache_dir = tmp_path / "cache"
        fetcher = DataFetcher(cache_dir=cache_dir)

        fetcher.fetch("QQQ")
        fetcher.fetch("QQQ", force_download=True)

        assert mock_download.call_count == 2

    @patch("quant_research.data.fetcher.yf.download")
    def test_fetch_raises_on_empty_data(
        self, mock_download: MagicMock, tmp_path: Path
    ) -> None:
        """DataFetcher raises ValueError when yfinance returns empty data."""
        mock_download.return_value = pd.DataFrame()

        fetcher = DataFetcher(cache_dir=tmp_path / "cache")
        with pytest.raises(ValueError, match="No data returned"):
            fetcher.fetch("INVALID_TICKER")

    @patch("quant_research.data.fetcher.yf.download")
    def test_fetch_handles_multiindex_columns(
        self, mock_download: MagicMock, sample_ohlcv: pd.DataFrame, tmp_path: Path
    ) -> None:
        """DataFetcher handles multi-level columns from yfinance."""
        # yfinance sometimes returns MultiIndex columns
        multi_idx = pd.MultiIndex.from_tuples(
            [(col, "QQQ") for col in sample_ohlcv.columns]
        )
        multi_df = sample_ohlcv.copy()
        multi_df.columns = multi_idx
        mock_download.return_value = multi_df

        fetcher = DataFetcher(cache_dir=tmp_path / "cache")
        result = fetcher.fetch("QQQ")

        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_train_test_split_default(self, sample_ohlcv: pd.DataFrame) -> None:
        """Train/test split uses 80/20 ratio by default."""
        fetcher = DataFetcher()
        train, test = fetcher.train_test_split(sample_ohlcv)

        expected_train_size = int(len(sample_ohlcv) * 0.8)
        assert len(train) == expected_train_size
        assert len(test) == len(sample_ohlcv) - expected_train_size

    def test_train_test_split_custom_ratio(self, sample_ohlcv: pd.DataFrame) -> None:
        """Train/test split respects custom ratio."""
        fetcher = DataFetcher()
        train, test = fetcher.train_test_split(sample_ohlcv, train_ratio=0.7)

        expected_train_size = int(len(sample_ohlcv) * 0.7)
        assert len(train) == expected_train_size
        assert len(test) == len(sample_ohlcv) - expected_train_size

    def test_train_test_split_chronological(self, sample_ohlcv: pd.DataFrame) -> None:
        """Train/test split maintains chronological order (no lookahead)."""
        fetcher = DataFetcher()
        train, test = fetcher.train_test_split(sample_ohlcv)

        assert train.index[-1] < test.index[0]

    def test_train_test_split_invalid_ratio(self, sample_ohlcv: pd.DataFrame) -> None:
        """Train/test split raises on invalid ratios."""
        fetcher = DataFetcher()
        with pytest.raises(ValueError):
            fetcher.train_test_split(sample_ohlcv, train_ratio=0.0)
        with pytest.raises(ValueError):
            fetcher.train_test_split(sample_ohlcv, train_ratio=1.0)
        with pytest.raises(ValueError):
            fetcher.train_test_split(sample_ohlcv, train_ratio=1.5)

    def test_train_test_split_empty_data(self) -> None:
        """Train/test split raises on empty DataFrame."""
        fetcher = DataFetcher()
        with pytest.raises(ValueError, match="Cannot split empty"):
            fetcher.train_test_split(pd.DataFrame())

    @patch("quant_research.data.fetcher.yf.download")
    def test_validate_handles_missing_values(
        self, mock_download: MagicMock, ohlcv_with_gaps: pd.DataFrame, tmp_path: Path
    ) -> None:
        """DataFetcher handles and cleans missing values in data."""
        mock_download.return_value = ohlcv_with_gaps.copy()

        fetcher = DataFetcher(cache_dir=tmp_path / "cache")
        result = fetcher.fetch("QQQ")

        # Should have no NaN after cleaning
        assert not result.isnull().any().any()

    @patch("quant_research.data.fetcher.yf.download")
    def test_cache_path_handles_special_chars(
        self, mock_download: MagicMock, sample_ohlcv: pd.DataFrame, tmp_path: Path
    ) -> None:
        """DataFetcher handles special characters in ticker symbols for caching."""
        mock_download.return_value = sample_ohlcv.copy()

        fetcher = DataFetcher(cache_dir=tmp_path / "cache")
        fetcher.fetch("^IXIC")

        # ^ is removed from filename
        cache_file = tmp_path / "cache" / "IXIC.parquet"
        assert cache_file.exists()


class TestFeatureEngine:
    """Tests for the FeatureEngine class."""

    def test_init_default_horizons(self) -> None:
        """FeatureEngine initializes with default return horizons."""
        engine = FeatureEngine()
        assert engine.return_horizons == [1, 5, 10, 20, 60, 120, 252]

    def test_init_custom_horizons(self) -> None:
        """FeatureEngine accepts custom return horizons."""
        engine = FeatureEngine(return_horizons=[1, 5, 20])
        assert engine.return_horizons == [1, 5, 20]

    def test_warmup_period(self) -> None:
        """Warmup period is based on max return horizon + buffer."""
        engine = FeatureEngine()
        assert engine.warmup_period >= 252

    def test_compute_returns(self, sample_ohlcv: pd.DataFrame) -> None:
        """Returns are computed at all specified horizons."""
        engine = FeatureEngine()
        returns = engine.compute_returns(sample_ohlcv)

        for horizon in engine.return_horizons:
            col = f"ret_{horizon}d"
            assert col in returns.columns
            # After warmup, values should be present
            assert returns[col].iloc[horizon:].notna().all()

    def test_compute_returns_no_lookahead(self, sample_ohlcv: pd.DataFrame) -> None:
        """Returns do not use future data (no lookahead bias).

        Test: modifying future data should not affect past returns.
        """
        engine = FeatureEngine()

        # Compute returns on original data
        returns_orig = engine.compute_returns(sample_ohlcv)

        # Modify the last 50 days of data
        modified = sample_ohlcv.copy()
        modified.iloc[-50:, modified.columns.get_loc("Close")] *= 2.0

        returns_modified = engine.compute_returns(modified)

        # Returns before the modification point should be identical
        # (accounting for the longest horizon lookback from modified area)
        safe_end = len(sample_ohlcv) - 50 - max(engine.return_horizons)
        if safe_end > 0:
            pd.testing.assert_frame_equal(
                returns_orig.iloc[:safe_end],
                returns_modified.iloc[:safe_end],
            )

    def test_compute_volatility(self, sample_ohlcv: pd.DataFrame) -> None:
        """Volatility features are computed correctly."""
        engine = FeatureEngine()
        vol = engine.compute_volatility(sample_ohlcv)

        # Check realized vol columns exist
        for window in [5, 10, 20, 60]:
            assert f"realized_vol_{window}d" in vol.columns

        # Check Parkinson columns
        for window in [10, 20, 60]:
            assert f"parkinson_vol_{window}d" in vol.columns

        # Check Garman-Klass columns
        for window in [10, 20, 60]:
            assert f"garman_klass_vol_{window}d" in vol.columns

        # Volatility should be non-negative after warmup
        warmup = 60
        for col in vol.columns:
            valid_data = vol[col].iloc[warmup:].dropna()
            assert (valid_data >= 0).all(), f"Negative volatility in {col}"

    def test_compute_volume_metrics(self, sample_ohlcv: pd.DataFrame) -> None:
        """Volume features are computed correctly."""
        engine = FeatureEngine()
        vol_metrics = engine.compute_volume_metrics(sample_ohlcv)

        # Check relative volume columns
        for window in [5, 10, 20]:
            assert f"relative_volume_{window}d" in vol_metrics.columns

        # Check volume ratio columns
        assert "volume_ratio_5_20" in vol_metrics.columns
        assert "volume_ratio_5_60" in vol_metrics.columns
        assert "volume_ratio_20_60" in vol_metrics.columns

        # Check VWAP proxy columns
        for window in [5, 10, 20]:
            assert f"vwap_proxy_{window}d" in vol_metrics.columns

        # Relative volume should be positive after warmup
        warmup = 60
        for col in vol_metrics.columns:
            if "relative_volume" in col or "volume_ratio" in col:
                valid_data = vol_metrics[col].iloc[warmup:].dropna()
                assert (valid_data > 0).all(), f"Non-positive value in {col}"

    def test_compute_gaps(self, sample_ohlcv: pd.DataFrame) -> None:
        """Gap features are computed correctly."""
        engine = FeatureEngine()
        gaps = engine.compute_gaps(sample_ohlcv)

        assert "overnight_gap" in gaps.columns
        assert "overnight_gap_abs" in gaps.columns
        assert "gap_up" in gaps.columns
        assert "gap_down" in gaps.columns

        # Gap abs should be non-negative
        valid = gaps["overnight_gap_abs"].dropna()
        assert (valid >= 0).all()

        # gap_up and gap_down should be binary
        assert set(gaps["gap_up"].dropna().unique()).issubset({0, 1})
        assert set(gaps["gap_down"].dropna().unique()).issubset({0, 1})

    def test_compute_session_indicators(self, sample_ohlcv: pd.DataFrame) -> None:
        """Session indicators are computed correctly."""
        engine = FeatureEngine()
        session = engine.compute_session_indicators(sample_ohlcv)

        assert "day_of_week" in session.columns
        assert "month" in session.columns
        assert "quarter" in session.columns
        assert "is_month_start" in session.columns
        assert "is_month_end" in session.columns
        assert "is_quarter_end" in session.columns
        assert "week_of_year" in session.columns

        # Day of week: 0-4 for business days
        assert session["day_of_week"].isin(range(5)).all()

        # Month: 1-12
        assert session["month"].isin(range(1, 13)).all()

        # Quarter: 1-4
        assert session["quarter"].isin(range(1, 5)).all()

        # Binary indicators
        assert set(session["is_month_start"].unique()).issubset({0, 1})
        assert set(session["is_month_end"].unique()).issubset({0, 1})
        assert set(session["is_quarter_end"].unique()).issubset({0, 1})

    def test_compute_technical_indicators(self, sample_ohlcv: pd.DataFrame) -> None:
        """Technical indicators are computed correctly."""
        engine = FeatureEngine()
        tech = engine.compute_technical_indicators(sample_ohlcv)

        # RSI
        assert "rsi_14" in tech.columns
        valid_rsi = tech["rsi_14"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

        # MACD
        assert "macd_line" in tech.columns
        assert "macd_signal" in tech.columns
        assert "macd_histogram" in tech.columns

        # Bollinger Bands
        assert "bb_percent_b" in tech.columns
        assert "bb_bandwidth" in tech.columns

        # ATR
        assert "atr_14" in tech.columns
        assert "atr_14_pct" in tech.columns
        valid_atr = tech["atr_14"].dropna()
        assert (valid_atr >= 0).all()

    def test_compute_all_no_nan_after_warmup(self, sample_ohlcv: pd.DataFrame) -> None:
        """No NaN values in features after warmup period."""
        engine = FeatureEngine()
        features = engine.compute_all(sample_ohlcv)

        warmup = engine.warmup_period
        after_warmup = features.iloc[warmup:]

        nan_cols = after_warmup.columns[after_warmup.isnull().any()]
        assert len(nan_cols) == 0, (
            f"NaN values found after warmup in columns: {list(nan_cols)}"
        )

    def test_compute_all_no_lookahead_bias(self, sample_ohlcv: pd.DataFrame) -> None:
        """Features do not use future data (no lookahead bias).

        Test: changing future data should not affect past features.
        """
        engine = FeatureEngine()

        features_orig = engine.compute_all(sample_ohlcv)

        # Modify the last 100 days significantly
        modified = sample_ohlcv.copy()
        modified.iloc[-100:] *= 3.0

        features_modified = engine.compute_all(modified)

        # Features before modification (with buffer for longest lookback)
        # should be the same
        safe_end = len(sample_ohlcv) - 100 - engine.warmup_period
        if safe_end > 0:
            orig_slice = features_orig.iloc[:safe_end]
            mod_slice = features_modified.iloc[:safe_end]

            # Compare with tolerance for floating point differences
            diff = (orig_slice - mod_slice).abs()
            max_diff = diff.max().max()
            assert max_diff < 1e-10, (
                f"Lookahead bias detected, max difference: {max_diff}"
            )

    def test_compute_all_output_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        """compute_all returns a DataFrame with same index length as input."""
        engine = FeatureEngine()
        features = engine.compute_all(sample_ohlcv)

        assert len(features) == len(sample_ohlcv)
        assert features.index.equals(sample_ohlcv.index)

    def test_rsi_bounds(self, sample_ohlcv: pd.DataFrame) -> None:
        """RSI values are bounded between 0 and 100."""
        engine = FeatureEngine()
        rsi = engine._compute_rsi(sample_ohlcv["Close"])
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_atr_non_negative(self, sample_ohlcv: pd.DataFrame) -> None:
        """ATR values are always non-negative."""
        engine = FeatureEngine()
        atr = engine._compute_atr(
            sample_ohlcv["High"],
            sample_ohlcv["Low"],
            sample_ohlcv["Close"],
        )
        valid = atr.dropna()
        assert (valid >= 0).all()
