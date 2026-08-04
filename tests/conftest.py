"""Shared test fixtures providing synthetic OHLCV data with known properties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with 500 trading days.

    The data has known statistical properties:
    - Upward trending price (drift = 0.0005/day)
    - Realistic OHLCV relationships (High >= Open, Close, Low; Low <= all)
    - Positive volume with some variation
    - No missing values
    - DatetimeIndex with business day frequency

    Returns
    -------
    pd.DataFrame
        Synthetic OHLCV data with 500 rows.
    """
    np.random.seed(42)
    n_days = 500

    dates = pd.bdate_range(start="2020-01-02", periods=n_days, freq="B")

    # Generate realistic price path with drift and volatility
    drift = 0.0005
    volatility = 0.015
    log_returns = np.random.normal(drift, volatility, n_days)
    close_prices = 100 * np.exp(np.cumsum(log_returns))

    # Generate OHLCV from close
    # Open: close shifted + small overnight gap
    overnight_gaps = np.random.normal(0, 0.003, n_days)
    open_prices = np.roll(close_prices, 1) * (1 + overnight_gaps)
    open_prices[0] = 100.0  # First day

    # High: max of open/close + positive noise
    high_noise = np.abs(np.random.normal(0, 0.005, n_days))
    high_prices = np.maximum(open_prices, close_prices) * (1 + high_noise)

    # Low: min of open/close - positive noise
    low_noise = np.abs(np.random.normal(0, 0.005, n_days))
    low_prices = np.minimum(open_prices, close_prices) * (1 - low_noise)

    # Volume: positive with variation
    base_volume = 50_000_000
    volume = (base_volume * np.exp(np.random.normal(0, 0.3, n_days))).astype(int)

    df = pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volume,
        },
        index=dates,
    )

    return df


@pytest.fixture
def small_ohlcv() -> pd.DataFrame:
    """Generate a small OHLCV DataFrame (50 rows) for quick tests.

    Returns
    -------
    pd.DataFrame
        Small synthetic OHLCV data.
    """
    np.random.seed(123)
    n_days = 50
    dates = pd.bdate_range(start="2023-01-02", periods=n_days, freq="B")

    close = 100 + np.cumsum(np.random.normal(0, 1, n_days))
    open_ = close + np.random.normal(0, 0.5, n_days)
    high = np.maximum(open_, close) + np.abs(np.random.normal(0, 0.5, n_days))
    low = np.minimum(open_, close) - np.abs(np.random.normal(0, 0.5, n_days))
    volume = np.random.randint(1_000_000, 100_000_000, n_days)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def ohlcv_with_gaps() -> pd.DataFrame:
    """Generate OHLCV data with some missing values for testing validation.

    Returns
    -------
    pd.DataFrame
        OHLCV data with NaN values at known positions.
    """
    np.random.seed(99)
    n_days = 100
    dates = pd.bdate_range(start="2022-01-03", periods=n_days, freq="B")

    close = 200 + np.cumsum(np.random.normal(0, 2, n_days))
    open_ = close + np.random.normal(0, 1, n_days)
    high = np.maximum(open_, close) + np.abs(np.random.normal(0, 1, n_days))
    low = np.minimum(open_, close) - np.abs(np.random.normal(0, 1, n_days))
    volume = np.random.randint(10_000_000, 50_000_000, n_days).astype(float)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )

    # Introduce some NaN values
    df.loc[df.index[10], "Close"] = np.nan
    df.loc[df.index[11], "Volume"] = np.nan
    df.loc[df.index[50:53], "High"] = np.nan

    return df
