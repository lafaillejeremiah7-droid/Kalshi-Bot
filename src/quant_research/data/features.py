"""
Feature engineering module for deriving quantitative signals from OHLCV data.

This module computes a comprehensive set of features from daily OHLCV bars.
All features are computed WITHOUT lookahead bias - each feature at time t
uses only data available at or before time t.

Data Limitations:
    All features are derived from OHLCV (Open, High, Low, Close, Volume) only.
    This means:
    - Volume-based features use exchange-reported volume, which may not capture
      all trading activity (dark pools, OTC, etc.)
    - VWAP is approximated using typical price * volume (not true tick-level VWAP)
    - No order flow, bid-ask spread, or market depth signals are available
    - Overnight gaps are inferred from Close->Open, not from actual extended-hours
      trading data

    These limitations should be considered when interpreting feature importance
    and strategy signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngine:
    """Computes derived features from OHLCV data without lookahead bias.

    All features are computed using only past and current data at each point.
    The warmup period is determined by the longest lookback window used
    (default: 252 days for annual return calculations).

    Feature Categories:
        - Returns: Multi-horizon log returns (1, 5, 10, 20, 60, 120, 252 days)
        - Volatility: Realized vol, Parkinson, Garman-Klass estimators
        - Volume: Relative volume, MA ratios, VWAP proxy
        - Gaps: Overnight gap measurements
        - Session: Calendar-based indicators
        - Technical: RSI, MACD, Bollinger Bands, ATR

    Note on OHLCV limitations:
        Features like VWAP are proxied from typical price * volume, not
        computed from actual intraday tick data. Volume metrics reflect
        only exchange-reported daily volume.

    Parameters
    ----------
    return_horizons : list[int], optional
        Horizons in days for return computation.
        Defaults to [1, 5, 10, 20, 60, 120, 252].

    Examples
    --------
    >>> engine = FeatureEngine()
    >>> features = engine.compute_all(ohlcv_df)
    """

    def __init__(
        self,
        return_horizons: list[int] | None = None,
    ) -> None:
        self.return_horizons = return_horizons or [1, 5, 10, 20, 60, 120, 252]

    @property
    def warmup_period(self) -> int:
        """Minimum number of rows needed before features are fully computed.

        Returns
        -------
        int
            The warmup period in trading days.
        """
        # Max of all lookback windows used across feature categories
        return max(self.return_horizons) + 20  # Extra buffer for rolling windows

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all features from OHLCV data.

        Computes returns, volatility, volume, gap, session, and technical
        features. All features use only past/current data (no lookahead).

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame with columns: Open, High, Low, Close, Volume.
            Must have a DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            DataFrame with all computed features. Same index as input.
            Rows within the warmup period will contain NaN values.
        """
        features = pd.DataFrame(index=data.index)

        # Compute each feature category
        features = features.join(self.compute_returns(data))
        features = features.join(self.compute_volatility(data))
        features = features.join(self.compute_volume_metrics(data))
        features = features.join(self.compute_gaps(data))
        features = features.join(self.compute_session_indicators(data))
        features = features.join(self.compute_technical_indicators(data))

        return features

    def compute_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns at multiple horizons.

        Uses log returns: ln(price_t / price_{t-horizon}).
        Each return at time t uses only the close at t and t-horizon (no future data).

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame with 'Close' column.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns like 'ret_1d', 'ret_5d', etc.
        """
        results = pd.DataFrame(index=data.index)
        close = data["Close"]

        for horizon in self.return_horizons:
            results[f"ret_{horizon}d"] = np.log(close / close.shift(horizon))

        return results

    def compute_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility measures using multiple estimators.

        Estimators:
        - Realized volatility: rolling std of log returns
        - Parkinson estimator: uses High-Low range
        - Garman-Klass estimator: uses OHLC data for efficiency

        All use backward-looking rolling windows only (no lookahead).

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame with Open, High, Low, Close columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with volatility features.
        """
        results = pd.DataFrame(index=data.index)
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        open_ = data["Open"]

        log_ret = np.log(close / close.shift(1))

        # Realized volatility at different windows
        for window in [5, 10, 20, 60]:
            results[f"realized_vol_{window}d"] = log_ret.rolling(window).std() * np.sqrt(252)

        # Parkinson estimator (uses High-Low range)
        # Var = (1 / (4 * ln(2))) * (ln(H/L))^2
        log_hl = np.log(high / low)
        parkinson_var = (log_hl**2) / (4 * np.log(2))
        for window in [10, 20, 60]:
            results[f"parkinson_vol_{window}d"] = (
                np.sqrt(parkinson_var.rolling(window).mean() * 252)
            )

        # Garman-Klass estimator (uses OHLC)
        # Var = 0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2
        log_co = np.log(close / open_)
        gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        for window in [10, 20, 60]:
            results[f"garman_klass_vol_{window}d"] = (
                np.sqrt(gk_var.rolling(window).mean().clip(lower=0) * 252)
            )

        return results

    def compute_volume_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based features.

        Note: Volume data is exchange-reported daily volume only.
        True VWAP requires tick data; here we approximate using typical price.

        Features:
        - Relative volume: current volume vs rolling average
        - Volume MA ratios: short-term vs long-term volume averages
        - VWAP proxy: cumulative (typical price * volume) / cumulative volume

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame with High, Low, Close, Volume columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with volume-based features.
        """
        results = pd.DataFrame(index=data.index)
        volume = data["Volume"].astype(float)
        typical_price = (data["High"] + data["Low"] + data["Close"]) / 3

        # Relative volume (current vs rolling averages)
        for window in [5, 10, 20]:
            vol_ma = volume.rolling(window).mean()
            results[f"relative_volume_{window}d"] = volume / vol_ma

        # Volume MA ratios (short vs long)
        vol_5 = volume.rolling(5).mean()
        vol_20 = volume.rolling(20).mean()
        vol_60 = volume.rolling(60).mean()
        results["volume_ratio_5_20"] = vol_5 / vol_20
        results["volume_ratio_5_60"] = vol_5 / vol_60
        results["volume_ratio_20_60"] = vol_20 / vol_60

        # VWAP proxy using rolling window (not cumulative from session start)
        # Since we have daily data, use rolling period VWAP approximation
        for window in [5, 10, 20]:
            tp_vol = (typical_price * volume).rolling(window).sum()
            vol_sum = volume.rolling(window).sum()
            vwap = tp_vol / vol_sum
            results[f"vwap_proxy_{window}d"] = data["Close"] / vwap

        return results

    def compute_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute overnight gap measurements.

        Gaps are measured as the log ratio of today's open to yesterday's close.
        This represents the overnight return that occurs between sessions.

        Note: Actual overnight trading may occur (extended hours), but we
        only observe the Open price at market open vs previous Close.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame with Open and Close columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with gap features.
        """
        results = pd.DataFrame(index=data.index)

        # Overnight gap: log(Open_t / Close_{t-1})
        prev_close = data["Close"].shift(1)
        gap = np.log(data["Open"] / prev_close)

        results["overnight_gap"] = gap
        results["overnight_gap_abs"] = gap.abs()

        # Rolling statistics of gaps
        for window in [5, 10, 20]:
            results[f"gap_mean_{window}d"] = gap.rolling(window).mean()
            results[f"gap_std_{window}d"] = gap.rolling(window).std()

        # Gap direction indicators
        results["gap_up"] = (gap > 0).astype(int)
        results["gap_down"] = (gap < 0).astype(int)

        return results

    def compute_session_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute calendar/session-based indicators.

        These are deterministic features based on the date and do not
        involve any price data, so lookahead bias is not a concern.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            DataFrame with session indicator features.
        """
        results = pd.DataFrame(index=data.index)
        idx = data.index

        # Day of week (0=Monday, 4=Friday)
        results["day_of_week"] = idx.dayofweek

        # Month (1-12)
        results["month"] = idx.month

        # Quarter (1-4)
        results["quarter"] = idx.quarter

        # Binary indicators
        results["is_month_start"] = idx.is_month_start.astype(int)
        results["is_month_end"] = idx.is_month_end.astype(int)
        results["is_quarter_end"] = idx.is_quarter_end.astype(int)

        # Week of year
        results["week_of_year"] = idx.isocalendar().week.astype(int).values

        return results

    def compute_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute technical analysis indicators.

        All indicators use only past/current data (no lookahead).

        Indicators:
        - RSI (14-day): Relative Strength Index
        - MACD (12/26/9): Moving Average Convergence Divergence
        - Bollinger Bands (20-day, 2 std): Bandwidth and %B
        - ATR (14-day): Average True Range

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame with High, Low, Close columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with technical indicator features.
        """
        results = pd.DataFrame(index=data.index)
        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        # RSI (14-day)
        results["rsi_14"] = self._compute_rsi(close, period=14)

        # MACD (12, 26, 9)
        macd_line, signal_line, histogram = self._compute_macd(close)
        results["macd_line"] = macd_line
        results["macd_signal"] = signal_line
        results["macd_histogram"] = histogram

        # Bollinger Bands (20-day, 2 std)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # %B: position within bands (0 = lower, 1 = upper)
        results["bb_percent_b"] = (close - bb_lower) / (bb_upper - bb_lower)
        # Bandwidth: width of bands relative to middle
        results["bb_bandwidth"] = (bb_upper - bb_lower) / bb_mid

        # ATR (14-day)
        results["atr_14"] = self._compute_atr(high, low, close, period=14)
        # Normalized ATR (as percentage of close)
        results["atr_14_pct"] = results["atr_14"] / close

        return results

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index.

        Parameters
        ----------
        close : pd.Series
            Close price series.
        period : int
            RSI lookback period.

        Returns
        -------
        pd.Series
            RSI values (0-100).
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _compute_macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Compute MACD indicator.

        Parameters
        ----------
        close : pd.Series
            Close price series.
        fast : int
            Fast EMA period.
        slow : int
            Slow EMA period.
        signal : int
            Signal line EMA period.

        Returns
        -------
        tuple[pd.Series, pd.Series, pd.Series]
            (macd_line, signal_line, histogram)
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _compute_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Compute Average True Range.

        Parameters
        ----------
        high : pd.Series
            High price series.
        low : pd.Series
            Low price series.
        close : pd.Series
            Close price series.
        period : int
            ATR lookback period.

        Returns
        -------
        pd.Series
            ATR values.
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        return atr
