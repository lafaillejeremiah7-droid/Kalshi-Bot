"""
Signal computation functions for market behavior hypotheses.

Each function takes a DataFrame with OHLCV+features columns (as produced by
FeatureEngine.compute_all()) and returns a pandas Series of signal values.

Signal values are either:
- Discrete: {-1, 0, 1} where 1=long, -1=short, 0=no position
- Continuous: [-1, 1] representing signal strength and direction

Data Limitations:
    All signals are derived from OHLCV (Open, High, Low, Close, Volume) data only.
    No order book, Level II, time-and-sales, or intraday tick data is used.
    Volume-based signals use exchange-reported daily volume only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# MOMENTUM SIGNALS
# =============================================================================


def momentum_1d(df: pd.DataFrame) -> pd.Series:
    """1-day price momentum signal."""
    ret = np.log(df["Close"] / df["Close"].shift(1))
    return np.sign(ret)


def momentum_5d(df: pd.DataFrame) -> pd.Series:
    """5-day price momentum signal."""
    ret = np.log(df["Close"] / df["Close"].shift(5))
    return np.sign(ret)


def momentum_10d(df: pd.DataFrame) -> pd.Series:
    """10-day price momentum signal."""
    ret = np.log(df["Close"] / df["Close"].shift(10))
    return np.sign(ret)


def momentum_20d(df: pd.DataFrame) -> pd.Series:
    """20-day price momentum signal."""
    ret = np.log(df["Close"] / df["Close"].shift(20))
    return np.sign(ret)


def momentum_60d(df: pd.DataFrame) -> pd.Series:
    """60-day price momentum signal."""
    ret = np.log(df["Close"] / df["Close"].shift(60))
    return np.sign(ret)


def momentum_120d(df: pd.DataFrame) -> pd.Series:
    """120-day price momentum signal."""
    ret = np.log(df["Close"] / df["Close"].shift(120))
    return np.sign(ret)


def momentum_252d(df: pd.DataFrame) -> pd.Series:
    """252-day (annual) price momentum signal."""
    ret = np.log(df["Close"] / df["Close"].shift(252))
    return np.sign(ret)


def dual_momentum(df: pd.DataFrame) -> pd.Series:
    """Dual momentum: absolute + relative (vs short-term).
    Long if both 12-month return > 0 and 1-month return > 0."""
    ret_252 = np.log(df["Close"] / df["Close"].shift(252))
    ret_20 = np.log(df["Close"] / df["Close"].shift(20))
    signal = pd.Series(0.0, index=df.index)
    signal[(ret_252 > 0) & (ret_20 > 0)] = 1.0
    signal[(ret_252 < 0) & (ret_20 < 0)] = -1.0
    return signal


def momentum_vol_scaled(df: pd.DataFrame) -> pd.Series:
    """Momentum scaled by inverse volatility (risk-parity momentum)."""
    ret_20 = np.log(df["Close"] / df["Close"].shift(20))
    vol_20 = ret_20.rolling(20).std()
    raw = ret_20 / vol_20.replace(0, np.nan)
    return raw.clip(-1, 1).fillna(0)


def rate_of_change(df: pd.DataFrame) -> pd.Series:
    """Rate of change (10-day) normalized to [-1, 1]."""
    roc = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)
    return roc.clip(-1, 1).fillna(0)


def momentum_reversal_extreme(df: pd.DataFrame) -> pd.Series:
    """Momentum reversal after extreme moves (>2 std 5-day return)."""
    ret_5 = np.log(df["Close"] / df["Close"].shift(5))
    std_5 = ret_5.rolling(60).std()
    z = ret_5 / std_5.replace(0, np.nan)
    signal = pd.Series(0.0, index=df.index)
    signal[z > 2] = -1.0  # Expect reversal after extreme up
    signal[z < -2] = 1.0  # Expect reversal after extreme down
    return signal


def momentum_acceleration(df: pd.DataFrame) -> pd.Series:
    """Acceleration: momentum of momentum (second derivative)."""
    ret_20 = np.log(df["Close"] / df["Close"].shift(20))
    accel = ret_20 - ret_20.shift(20)
    return np.sign(accel).fillna(0)


def trend_following_ma_cross(df: pd.DataFrame) -> pd.Series:
    """Trend following: price vs 50-day MA crossover."""
    ma_50 = df["Close"].rolling(50).mean()
    signal = pd.Series(0.0, index=df.index)
    signal[df["Close"] > ma_50] = 1.0
    signal[df["Close"] < ma_50] = -1.0
    return signal


def dual_ma_crossover(df: pd.DataFrame) -> pd.Series:
    """Dual MA crossover: 20-day MA vs 50-day MA."""
    ma_20 = df["Close"].rolling(20).mean()
    ma_50 = df["Close"].rolling(50).mean()
    signal = pd.Series(0.0, index=df.index)
    signal[ma_20 > ma_50] = 1.0
    signal[ma_20 < ma_50] = -1.0
    return signal


def momentum_crash_vol_filter(df: pd.DataFrame) -> pd.Series:
    """Momentum with vol filter: long momentum in low vol, reduce in high vol."""
    ret_60 = np.log(df["Close"] / df["Close"].shift(60))
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
    vol_median = vol_20.rolling(252).median()
    signal = np.sign(ret_60).fillna(0)
    # Reduce signal in high vol regime
    high_vol = vol_20 > vol_median * 1.5
    signal[high_vol] = signal[high_vol] * 0.5
    return signal.clip(-1, 1)




# =============================================================================
# MEAN REVERSION SIGNALS
# =============================================================================


def rsi_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """RSI extremes mean reversion: buy oversold (<30), sell overbought (>70)."""
    if "rsi_14" in df.columns:
        rsi = df["rsi_14"]
    else:
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    signal = pd.Series(0.0, index=df.index)
    signal[rsi < 30] = 1.0
    signal[rsi > 70] = -1.0
    return signal


def bollinger_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """Bollinger Band mean reversion: buy at lower band, sell at upper."""
    if "bb_percent_b" in df.columns:
        pct_b = df["bb_percent_b"]
    else:
        ma = df["Close"].rolling(20).mean()
        std = df["Close"].rolling(20).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        pct_b = (df["Close"] - lower) / (upper - lower)
    signal = pd.Series(0.0, index=df.index)
    signal[pct_b < 0.0] = 1.0
    signal[pct_b > 1.0] = -1.0
    signal[(pct_b >= 0.0) & (pct_b < 0.2)] = 0.5
    signal[(pct_b > 0.8) & (pct_b <= 1.0)] = -0.5
    return signal


def gap_fill_reversion(df: pd.DataFrame) -> pd.Series:
    """Gap fill tendency: fade overnight gaps expecting fill."""
    if "overnight_gap" in df.columns:
        gap = df["overnight_gap"]
    else:
        gap = np.log(df["Open"] / df["Close"].shift(1))
    signal = pd.Series(0.0, index=df.index)
    gap_std = gap.rolling(20).std()
    z = gap / gap_std.replace(0, np.nan)
    signal[z > 1] = -1.0  # Large up gap -> expect fill (short)
    signal[z < -1] = 1.0  # Large down gap -> expect fill (long)
    return signal


def consecutive_move_reversion(df: pd.DataFrame) -> pd.Series:
    """Multi-day consecutive move reversion (3+ days same direction)."""
    ret = np.log(df["Close"] / df["Close"].shift(1))
    up = (ret > 0).astype(int)
    down = (ret < 0).astype(int)
    consec_up = up.groupby((up != up.shift()).cumsum()).cumsum()
    consec_down = down.groupby((down != down.shift()).cumsum()).cumsum()
    signal = pd.Series(0.0, index=df.index)
    signal[consec_up >= 3] = -1.0  # 3+ up days -> expect reversal
    signal[consec_down >= 3] = 1.0  # 3+ down days -> expect reversal
    return signal


def volatility_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """Volatility mean reversion: high vol reverts to mean."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
    vol_60 = log_ret.rolling(60).std() * np.sqrt(252)
    ratio = vol_20 / vol_60.replace(0, np.nan)
    signal = pd.Series(0.0, index=df.index)
    # High vol -> expect contraction (neutral/long bias)
    signal[ratio > 1.5] = 1.0
    # Low vol -> expect expansion (cautious)
    signal[ratio < 0.5] = -1.0
    return signal


def volume_weighted_reversion(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted reversion: high-volume extreme moves revert faster."""
    ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_ratio = df["Volume"] / df["Volume"].rolling(20).mean()
    # High volume + extreme return -> stronger reversion signal
    signal = -ret * vol_ratio
    return signal.clip(-1, 1).fillna(0)


def distance_from_ma_reversion(df: pd.DataFrame) -> pd.Series:
    """Distance from 50-day MA reversion."""
    ma_50 = df["Close"].rolling(50).mean()
    dist = (df["Close"] - ma_50) / ma_50
    std_dist = dist.rolling(60).std()
    z = dist / std_dist.replace(0, np.nan)
    signal = pd.Series(0.0, index=df.index)
    signal[z > 2] = -1.0
    signal[z < -2] = 1.0
    signal[(z > 1) & (z <= 2)] = -0.5
    signal[(z < -1) & (z >= -2)] = 0.5
    return signal


def zscore_returns_reversion(df: pd.DataFrame) -> pd.Series:
    """Z-score of 5-day returns: fade extreme z-scores."""
    ret_5 = np.log(df["Close"] / df["Close"].shift(5))
    z = (ret_5 - ret_5.rolling(60).mean()) / ret_5.rolling(60).std()
    signal = pd.Series(0.0, index=df.index)
    signal[z > 2] = -1.0
    signal[z < -2] = 1.0
    return signal


def hurst_regime_reversion(df: pd.DataFrame) -> pd.Series:
    """Hurst exponent proxy: low Hurst (<0.5) suggests mean reversion."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    # Simplified Hurst proxy using rescaled range
    window = 60
    rolling_std = log_ret.rolling(window).std()
    rolling_range = (
        df["Close"].rolling(window).max() - df["Close"].rolling(window).min()
    )
    # Hurst proxy: if range grows slower than sqrt(n), mean reverting
    hurst_proxy = np.log(rolling_range / rolling_std.replace(0, np.nan)) / np.log(window)
    signal = pd.Series(0.0, index=df.index)
    # Mean reverting regime -> trade reversion
    signal[hurst_proxy < 0.4] = np.sign(-log_ret[hurst_proxy < 0.4])
    return signal.fillna(0).clip(-1, 1)


def put_call_proxy_reversion(df: pd.DataFrame) -> pd.Series:
    """Put-call proxy from volatility skew: high fear -> contrarian long."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_5 = log_ret.rolling(5).std() * np.sqrt(252)
    vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
    # Short-term vol spike vs longer-term = fear proxy
    fear = vol_5 / vol_20.replace(0, np.nan)
    signal = pd.Series(0.0, index=df.index)
    signal[fear > 1.5] = 1.0  # High fear -> contrarian long
    signal[fear < 0.6] = -1.0  # Low fear/complacency -> cautious
    return signal


def rsi_divergence_reversion(df: pd.DataFrame) -> pd.Series:
    """RSI divergence: price makes new low but RSI does not."""
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    price_low_20 = df["Close"].rolling(20).min()
    rsi_low_20 = rsi.rolling(20).min()
    # Bullish divergence: price at 20d low but RSI is not
    signal = pd.Series(0.0, index=df.index)
    at_price_low = df["Close"] <= price_low_20 * 1.01
    rsi_above_low = rsi > rsi_low_20 * 1.1
    signal[at_price_low & rsi_above_low] = 1.0
    # Bearish divergence
    price_high_20 = df["Close"].rolling(20).max()
    rsi_high_20 = rsi.rolling(20).max()
    at_price_high = df["Close"] >= price_high_20 * 0.99
    rsi_below_high = rsi < rsi_high_20 * 0.9
    signal[at_price_high & rsi_below_high] = -1.0
    return signal


def overextension_reversion(df: pd.DataFrame) -> pd.Series:
    """Overextension: price far from 200-day MA reverts."""
    ma_200 = df["Close"].rolling(200).mean()
    pct_away = (df["Close"] - ma_200) / ma_200
    signal = pd.Series(0.0, index=df.index)
    signal[pct_away > 0.15] = -1.0  # >15% above -> short
    signal[pct_away < -0.15] = 1.0  # >15% below -> long
    signal[(pct_away > 0.08) & (pct_away <= 0.15)] = -0.5
    signal[(pct_away < -0.08) & (pct_away >= -0.15)] = 0.5
    return signal


def macd_reversion(df: pd.DataFrame) -> pd.Series:
    """MACD histogram reversion at extremes."""
    if "macd_histogram" in df.columns:
        hist = df["macd_histogram"]
    else:
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal_line
    hist_std = hist.rolling(60).std()
    z = hist / hist_std.replace(0, np.nan)
    signal = pd.Series(0.0, index=df.index)
    signal[z > 2] = -1.0
    signal[z < -2] = 1.0
    return signal


def keltner_reversion(df: pd.DataFrame) -> pd.Series:
    """Keltner channel reversion: price outside channel reverts."""
    ema_20 = df["Close"].ewm(span=20, adjust=False).mean()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean()
    upper = ema_20 + 2 * atr
    lower = ema_20 - 2 * atr
    signal = pd.Series(0.0, index=df.index)
    signal[df["Close"] > upper] = -1.0
    signal[df["Close"] < lower] = 1.0
    return signal




# =============================================================================
# VOLATILITY SIGNALS
# =============================================================================


def vol_expansion_breakout(df: pd.DataFrame) -> pd.Series:
    """Volatility expansion breakout: trade in direction of vol expansion."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_5 = log_ret.rolling(5).std()
    vol_20 = log_ret.rolling(20).std()
    expansion = vol_5 > vol_20 * 1.5
    signal = pd.Series(0.0, index=df.index)
    signal[expansion & (log_ret > 0)] = 1.0
    signal[expansion & (log_ret < 0)] = -1.0
    return signal


def vol_compression_squeeze(df: pd.DataFrame) -> pd.Series:
    """Volatility compression (squeeze): low vol precedes big moves."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_10 = log_ret.rolling(10).std()
    vol_60 = log_ret.rolling(60).std()
    squeeze = vol_10 < vol_60 * 0.5
    # Direction from recent momentum
    ret_5 = np.log(df["Close"] / df["Close"].shift(5))
    signal = pd.Series(0.0, index=df.index)
    signal[squeeze & (ret_5 > 0)] = 1.0
    signal[squeeze & (ret_5 < 0)] = -1.0
    return signal


def vol_clustering(df: pd.DataFrame) -> pd.Series:
    """Volatility clustering: high vol follows high vol (persist direction)."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_5 = log_ret.rolling(5).std() * np.sqrt(252)
    vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
    high_vol = vol_5 > vol_20 * 1.2
    ret_5 = np.log(df["Close"] / df["Close"].shift(5))
    signal = pd.Series(0.0, index=df.index)
    signal[high_vol] = np.sign(ret_5[high_vol])
    return signal


def vix_analog_signal(df: pd.DataFrame) -> pd.Series:
    """VIX analog: use realized vol as fear proxy. High fear -> contrarian long."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_10 = log_ret.rolling(10).std() * np.sqrt(252)
    vol_percentile = vol_10.rolling(252).rank(pct=True)
    signal = pd.Series(0.0, index=df.index)
    signal[vol_percentile > 0.9] = 1.0   # Extreme fear -> long
    signal[vol_percentile < 0.1] = -1.0  # Complacency -> cautious
    return signal


def vol_term_structure_proxy(df: pd.DataFrame) -> pd.Series:
    """Vol term structure proxy: short-term vs long-term vol ratio."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_5 = log_ret.rolling(5).std() * np.sqrt(252)
    vol_60 = log_ret.rolling(60).std() * np.sqrt(252)
    ratio = vol_5 / vol_60.replace(0, np.nan)
    signal = pd.Series(0.0, index=df.index)
    # Inverted term structure (short > long) -> expect mean reversion
    signal[ratio > 1.3] = 1.0
    signal[ratio < 0.7] = -1.0
    return signal


def vol_regime_persistence(df: pd.DataFrame) -> pd.Series:
    """Volatility regime persistence: stay with current vol regime."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
    vol_ma = vol_20.rolling(60).mean()
    high_vol = vol_20 > vol_ma
    signal = pd.Series(0.0, index=df.index)
    # In high vol regime, trend following works better
    ret_10 = np.log(df["Close"] / df["Close"].shift(10))
    signal[high_vol] = np.sign(ret_10[high_vol])
    # In low vol regime, mean reversion
    signal[~high_vol] = -np.sign(ret_10[~high_vol])
    return signal.fillna(0)


def garch_conditional_vol(df: pd.DataFrame) -> pd.Series:
    """GARCH-inspired conditional vol: weighted avg of recent squared returns."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    # Simple GARCH(1,1) approximation
    omega = 0.00001
    alpha = 0.1
    beta = 0.85
    var = pd.Series(0.0, index=df.index, dtype=float)
    var.iloc[0] = log_ret.iloc[:20].var() if len(log_ret) > 20 else 0.0001
    for i in range(1, len(var)):
        var.iloc[i] = omega + alpha * log_ret.iloc[i-1]**2 + beta * var.iloc[i-1]
    cond_vol = np.sqrt(var) * np.sqrt(252)
    vol_percentile = cond_vol.rolling(252, min_periods=60).rank(pct=True)
    signal = pd.Series(0.0, index=df.index)
    signal[vol_percentile > 0.8] = 1.0  # High conditional vol -> mean reversion
    signal[vol_percentile < 0.2] = -1.0  # Low vol -> breakout expected
    return signal


def vol_of_vol_signal(df: pd.DataFrame) -> pd.Series:
    """Vol-of-vol: volatility of volatility indicates regime uncertainty."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_5 = log_ret.rolling(5).std() * np.sqrt(252)
    vol_of_vol = vol_5.rolling(20).std()
    vov_percentile = vol_of_vol.rolling(252, min_periods=60).rank(pct=True)
    signal = pd.Series(0.0, index=df.index)
    signal[vov_percentile > 0.8] = 1.0  # High uncertainty -> contrarian
    signal[vov_percentile < 0.2] = -1.0
    return signal


def range_expansion_signal(df: pd.DataFrame) -> pd.Series:
    """Range expansion: unusually wide daily range signals continuation."""
    daily_range = (df["High"] - df["Low"]) / df["Close"]
    avg_range = daily_range.rolling(20).mean()
    expansion = daily_range > avg_range * 2
    ret = np.log(df["Close"] / df["Open"])
    signal = pd.Series(0.0, index=df.index)
    signal[expansion & (ret > 0)] = 1.0
    signal[expansion & (ret < 0)] = -1.0
    return signal


def vol_smile_proxy(df: pd.DataFrame) -> pd.Series:
    """Vol smile proxy from high-low: asymmetric ranges suggest directional bias."""
    upper_wick = df["High"] - np.maximum(df["Open"], df["Close"])
    lower_wick = np.minimum(df["Open"], df["Close"]) - df["Low"]
    total_range = df["High"] - df["Low"]
    # Rolling average of wick asymmetry
    wick_ratio = (upper_wick - lower_wick) / total_range.replace(0, np.nan)
    avg_ratio = wick_ratio.rolling(10).mean()
    signal = pd.Series(0.0, index=df.index)
    signal[avg_ratio > 0.2] = -1.0  # More upper wicks -> selling pressure
    signal[avg_ratio < -0.2] = 1.0  # More lower wicks -> buying support
    return signal


def atr_breakout(df: pd.DataFrame) -> pd.Series:
    """ATR breakout: price moves beyond ATR band from previous close."""
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    prev_close = df["Close"].shift(1)
    signal = pd.Series(0.0, index=df.index)
    signal[df["Close"] > prev_close + 2 * atr] = 1.0
    signal[df["Close"] < prev_close - 2 * atr] = -1.0
    return signal


def realized_vs_implied_proxy(df: pd.DataFrame) -> pd.Series:
    """Realized vs implied vol proxy: current vol vs recent average."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_5 = log_ret.rolling(5).std() * np.sqrt(252)
    vol_60 = log_ret.rolling(60).std() * np.sqrt(252)
    # If realized < historical -> vol cheap -> expect expansion
    ratio = vol_5 / vol_60.replace(0, np.nan)
    signal = pd.Series(0.0, index=df.index)
    signal[ratio < 0.5] = 1.0  # Vol cheap
    signal[ratio > 2.0] = -1.0  # Vol expensive
    return signal


def vol_mean_reversion_20_60(df: pd.DataFrame) -> pd.Series:
    """Vol ratio 20/60 mean reversion."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
    vol_60 = log_ret.rolling(60).std() * np.sqrt(252)
    ratio = vol_20 / vol_60.replace(0, np.nan)
    signal = pd.Series(0.0, index=df.index)
    signal[ratio > 1.5] = -1.0  # High short-term vol -> will revert
    signal[ratio < 0.6] = 1.0  # Low short-term vol -> will expand
    return signal


def parkinson_vs_close_vol(df: pd.DataFrame) -> pd.Series:
    """Parkinson vol vs close-to-close vol divergence."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    cc_vol = log_ret.rolling(20).std() * np.sqrt(252)
    log_hl = np.log(df["High"] / df["Low"])
    park_var = (log_hl**2) / (4 * np.log(2))
    park_vol = np.sqrt(park_var.rolling(20).mean() * 252)
    ratio = park_vol / cc_vol.replace(0, np.nan)
    signal = pd.Series(0.0, index=df.index)
    # High Parkinson/CC ratio -> intraday moves not captured by close
    signal[ratio > 1.5] = 1.0
    signal[ratio < 0.7] = -1.0
    return signal




# =============================================================================
# GAP SIGNALS
# =============================================================================


def gap_reversion_large(df: pd.DataFrame) -> pd.Series:
    """Large overnight gap reversion: buy after large down gap."""
    gap = np.log(df["Open"] / df["Close"].shift(1))
    gap_std = gap.rolling(20).std()
    z = gap / gap_std.replace(0, np.nan)
    signal = pd.Series(0.0, index=df.index)
    signal[z < -1.5] = 1.0  # Large down gap -> buy for fill
    signal[z > 1.5] = -1.0  # Large up gap -> sell for fill
    return signal


def gap_fill_probability_size(df: pd.DataFrame) -> pd.Series:
    """Gap fill probability by size: smaller gaps more likely to fill."""
    gap = np.log(df["Open"] / df["Close"].shift(1))
    gap_abs = gap.abs()
    median_gap = gap_abs.rolling(60).median()
    # Small gaps (below median) tend to fill
    small_gap = gap_abs < median_gap
    signal = pd.Series(0.0, index=df.index)
    signal[small_gap & (gap > 0)] = -1.0  # Small up gap fills
    signal[small_gap & (gap < 0)] = 1.0  # Small down gap fills
    return signal


def gap_fill_by_direction(df: pd.DataFrame) -> pd.Series:
    """Gap fill direction bias: down gaps fill more often than up gaps."""
    gap = np.log(df["Open"] / df["Close"].shift(1))
    signal = pd.Series(0.0, index=df.index)
    signal[gap < -0.005] = 1.0  # Down gaps fill -> long
    signal[gap > 0.01] = -0.5  # Up gaps partially fill
    return signal


def gap_and_go(df: pd.DataFrame) -> pd.Series:
    """Gap and go: trade in gap direction if momentum confirms."""
    gap = np.log(df["Open"] / df["Close"].shift(1))
    # If close > open after gap up, gap-and-go confirmed
    intraday_ret = np.log(df["Close"] / df["Open"])
    signal = pd.Series(0.0, index=df.index)
    signal[(gap > 0.005) & (intraday_ret > 0)] = 1.0
    signal[(gap < -0.005) & (intraday_ret < 0)] = -1.0
    return signal


def unfilled_gap_support(df: pd.DataFrame) -> pd.Series:
    """Unfilled gaps as support/resistance levels."""
    gap = np.log(df["Open"] / df["Close"].shift(1))
    # Look for price approaching previous gap levels
    signal = pd.Series(0.0, index=df.index)
    # If recent gap down unfilled and price approaching, support
    for i in range(5, len(df)):
        recent_gaps = gap.iloc[max(0, i-20):i]
        large_down_gaps = recent_gaps[recent_gaps < -0.005]
        if len(large_down_gaps) > 0:
            signal.iloc[i] = 0.5  # Unfilled gap support nearby
    return signal.clip(-1, 1)


def monday_gap_effect(df: pd.DataFrame) -> pd.Series:
    """Monday gap effect: weekend gaps tend to be larger and fill."""
    gap = np.log(df["Open"] / df["Close"].shift(1))
    is_monday = df.index.dayofweek == 0
    signal = pd.Series(0.0, index=df.index)
    signal[is_monday & (gap > 0.003)] = -1.0  # Monday up gap fills
    signal[is_monday & (gap < -0.003)] = 1.0  # Monday down gap fills
    return signal


def holiday_gap_effect(df: pd.DataFrame) -> pd.Series:
    """Holiday gap: gaps after multi-day closures tend to fill."""
    # Detect gaps after non-consecutive trading days
    day_diff = pd.Series(df.index, index=df.index).diff().dt.days
    holiday_gap = day_diff > 3  # More than a weekend
    gap = np.log(df["Open"] / df["Close"].shift(1))
    signal = pd.Series(0.0, index=df.index)
    signal[holiday_gap & (gap > 0)] = -1.0
    signal[holiday_gap & (gap < 0)] = 1.0
    return signal


def gap_volume_confirmation(df: pd.DataFrame) -> pd.Series:
    """Gap + volume confirmation: gaps with high volume persist."""
    gap = np.log(df["Open"] / df["Close"].shift(1))
    vol_ratio = df["Volume"] / df["Volume"].rolling(20).mean()
    signal = pd.Series(0.0, index=df.index)
    # High volume gaps persist (gap and go)
    signal[(gap > 0.005) & (vol_ratio > 1.5)] = 1.0
    signal[(gap < -0.005) & (vol_ratio > 1.5)] = -1.0
    # Low volume gaps fill
    signal[(gap > 0.005) & (vol_ratio < 0.7)] = -0.5
    signal[(gap < -0.005) & (vol_ratio < 0.7)] = 0.5
    return signal


def gap_streak(df: pd.DataFrame) -> pd.Series:
    """Gap streak: consecutive same-direction gaps."""
    gap = np.log(df["Open"] / df["Close"].shift(1))
    gap_up = (gap > 0.001).astype(int)
    gap_down = (gap < -0.001).astype(int)
    consec_up = gap_up.groupby((gap_up != gap_up.shift()).cumsum()).cumsum()
    consec_down = gap_down.groupby((gap_down != gap_down.shift()).cumsum()).cumsum()
    signal = pd.Series(0.0, index=df.index)
    signal[consec_up >= 3] = -1.0  # 3+ up gaps -> expect reversal
    signal[consec_down >= 3] = 1.0
    return signal


def gap_size_relative(df: pd.DataFrame) -> pd.Series:
    """Gap size relative to ATR: larger relative gaps have different behavior."""
    gap = df["Open"] - df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    gap_atr = gap / atr.replace(0, np.nan)
    signal = pd.Series(0.0, index=df.index)
    # Gaps > 1 ATR tend to continue
    signal[gap_atr > 1] = 1.0
    signal[gap_atr < -1] = -1.0
    # Gaps < 0.5 ATR tend to fill
    signal[(gap_atr > 0) & (gap_atr < 0.5)] = -0.5
    signal[(gap_atr < 0) & (gap_atr > -0.5)] = 0.5
    return signal


# =============================================================================
# SESSION EFFECTS SIGNALS
# =============================================================================


def day_of_week_monday(df: pd.DataFrame) -> pd.Series:
    """Monday effect: historical weakness on Mondays."""
    signal = pd.Series(0.0, index=df.index)
    signal[df.index.dayofweek == 0] = -1.0  # Monday weakness
    return signal


def day_of_week_wednesday(df: pd.DataFrame) -> pd.Series:
    """Wednesday reversal effect."""
    # Wednesday tends to reverse Monday-Tuesday trend
    ret_2d = np.log(df["Close"] / df["Close"].shift(2))
    signal = pd.Series(0.0, index=df.index)
    is_wed = df.index.dayofweek == 2
    signal[is_wed & (ret_2d > 0)] = -1.0
    signal[is_wed & (ret_2d < 0)] = 1.0
    return signal


def day_of_week_friday(df: pd.DataFrame) -> pd.Series:
    """Friday effect: risk reduction before weekend."""
    signal = pd.Series(0.0, index=df.index)
    signal[df.index.dayofweek == 4] = -0.5  # Slight negative Friday bias
    return signal


def month_of_year_january(df: pd.DataFrame) -> pd.Series:
    """January effect: historically strong in January."""
    signal = pd.Series(0.0, index=df.index)
    signal[df.index.month == 1] = 1.0
    return signal


def month_of_year_september(df: pd.DataFrame) -> pd.Series:
    """September weakness: historically worst month."""
    signal = pd.Series(0.0, index=df.index)
    signal[df.index.month == 9] = -1.0
    return signal


def turn_of_month(df: pd.DataFrame) -> pd.Series:
    """Turn of month: last 3 and first 3 trading days tend to be strong."""
    day = df.index.day
    is_month_end = day >= 26
    is_month_start = day <= 3
    signal = pd.Series(0.0, index=df.index)
    signal[is_month_end | is_month_start] = 1.0
    return signal


def pre_holiday_bullishness(df: pd.DataFrame) -> pd.Series:
    """Post-holiday bullishness: first day back after multi-day break tends positive.

    Original "pre-holiday" detection requires knowing tomorrow's date (look-ahead).
    This implementation uses backward-looking data: if the gap between today and
    the previous trading day exceeds 3 calendar days, today is post-holiday.
    The economic rationale (short covering, optimism) applies similarly to
    the session immediately following a holiday break.
    """
    day_diff = pd.Series(df.index, index=df.index).diff().dt.days
    post_holiday = day_diff > 3
    signal = pd.Series(0.0, index=df.index)
    signal[post_holiday] = 1.0
    return signal


def options_expiration_week(df: pd.DataFrame) -> pd.Series:
    """Options expiration: third Friday of month and surrounding days."""
    # Third Friday: day between 15-21 that is a Friday
    is_third_week = (df.index.day >= 15) & (df.index.day <= 21)
    is_opex_week = is_third_week
    signal = pd.Series(0.0, index=df.index)
    # Opex week tends to have mean reversion (pin risk)
    ret_5 = np.log(df["Close"] / df["Close"].shift(5))
    signal[is_opex_week & (ret_5 > 0)] = -0.5
    signal[is_opex_week & (ret_5 < 0)] = 0.5
    return signal


def quarter_end_rebalancing(df: pd.DataFrame) -> pd.Series:
    """Quarter-end rebalancing: last 5 days of quarter."""
    is_quarter_end_month = df.index.month.isin([3, 6, 9, 12])
    is_late_month = df.index.day >= 25
    signal = pd.Series(0.0, index=df.index)
    signal[is_quarter_end_month & is_late_month] = 1.0  # Window dressing bullish
    return signal


def sell_in_may(df: pd.DataFrame) -> pd.Series:
    """Sell in May: reduce exposure May-October."""
    summer = df.index.month.isin([5, 6, 7, 8, 9, 10])
    signal = pd.Series(0.0, index=df.index)
    signal[summer] = -0.5
    signal[~summer] = 0.5
    return signal


def santa_rally(df: pd.DataFrame) -> pd.Series:
    """Santa rally: last 5 days of Dec + first 2 of January."""
    is_late_dec = (df.index.month == 12) & (df.index.day >= 25)
    is_early_jan = (df.index.month == 1) & (df.index.day <= 3)
    signal = pd.Series(0.0, index=df.index)
    signal[is_late_dec | is_early_jan] = 1.0
    return signal


def first_hour_proxy(df: pd.DataFrame) -> pd.Series:
    """First hour proxy: open-to-close direction persists from open strength."""
    # Use Open vs previous Close as proxy for first-hour momentum
    overnight = np.log(df["Open"] / df["Close"].shift(1))
    signal = pd.Series(0.0, index=df.index)
    signal[overnight > 0.003] = 1.0
    signal[overnight < -0.003] = -1.0
    return signal




# =============================================================================
# ORDER FLOW PROXY SIGNALS
# =============================================================================


def volume_imbalance(df: pd.DataFrame) -> pd.Series:
    """Volume imbalance: up-vol vs down-vol estimated from close in range."""
    close_pos = (df["Close"] - df["Low"]) / (df["High"] - df["Low"]).replace(0, np.nan)
    up_vol = close_pos * df["Volume"]
    down_vol = (1 - close_pos) * df["Volume"]
    imbalance = (up_vol - down_vol) / (up_vol + down_vol).replace(0, np.nan)
    signal = imbalance.rolling(5).mean()
    return signal.clip(-1, 1).fillna(0)


def obv_divergence(df: pd.DataFrame) -> pd.Series:
    """OBV divergence: price new high but OBV not confirming."""
    ret = np.sign(df["Close"].diff())
    obv = (ret * df["Volume"]).cumsum()
    price_high_20 = df["Close"].rolling(20).max()
    obv_high_20 = obv.rolling(20).max()
    signal = pd.Series(0.0, index=df.index)
    # Price at high but OBV not -> bearish divergence
    at_price_high = df["Close"] >= price_high_20 * 0.99
    obv_below_high = obv < obv_high_20 * 0.95
    signal[at_price_high & obv_below_high] = -1.0
    # Price at low but OBV not -> bullish divergence
    price_low_20 = df["Close"].rolling(20).min()
    obv_low_20 = obv.rolling(20).min()
    at_price_low = df["Close"] <= price_low_20 * 1.01
    obv_above_low = obv > obv_low_20 * 1.05
    signal[at_price_low & obv_above_low] = 1.0
    return signal


def price_volume_confirmation(df: pd.DataFrame) -> pd.Series:
    """Price-volume confirmation: price up + volume up = bullish."""
    ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_change = df["Volume"] / df["Volume"].shift(1) - 1
    signal = pd.Series(0.0, index=df.index)
    signal[(ret > 0) & (vol_change > 0.2)] = 1.0
    signal[(ret < 0) & (vol_change > 0.2)] = -1.0
    return signal


def price_volume_divergence(df: pd.DataFrame) -> pd.Series:
    """Price-volume divergence: price up but volume declining = weak."""
    ret_5 = np.log(df["Close"] / df["Close"].shift(5))
    vol_5 = df["Volume"].rolling(5).mean()
    vol_20 = df["Volume"].rolling(20).mean()
    vol_declining = vol_5 < vol_20 * 0.8
    signal = pd.Series(0.0, index=df.index)
    signal[(ret_5 > 0) & vol_declining] = -1.0  # Weak rally
    signal[(ret_5 < 0) & vol_declining] = 1.0  # Weak selloff
    return signal


def volume_at_extremes(df: pd.DataFrame) -> pd.Series:
    """Volume at price extremes: high volume at highs/lows signals reversal."""
    vol_ratio = df["Volume"] / df["Volume"].rolling(20).mean()
    price_high = df["Close"] >= df["Close"].rolling(20).max() * 0.99
    price_low = df["Close"] <= df["Close"].rolling(20).min() * 1.01
    signal = pd.Series(0.0, index=df.index)
    signal[price_high & (vol_ratio > 2)] = -1.0  # Climax top
    signal[price_low & (vol_ratio > 2)] = 1.0  # Selling climax
    return signal


def buying_pressure_proxy(df: pd.DataFrame) -> pd.Series:
    """Buying pressure: (Close - Low) / (High - Low) * Volume."""
    close_pos = (df["Close"] - df["Low"]) / (df["High"] - df["Low"]).replace(0, np.nan)
    buying = close_pos * df["Volume"]
    selling = (1 - close_pos) * df["Volume"]
    pressure = (buying - selling) / df["Volume"].replace(0, np.nan)
    signal = pressure.rolling(5).mean()
    return signal.clip(-1, 1).fillna(0)


def selling_pressure_proxy(df: pd.DataFrame) -> pd.Series:
    """Selling pressure: (High - Close) / (High - Low) * Volume."""
    sell_pos = (df["High"] - df["Close"]) / (df["High"] - df["Low"]).replace(0, np.nan)
    sell_pressure = sell_pos.rolling(10).mean()
    signal = pd.Series(0.0, index=df.index)
    signal[sell_pressure > 0.6] = -1.0  # High selling pressure
    signal[sell_pressure < 0.3] = 1.0  # Low selling pressure
    return signal


def chaikin_money_flow(df: pd.DataFrame) -> pd.Series:
    """Chaikin Money Flow (20-period)."""
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (
        df["High"] - df["Low"]
    ).replace(0, np.nan)
    mfv = mfm * df["Volume"]
    cmf = mfv.rolling(20).sum() / df["Volume"].rolling(20).sum()
    signal = cmf.clip(-1, 1).fillna(0)
    return signal


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution line trend."""
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (
        df["High"] - df["Low"]
    ).replace(0, np.nan)
    ad = (mfm.fillna(0) * df["Volume"]).cumsum()
    ad_ma = ad.rolling(20).mean()
    signal = pd.Series(0.0, index=df.index)
    signal[ad > ad_ma] = 1.0
    signal[ad < ad_ma] = -1.0
    return signal


def money_flow_index(df: pd.DataFrame) -> pd.Series:
    """Money Flow Index (MFI) - volume-weighted RSI."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = typical * df["Volume"]
    pos_mf = mf.where(typical > typical.shift(1), 0)
    neg_mf = mf.where(typical < typical.shift(1), 0)
    pos_sum = pos_mf.rolling(14).sum()
    neg_sum = neg_mf.rolling(14).sum()
    mfi = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))
    signal = pd.Series(0.0, index=df.index)
    signal[mfi < 20] = 1.0  # Oversold
    signal[mfi > 80] = -1.0  # Overbought
    return signal


def volume_weighted_momentum(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted price momentum."""
    ret = np.log(df["Close"] / df["Close"].shift(1))
    vw_ret = (ret * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()
    return np.sign(vw_ret).fillna(0)


def volume_breakout(df: pd.DataFrame) -> pd.Series:
    """Volume breakout: volume spike + directional move."""
    vol_ratio = df["Volume"] / df["Volume"].rolling(20).mean()
    ret = np.log(df["Close"] / df["Close"].shift(1))
    signal = pd.Series(0.0, index=df.index)
    signal[(vol_ratio > 2) & (ret > 0.01)] = 1.0
    signal[(vol_ratio > 2) & (ret < -0.01)] = -1.0
    return signal


def volume_dryup(df: pd.DataFrame) -> pd.Series:
    """Volume dry-up: very low volume precedes directional move."""
    vol_ratio = df["Volume"] / df["Volume"].rolling(20).mean()
    ret_5 = np.log(df["Close"] / df["Close"].shift(5))
    signal = pd.Series(0.0, index=df.index)
    # Low volume + slight uptrend -> continuation expected
    signal[(vol_ratio < 0.5) & (ret_5 > 0)] = 1.0
    signal[(vol_ratio < 0.5) & (ret_5 < 0)] = -1.0
    return signal


def up_down_volume_ratio(df: pd.DataFrame) -> pd.Series:
    """Up/down volume ratio over rolling window."""
    ret = df["Close"].diff()
    up_vol = df["Volume"].where(ret > 0, 0).rolling(10).sum()
    down_vol = df["Volume"].where(ret < 0, 0).rolling(10).sum()
    ratio = up_vol / down_vol.replace(0, np.nan)
    signal = pd.Series(0.0, index=df.index)
    signal[ratio > 2] = 1.0
    signal[ratio < 0.5] = -1.0
    return signal


def vwap_deviation(df: pd.DataFrame) -> pd.Series:
    """VWAP deviation: price far from rolling VWAP."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap_20 = (typical * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()
    deviation = (df["Close"] - vwap_20) / vwap_20
    signal = pd.Series(0.0, index=df.index)
    signal[deviation > 0.02] = -1.0  # Above VWAP -> mean revert
    signal[deviation < -0.02] = 1.0  # Below VWAP -> mean revert
    return signal


def volume_climax(df: pd.DataFrame) -> pd.Series:
    """Volume climax: extreme volume at price extreme = reversal."""
    vol_z = (df["Volume"] - df["Volume"].rolling(60).mean()) / df["Volume"].rolling(60).std()
    ret = np.log(df["Close"] / df["Close"].shift(1))
    signal = pd.Series(0.0, index=df.index)
    signal[(vol_z > 3) & (ret > 0.02)] = -1.0  # Buying climax
    signal[(vol_z > 3) & (ret < -0.02)] = 1.0  # Selling climax
    return signal




# =============================================================================
# REGIME / MARKET STRUCTURE SIGNALS
# =============================================================================


def adx_trend_detection(df: pd.DataFrame) -> pd.Series:
    """ADX-based trend detection: strong trend -> follow, weak -> fade."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    # When +DM > -DM, zero out -DM and vice versa
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=14, adjust=False).mean()
    signal = pd.Series(0.0, index=df.index)
    trending = adx > 25
    ret_10 = np.log(close / close.shift(10))
    signal[trending & (ret_10 > 0)] = 1.0
    signal[trending & (ret_10 < 0)] = -1.0
    return signal.fillna(0)


def volatility_regime_signal(df: pd.DataFrame) -> pd.Series:
    """Volatility regime: low vol = trend follow, high vol = mean revert."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
    vol_median = vol_20.rolling(252, min_periods=60).median()
    ret_10 = np.log(df["Close"] / df["Close"].shift(10))
    signal = pd.Series(0.0, index=df.index)
    low_vol = vol_20 < vol_median
    signal[low_vol] = np.sign(ret_10[low_vol])  # Trend follow
    signal[~low_vol] = -np.sign(ret_10[~low_vol])  # Mean revert
    return signal.fillna(0)


def correlation_regime_proxy(df: pd.DataFrame) -> pd.Series:
    """Correlation regime proxy: rolling correlation of returns with vol."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_5 = log_ret.rolling(5).std()
    corr = log_ret.rolling(60).corr(vol_5)
    signal = pd.Series(0.0, index=df.index)
    # Negative corr (normal) -> trend; positive corr (crisis) -> contrarian
    signal[corr < -0.3] = 1.0
    signal[corr > 0.3] = -1.0
    return signal.fillna(0)


def drawdown_recovery(df: pd.DataFrame) -> pd.Series:
    """Drawdown-relative signal: buy when recovering from drawdown."""
    cummax = df["Close"].cummax()
    drawdown = (df["Close"] - cummax) / cummax
    signal = pd.Series(0.0, index=df.index)
    # Deep drawdown + recent recovery
    recovering = (drawdown > -0.1) & (drawdown.shift(5) < -0.1)
    signal[recovering] = 1.0
    signal[drawdown < -0.15] = -1.0  # Still falling
    return signal


def momentum_dispersion(df: pd.DataFrame) -> pd.Series:
    """Market breadth proxy: momentum dispersion across timeframes."""
    rets = []
    for h in [5, 10, 20, 60]:
        rets.append(np.sign(np.log(df["Close"] / df["Close"].shift(h))))
    aligned = sum(rets) / 4
    return aligned.fillna(0)


def trend_strength(df: pd.DataFrame) -> pd.Series:
    """Trend strength: ratio of net move to total path length."""
    net_move = (df["Close"] - df["Close"].shift(20)).abs()
    path_length = df["Close"].diff().abs().rolling(20).sum()
    efficiency = net_move / path_length.replace(0, np.nan)
    ret_20 = np.log(df["Close"] / df["Close"].shift(20))
    signal = efficiency * np.sign(ret_20)
    return signal.clip(-1, 1).fillna(0)


def support_resistance_levels(df: pd.DataFrame) -> pd.Series:
    """Support/resistance from recent highs/lows."""
    high_20 = df["High"].rolling(20).max()
    low_20 = df["Low"].rolling(20).min()
    range_20 = high_20 - low_20
    pos_in_range = (df["Close"] - low_20) / range_20.replace(0, np.nan)
    signal = pd.Series(0.0, index=df.index)
    signal[pos_in_range > 0.9] = -0.5  # Near resistance
    signal[pos_in_range < 0.1] = 0.5  # Near support
    return signal.fillna(0)


def range_contraction_expansion(df: pd.DataFrame) -> pd.Series:
    """Range contraction/expansion cycle."""
    daily_range = df["High"] - df["Low"]
    range_ratio = daily_range / daily_range.rolling(20).mean()
    signal = pd.Series(0.0, index=df.index)
    # Contraction -> expect expansion in recent direction
    ret_5 = np.log(df["Close"] / df["Close"].shift(5))
    signal[range_ratio < 0.5] = np.sign(ret_5[range_ratio < 0.5])
    return signal.fillna(0)


def new_high_low_proximity(df: pd.DataFrame) -> pd.Series:
    """New high/low proximity: close to 252-day high or low."""
    high_252 = df["Close"].rolling(252, min_periods=60).max()
    low_252 = df["Close"].rolling(252, min_periods=60).min()
    near_high = df["Close"] > high_252 * 0.98
    near_low = df["Close"] < low_252 * 1.02
    signal = pd.Series(0.0, index=df.index)
    signal[near_high] = 1.0  # Near highs -> momentum
    signal[near_low] = -1.0  # Near lows -> further weakness
    return signal


def ma_ribbon(df: pd.DataFrame) -> pd.Series:
    """Moving average ribbon: multiple MAs alignment."""
    ma_10 = df["Close"].rolling(10).mean()
    ma_20 = df["Close"].rolling(20).mean()
    ma_50 = df["Close"].rolling(50).mean()
    ma_100 = df["Close"].rolling(100).mean()
    # All aligned bullish: 10 > 20 > 50 > 100
    bullish = (ma_10 > ma_20) & (ma_20 > ma_50) & (ma_50 > ma_100)
    bearish = (ma_10 < ma_20) & (ma_20 < ma_50) & (ma_50 < ma_100)
    signal = pd.Series(0.0, index=df.index)
    signal[bullish] = 1.0
    signal[bearish] = -1.0
    return signal


# =============================================================================
# MICROSTRUCTURE PROXY SIGNALS
# =============================================================================


def bar_range_analysis(df: pd.DataFrame) -> pd.Series:
    """Bar range analysis: wide vs narrow range bars."""
    daily_range = (df["High"] - df["Low"]) / df["Close"]
    avg_range = daily_range.rolling(20).mean()
    wide = daily_range > avg_range * 1.5
    narrow = daily_range < avg_range * 0.5
    ret = np.log(df["Close"] / df["Open"])
    signal = pd.Series(0.0, index=df.index)
    # Wide range bars in direction -> continuation
    signal[wide & (ret > 0)] = 1.0
    signal[wide & (ret < 0)] = -1.0
    # Narrow range -> breakout pending (use recent direction)
    ret_3 = np.log(df["Close"] / df["Close"].shift(3))
    signal[narrow] = np.sign(ret_3[narrow]) * 0.5
    return signal.fillna(0)


def doji_pattern(df: pd.DataFrame) -> pd.Series:
    """Doji pattern: small body relative to range (indecision)."""
    body = (df["Close"] - df["Open"]).abs()
    total_range = df["High"] - df["Low"]
    body_ratio = body / total_range.replace(0, np.nan)
    is_doji = body_ratio < 0.1
    # Doji after trend -> reversal signal
    ret_5 = np.log(df["Close"] / df["Close"].shift(5))
    signal = pd.Series(0.0, index=df.index)
    signal[is_doji & (ret_5 > 0.03)] = -1.0
    signal[is_doji & (ret_5 < -0.03)] = 1.0
    return signal


def hammer_pattern(df: pd.DataFrame) -> pd.Series:
    """Hammer/shooting star: long lower/upper wick patterns."""
    body = (df["Close"] - df["Open"]).abs()
    upper_wick = df["High"] - pd.concat([df["Close"], df["Open"]], axis=1).max(axis=1)
    lower_wick = pd.concat([df["Close"], df["Open"]], axis=1).min(axis=1) - df["Low"]
    total_range = df["High"] - df["Low"]
    # Hammer: lower wick > 2x body, small upper wick
    is_hammer = (lower_wick > 2 * body) & (upper_wick < body)
    # Shooting star: upper wick > 2x body, small lower wick
    is_shooting = (upper_wick > 2 * body) & (lower_wick < body)
    signal = pd.Series(0.0, index=df.index)
    signal[is_hammer] = 1.0  # Bullish reversal
    signal[is_shooting] = -1.0  # Bearish reversal
    return signal


def engulfing_pattern(df: pd.DataFrame) -> pd.Series:
    """Engulfing pattern: current bar body engulfs previous bar."""
    curr_body_high = pd.concat([df["Close"], df["Open"]], axis=1).max(axis=1)
    curr_body_low = pd.concat([df["Close"], df["Open"]], axis=1).min(axis=1)
    prev_body_high = pd.concat([df["Close"].shift(1), df["Open"].shift(1)], axis=1).max(axis=1)
    prev_body_low = pd.concat([df["Close"].shift(1), df["Open"].shift(1)], axis=1).min(axis=1)
    bullish_engulf = (curr_body_high > prev_body_high) & (curr_body_low < prev_body_low) & (df["Close"] > df["Open"])
    bearish_engulf = (curr_body_high > prev_body_high) & (curr_body_low < prev_body_low) & (df["Close"] < df["Open"])
    signal = pd.Series(0.0, index=df.index)
    signal[bullish_engulf] = 1.0
    signal[bearish_engulf] = -1.0
    return signal


def consecutive_direction(df: pd.DataFrame) -> pd.Series:
    """Consecutive direction analysis: streak of up/down closes."""
    ret = df["Close"].diff()
    up = (ret > 0).astype(int)
    down = (ret < 0).astype(int)
    consec_up = up.groupby((up != up.shift()).cumsum()).cumsum()
    consec_down = down.groupby((down != down.shift()).cumsum()).cumsum()
    signal = pd.Series(0.0, index=df.index)
    # Strong streak -> momentum
    signal[consec_up >= 4] = 1.0
    signal[consec_down >= 4] = -1.0
    # Very long streak -> exhaustion
    signal[consec_up >= 7] = -0.5
    signal[consec_down >= 7] = 0.5
    return signal


def opening_range_proxy(df: pd.DataFrame) -> pd.Series:
    """Opening range proxy: open-to-high/low ratio."""
    open_to_high = df["High"] - df["Open"]
    open_to_low = df["Open"] - df["Low"]
    total = open_to_high + open_to_low
    # If most range is above open -> bullish
    ratio = open_to_high / total.replace(0, np.nan)
    avg_ratio = ratio.rolling(5).mean()
    signal = pd.Series(0.0, index=df.index)
    signal[avg_ratio > 0.65] = 1.0
    signal[avg_ratio < 0.35] = -1.0
    return signal.fillna(0)


def close_position_in_range(df: pd.DataFrame) -> pd.Series:
    """Close position within daily range (IBS - Internal Bar Strength)."""
    ibs = (df["Close"] - df["Low"]) / (df["High"] - df["Low"]).replace(0, np.nan)
    signal = pd.Series(0.0, index=df.index)
    signal[ibs < 0.2] = 1.0  # Close near low -> buy next day
    signal[ibs > 0.8] = -1.0  # Close near high -> sell next day
    return signal.fillna(0)


def true_range_ratio(df: pd.DataFrame) -> pd.Series:
    """True range vs average true range ratio."""
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    ratio = tr / atr.replace(0, np.nan)
    ret = np.log(df["Close"] / df["Open"])
    signal = pd.Series(0.0, index=df.index)
    # High TR ratio -> continuation
    signal[(ratio > 2) & (ret > 0)] = 1.0
    signal[(ratio > 2) & (ret < 0)] = -1.0
    return signal.fillna(0)


def momentum_acceleration_bar(df: pd.DataFrame) -> pd.Series:
    """Bar-to-bar momentum acceleration."""
    ret = np.log(df["Close"] / df["Close"].shift(1))
    accel = ret - ret.shift(1)
    signal = pd.Series(0.0, index=df.index)
    signal[accel > 0.01] = 1.0
    signal[accel < -0.01] = -1.0
    return signal


def volume_range_relationship(df: pd.DataFrame) -> pd.Series:
    """Volume-range relationship: high volume + narrow range = accumulation."""
    daily_range = (df["High"] - df["Low"]) / df["Close"]
    vol_ratio = df["Volume"] / df["Volume"].rolling(20).mean()
    range_ratio = daily_range / daily_range.rolling(20).mean()
    signal = pd.Series(0.0, index=df.index)
    # High vol + narrow range = accumulation (bullish)
    signal[(vol_ratio > 1.5) & (range_ratio < 0.5)] = 1.0
    # High vol + wide range in direction
    ret = np.log(df["Close"] / df["Open"])
    signal[(vol_ratio > 1.5) & (range_ratio > 1.5) & (ret > 0)] = 1.0
    signal[(vol_ratio > 1.5) & (range_ratio > 1.5) & (ret < 0)] = -1.0
    return signal


def price_rejection(df: pd.DataFrame) -> pd.Series:
    """Price rejection: long wicks show rejection of price levels."""
    upper_wick = df["High"] - pd.concat([df["Close"], df["Open"]], axis=1).max(axis=1)
    lower_wick = pd.concat([df["Close"], df["Open"]], axis=1).min(axis=1) - df["Low"]
    total_range = (df["High"] - df["Low"]).replace(0, np.nan)
    upper_pct = upper_wick / total_range
    lower_pct = lower_wick / total_range
    signal = pd.Series(0.0, index=df.index)
    signal[upper_pct > 0.6] = -1.0  # Upper rejection -> bearish
    signal[lower_pct > 0.6] = 1.0  # Lower rejection -> bullish
    return signal.fillna(0)


def inside_outside_bar(df: pd.DataFrame) -> pd.Series:
    """Inside/outside bar patterns."""
    inside = (df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))
    outside = (df["High"] > df["High"].shift(1)) & (df["Low"] < df["Low"].shift(1))
    ret_5 = np.log(df["Close"] / df["Close"].shift(5))
    signal = pd.Series(0.0, index=df.index)
    # Inside bar -> breakout in trend direction
    signal[inside] = np.sign(ret_5[inside]) * 0.5
    # Outside bar -> reversal
    signal[outside & (df["Close"] > df["Open"])] = 1.0
    signal[outside & (df["Close"] < df["Open"])] = -1.0
    return signal.fillna(0)


