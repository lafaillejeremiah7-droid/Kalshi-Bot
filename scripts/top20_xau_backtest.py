from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

from xau_company.backtest import TradeLifecycleBacktester, TradeOutcome


SOURCE_REPO = "simom1/XAUUSD-history"
RAW_ROOT = f"https://raw.githubusercontent.com/{SOURCE_REPO}/main"
XAU_FILE = "Gold-Cash/XAUUSD/XAUUSD_M15_2010_2026.csv"
CACHE_ROOT = Path(os.getenv("BACKTEST_CACHE", "/tmp/xau-top20"))
REPORT_PATH = Path(os.getenv("TOP20_REPORT", "top20-xau-backtest-report.json"))

SELECTION_START = pd.Timestamp("2016-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2022-01-01", tz="UTC")
MIN_SELECTION_TRADES = 40
MAX_SELECTION_BH_Q = 0.10

BACKTESTER = TradeLifecycleBacktester(
    spread_bps=1.5,
    slippage_bps=0.5,
    stop_atr=1.20,
    reward_risk=1.70,
)


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    name: str
    hypothesis: str
    rule: str
    max_holding: int
    signal_builder: Callable[[pd.DataFrame], pd.Series]


def _download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1024:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _ in range(5):
        try:
            with requests.get(url, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                with path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
            if path.stat().st_size > 1024:
                return
        except Exception as exc:
            last_error = exc
            path.unlink(missing_ok=True)
    raise RuntimeError(f"download failed: {url}: {last_error}")


def _load_ohlc(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [
        str(c).strip().lower().replace("<", "").replace(">", "").replace(" ", "_")
        for c in raw.columns
    ]
    if "datetime" in raw:
        stamp = raw["datetime"]
    elif "time" in raw and "date" in raw:
        stamp = raw["date"].astype(str) + " " + raw["time"].astype(str)
    elif "time" in raw:
        stamp = raw["time"]
    elif "date" in raw:
        stamp = raw["date"]
    else:
        raise RuntimeError(f"no timestamp column in {path}; columns={list(raw.columns)}")

    out = pd.DataFrame({"datetime": pd.to_datetime(stamp, utc=True, errors="coerce")})
    for col in ("open", "high", "low", "close"):
        if col not in raw:
            raise RuntimeError(f"missing {col} in {path}; columns={list(raw.columns)}")
        out[col] = pd.to_numeric(raw[col], errors="coerce")

    out = (
        out.dropna(subset=["datetime", "open", "high", "low", "close"])
        .drop_duplicates(subset=["datetime"], keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    sane = (
        (out.open > 0)
        & (out.high >= out[["open", "close"]].max(axis=1))
        & (out.low <= out[["open", "close"]].min(axis=1))
        & (out.high >= out.low)
    )
    return out.loc[sane].reset_index(drop=True)


def _edge(cond: pd.Series) -> pd.Series:
    cond = cond.fillna(False).astype(bool)
    return cond & ~cond.shift(1, fill_value=False)


def _signed(df: pd.DataFrame, long_cond: pd.Series, short_cond: pd.Series) -> pd.Series:
    out = pd.Series(0, index=df.index, dtype="int8")
    out[_edge(long_cond)] = 1
    out[_edge(short_cond)] = -1
    return out


def _cross_up(a: pd.Series, b: pd.Series | float) -> pd.Series:
    if not isinstance(b, pd.Series):
        b = pd.Series(float(b), index=a.index)
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_down(a: pd.Series, b: pd.Series | float) -> pd.Series:
    if not isinstance(b, pd.Series):
        b = pd.Series(float(b), index=a.index)
    return (a < b) & (a.shift(1) >= b.shift(1))


def _ema(c: pd.Series, n: int) -> pd.Series:
    return c.ewm(span=n, adjust=False).mean()


def _sma(c: pd.Series, n: int) -> pd.Series:
    return c.rolling(n, min_periods=n).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df.close.shift(1)
    tr = pd.concat(
        [df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _adx(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = df.high.diff()
    dn = -df.low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = _atr(df, n).replace(0, np.nan)
    plus = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / a
    minus = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx, plus, minus


def _bollinger(c: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    m = _sma(c, n)
    s = c.rolling(n, min_periods=n).std()
    return m, m + k * s, m - k * s


def _prior_day_levels(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    day = pd.to_datetime(df.datetime, utc=True).dt.floor("D")
    temp = pd.DataFrame({"day": day, "high": df.high, "low": df.low})
    daily = temp.groupby("day", sort=True).agg(high=("high", "max"), low=("low", "min")).shift(1)
    return day.map(daily.high.to_dict()), day.map(daily.low.to_dict())


def _opening_range(df: pd.DataFrame, bars: int = 4) -> tuple[pd.Series, pd.Series, pd.Series]:
    day = pd.to_datetime(df.datetime, utc=True).dt.floor("D")
    order = df.close.groupby(day).cumcount()
    first = order < bars
    temp = pd.DataFrame({"day": day, "high": df.high.where(first), "low": df.low.where(first)})
    orh = temp.groupby("day").high.transform("max")
    orl = temp.groupby("day").low.transform("min")
    ready = order >= bars
    return orh.where(ready), orl.where(ready), ready


def _s01_sma_50_200(df: pd.DataFrame) -> pd.Series:
    fast, slow = _sma(df.close, 50), _sma(df.close, 200)
    return _signed(df, _cross_up(fast, slow), _cross_down(fast, slow))


def _s02_ema_20_50(df: pd.DataFrame) -> pd.Series:
    fast, slow = _ema(df.close, 20), _ema(df.close, 50)
    return _signed(df, _cross_up(fast, slow), _cross_down(fast, slow))


def _s03_ema_pullback(df: pd.DataFrame) -> pd.Series:
    e20, e50, e200 = _ema(df.close, 20), _ema(df.close, 50), _ema(df.close, 200)
    long_cond = (e50 > e200) & (df.low <= e20) & (df.close > e20)
    short_cond = (e50 < e200) & (df.high >= e20) & (df.close < e20)
    return _signed(df, long_cond, short_cond)


def _donchian_signal(df: pd.DataFrame, n: int) -> pd.Series:
    upper = df.high.rolling(n, min_periods=n).max().shift(1)
    lower = df.low.rolling(n, min_periods=n).min().shift(1)
    return _signed(df, df.close > upper, df.close < lower)


def _s04_donchian20(df: pd.DataFrame) -> pd.Series:
    return _donchian_signal(df, 20)


def _s05_donchian55(df: pd.DataFrame) -> pd.Series:
    return _donchian_signal(df, 55)


def _s06_tsmom_63d(df: pd.DataFrame) -> pd.Series:
    lookback = 96 * 63
    roc = df.close / df.close.shift(lookback) - 1.0
    return _signed(df, _cross_up(roc, 0.0), _cross_down(roc, 0.0))


def _s07_52week_breakout(df: pd.DataFrame) -> pd.Series:
    lookback = 96 * 252
    upper = df.high.rolling(lookback, min_periods=lookback).max().shift(1)
    lower = df.low.rolling(lookback, min_periods=lookback).min().shift(1)
    return _signed(df, df.close > upper, df.close < lower)


def _s08_adx_dmi(df: pd.DataFrame) -> pd.Series:
    adx, plus, minus = _adx(df, 14)
    long_cond = (adx > 25) & _cross_up(plus, minus)
    short_cond = (adx > 25) & _cross_down(plus, minus)
    return _signed(df, long_cond, short_cond)


def _s09_macd_signal(df: pd.DataFrame) -> pd.Series:
    macd = _ema(df.close, 12) - _ema(df.close, 26)
    sig = _ema(macd, 9)
    return _signed(df, _cross_up(macd, sig), _cross_down(macd, sig))


def _s10_rsi_centerline(df: pd.DataFrame) -> pd.Series:
    r = _rsi(df.close, 14)
    return _signed(df, _cross_up(r, 50.0), _cross_down(r, 50.0))


def _s11_bollinger_breakout(df: pd.DataFrame) -> pd.Series:
    _, upper, lower = _bollinger(df.close, 20, 2.0)
    return _signed(df, df.close > upper, df.close < lower)


def _s12_bollinger_reentry(df: pd.DataFrame) -> pd.Series:
    _, upper, lower = _bollinger(df.close, 20, 2.0)
    long_cond = (df.close > lower) & (df.close.shift(1) <= lower.shift(1))
    short_cond = (df.close < upper) & (df.close.shift(1) >= upper.shift(1))
    return _signed(df, long_cond, short_cond)


def _s13_rsi2_trend_reversion(df: pd.DataFrame) -> pd.Series:
    r = _rsi(df.close, 2)
    trend = _sma(df.close, 200)
    long_cond = (df.close > trend) & (r < 10)
    short_cond = (df.close < trend) & (r > 90)
    return _signed(df, long_cond, short_cond)


def _s14_zscore_reversion(df: pd.DataFrame) -> pd.Series:
    m = _sma(df.close, 40)
    sd = df.close.rolling(40, min_periods=40).std().replace(0, np.nan)
    z = (df.close - m) / sd
    return _signed(df, z < -2.0, z > 2.0)


def _s15_volatility_expansion(df: pd.DataFrame) -> pd.Series:
    a = _atr(df, 20).shift(1)
    rng = (df.high - df.low).replace(0, np.nan)
    body_pos = (df.close - df.low) / rng
    long_cond = (rng > 1.5 * a) & (body_pos >= 0.80) & (df.close > df.open)
    short_cond = (rng > 1.5 * a) & (body_pos <= 0.20) & (df.close < df.open)
    return _signed(df, long_cond, short_cond)


def _s16_bollinger_squeeze(df: pd.DataFrame) -> pd.Series:
    mid, upper, lower = _bollinger(df.close, 20, 2.0)
    width = (upper - lower) / mid.replace(0, np.nan)
    threshold = width.rolling(252, min_periods=252).quantile(0.20).shift(1)
    squeeze = width.shift(1) <= threshold
    return _signed(df, squeeze & (df.close > upper), squeeze & (df.close < lower))


def _s17_opening_range_breakout(df: pd.DataFrame) -> pd.Series:
    orh, orl, ready = _opening_range(df, 4)
    return _signed(df, ready & (df.close > orh), ready & (df.close < orl))


def _s18_previous_day_breakout(df: pd.DataFrame) -> pd.Series:
    pdh, pdl = _prior_day_levels(df)
    return _signed(df, df.close > pdh, df.close < pdl)


def _s19_regression_channel(df: pd.DataFrame) -> pd.Series:
    n = 80
    x = np.arange(n, dtype=float)
    xm = float(x.mean())
    denom = float(((x - xm) ** 2).sum())

    def endpoint(y: np.ndarray) -> float:
        slope = float(((x - xm) * (y - y.mean())).sum() / denom)
        return float(y.mean() + slope * (x[-1] - xm))

    fitted = df.close.rolling(n, min_periods=n).apply(endpoint, raw=True)
    resid = df.close - fitted
    sd = resid.rolling(n, min_periods=n).std()
    upper, lower = fitted + 2.0 * sd, fitted - 2.0 * sd
    return _signed(df, df.close > upper, df.close < lower)


def _s20_keltner_breakout(df: pd.DataFrame) -> pd.Series:
    mid = _ema(df.close, 20)
    a = _atr(df, 20)
    upper, lower = mid + 2.0 * a, mid - 2.0 * a
    return _signed(df, df.close > upper, df.close < lower)


STRATEGIES = (
    StrategySpec("T01", "SMA 50/200 crossover", "Persistent gold trends can outlast intermediate noise.", "Long when SMA50 crosses above SMA200; short on the inverse crossover.", 384, _s01_sma_50_200),
    StrategySpec("T02", "EMA 20/50 crossover", "Faster exponential trend detection may capture medium-horizon continuation.", "Long on EMA20>EMA50 crossover; short on EMA20<EMA50 crossover.", 192, _s02_ema_20_50),
    StrategySpec("T03", "EMA trend pullback/reclaim", "Pullbacks inside an established trend can offer continuation entries.", "Require EMA50 above/below EMA200; enter when price tests EMA20 and closes back with the trend.", 96, _s03_ema_pullback),
    StrategySpec("T04", "Donchian 20 breakout", "New 20-bar extremes may initiate continuation.", "Long on close above prior 20-bar high; short below prior 20-bar low.", 96, _s04_donchian20),
    StrategySpec("T05", "Donchian 55 breakout", "Longer channel breaks may isolate stronger trends.", "Long on close above prior 55-bar high; short below prior 55-bar low.", 192, _s05_donchian55),
    StrategySpec("T06", "63-day time-series momentum", "Intermediate-horizon gold returns may exhibit continuation.", "Long when 63-day approximate M15 return crosses above zero; short when below zero.", 384, _s06_tsmom_63d),
    StrategySpec("T07", "52-week extreme breakout", "Major new highs/lows may signal structural regime continuation.", "Long above prior approximate 52-week high; short below prior 52-week low.", 384, _s07_52week_breakout),
    StrategySpec("T08", "ADX/DMI trend", "Directional movement with strong ADX may distinguish trend from chop.", "ADX14>25 and +DI crosses -DI for long; inverse for short.", 96, _s08_adx_dmi),
    StrategySpec("T09", "MACD signal crossover", "Momentum acceleration relative to two EMAs may predict continuation.", "Long when MACD(12,26) crosses above signal(9); short on inverse.", 96, _s09_macd_signal),
    StrategySpec("T10", "RSI centerline momentum", "RSI crossing its neutral level can represent directional momentum.", "Long on RSI14 cross above 50; short below 50.", 64, _s10_rsi_centerline),
    StrategySpec("T11", "Bollinger breakout", "Closes outside volatility bands may mark expansion continuation.", "Long close above upper 20,2 band; short below lower band.", 64, _s11_bollinger_breakout),
    StrategySpec("T12", "Bollinger re-entry mean reversion", "Extreme deviations may revert after price re-enters the band.", "Long on close crossing back above lower band after being below; short inverse at upper band.", 32, _s12_bollinger_reentry),
    StrategySpec("T13", "RSI(2) trend-filtered reversion", "Very short-term exhaustion can revert inside a long-term trend.", "Long RSI2<10 only above SMA200; short RSI2>90 only below SMA200.", 32, _s13_rsi2_trend_reversion),
    StrategySpec("T14", "40-bar z-score mean reversion", "Large standardized deviations from a rolling mean may revert.", "Long z<-2; short z>2 using rolling 40-bar mean/std.", 32, _s14_zscore_reversion),
    StrategySpec("T15", "ATR volatility-expansion continuation", "Large directional range expansion can persist into the next bars.", "Signal when range>1.5x prior ATR20 and close is in outer 20% of the candle in its direction.", 32, _s15_volatility_expansion),
    StrategySpec("T16", "Bollinger squeeze breakout", "Low-volatility compression can precede directional expansion.", "Require prior band width in bottom 20% of trailing 252 bars, then close outside current band.", 96, _s16_bollinger_squeeze),
    StrategySpec("T17", "UTC opening-range breakout", "Early-session price discovery can define intraday directional levels.", "After first four M15 bars of each UTC day, long above opening-range high; short below low.", 32, _s17_opening_range_breakout),
    StrategySpec("T18", "Previous-day high/low breakout", "Prior daily extremes act as objective liquidity/price-discovery thresholds.", "Long above previous UTC-day high; short below previous UTC-day low.", 64, _s18_previous_day_breakout),
    StrategySpec("T19", "Regression-channel breakout", "Breaks away from a rolling linear trend by >2 residual SD may continue.", "Fit 80-bar OLS endpoint; long above +2 residual SD, short below -2 residual SD.", 96, _s19_regression_channel),
    StrategySpec("T20", "Keltner-channel breakout", "ATR-normalized channel breaks may capture volatility-adjusted trends.", "Long close above EMA20+2*ATR20; short below EMA20-2*ATR20.", 64, _s20_keltner_breakout),
)


def _metrics(trades: list[TradeOutcome]) -> dict[str, float | int]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "profit_factor": 1.0,
            "max_drawdown_r": 0.0,
            "max_loss_streak": 0,
            "total_r": 0.0,
            "trade_sharpe": 0.0,
        }
    rs = np.asarray([t.r_multiple for t in trades], dtype=float)
    sd = float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0
    return {
        "trades": int(len(trades)),
        "win_rate": float(np.mean([t.won for t in trades])),
        "avg_r": float(np.mean(rs)),
        "profit_factor": float(BACKTESTER.profit_factor(trades)),
        "max_drawdown_r": float(BACKTESTER.max_drawdown_r(trades)),
        "max_loss_streak": int(BACKTESTER.max_loss_streak(trades)),
        "total_r": float(rs.sum()),
        "trade_sharpe": float(np.mean(rs) / sd * math.sqrt(len(rs))) if sd > 1e-12 else 0.0,
    }


def _subset(trades: list[TradeOutcome], times: pd.Series, start: pd.Timestamp | None, end: pd.Timestamp | None) -> list[TradeOutcome]:
    out: list[TradeOutcome] = []
    for trade in trades:
        ts = pd.Timestamp(times.iloc[trade.signal_index])
        if start is not None and ts < start:
            continue
        if end is not None and ts >= end:
            continue
        out.append(trade)
    return out


def _yearly(trades: list[TradeOutcome], times: pd.Series) -> dict[str, dict[str, float | int]]:
    buckets: dict[int, list[TradeOutcome]] = {}
    for trade in trades:
        year = int(pd.Timestamp(times.iloc[trade.signal_index]).year)
        buckets.setdefault(year, []).append(trade)
    return {str(year): _metrics(rows) for year, rows in sorted(buckets.items())}


def _positive_edge_pvalue(trades: list[TradeOutcome]) -> float:
    if len(trades) < 3:
        return 1.0
    a = np.asarray([t.r_multiple for t in trades], dtype=float)
    sd = float(np.std(a, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        return 0.0 if float(np.mean(a)) > 0 else 1.0
    z = float(np.mean(a)) / (sd / math.sqrt(len(a)))
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def _bh_qvalues(rows: list[dict[str, object]]) -> None:
    valid = [(i, float(r["selection_p_value"])) for i, r in enumerate(rows)]
    valid.sort(key=lambda item: item[1])
    m = len(valid)
    running = 1.0
    q = [1.0] * m
    for pos in range(m - 1, -1, -1):
        _, p = valid[pos]
        running = min(running, p * m / (pos + 1))
        q[pos] = min(1.0, running)
    for pos, (idx, _) in enumerate(valid):
        rows[idx]["selection_bh_q"] = q[pos]


def _selection_score(m: dict[str, float | int], yearly: dict[str, dict[str, float | int]]) -> float:
    n = int(m["trades"])
    if n <= 0:
        return -1e9
    avg_r = float(m["avg_r"])
    pf = float(m["profit_factor"])
    dd = float(m["max_drawdown_r"])
    relevant = [v for y, v in yearly.items() if 2016 <= int(y) < 2022 and int(v["trades"]) >= 3]
    consistency = sum(float(v["avg_r"]) > 0 for v in relevant) / max(1, len(relevant))
    expectancy = 0.5 + 0.5 * math.tanh(avg_r / 0.25)
    pf_score = min(1.0, max(0.0, (pf - 0.8) / 1.2))
    sample = min(1.0, math.log1p(n) / math.log(501))
    dd_score = math.exp(-max(0.0, dd) / 15.0)
    return float(0.35 * expectancy + 0.20 * pf_score + 0.15 * consistency + 0.15 * sample + 0.15 * dd_score)


def _lookahead_check(spec: StrategySpec, df: pd.DataFrame) -> tuple[bool, str | None]:
    try:
        full = spec.signal_builder(df)
    except Exception as exc:
        return False, f"full signal failed: {type(exc).__name__}: {exc}"
    for ratio in (0.60, 0.78):
        cut = int(len(df) * ratio)
        try:
            prefix = spec.signal_builder(df.iloc[:cut].copy().reset_index(drop=True))
        except Exception as exc:
            return False, f"prefix signal failed at {ratio:.2f}: {type(exc).__name__}: {exc}"
        a = full.iloc[:cut].reset_index(drop=True).to_numpy(dtype=np.int8)
        b = prefix.reset_index(drop=True).to_numpy(dtype=np.int8)
        if len(a) != len(b) or not np.array_equal(a, b):
            return False, f"future-data sensitivity detected at prefix ratio {ratio:.2f}"
    return True, None


def main() -> None:
    if len(STRATEGIES) != 20 or len({s.strategy_id for s in STRATEGIES}) != 20:
        raise RuntimeError("top-20 study must contain exactly 20 unique methodologies")

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    xau_path = CACHE_ROOT / "xau_m15.csv"
    _download(f"{RAW_ROOT}/{XAU_FILE}", xau_path)
    xau = _load_ohlc(xau_path)
    atr14 = _atr(xau, 14)

    rows: list[dict[str, object]] = []
    for ordinal, spec in enumerate(STRATEGIES, start=1):
        passed, reason = _lookahead_check(spec, xau)
        row: dict[str, object] = {
            "strategy_id": spec.strategy_id,
            "name": spec.name,
            "hypothesis": spec.hypothesis,
            "objective_rule": spec.rule,
            "max_holding_m15_bars": spec.max_holding,
            "lookahead_check": "passed" if passed else "failed",
        }
        if not passed:
            row.update(status="rejected_lookahead", rejection_reason=reason)
            rows.append(row)
            print(f"[{ordinal:02d}/20] {spec.strategy_id} rejected: {reason}")
            continue

        signal = spec.signal_builder(xau)
        trades = BACKTESTER.simulate(xau, signal, atr14, spec.max_holding)
        dev = _subset(trades, xau.datetime, None, SELECTION_START)
        sel = _subset(trades, xau.datetime, SELECTION_START, HOLDOUT_START)
        hold = _subset(trades, xau.datetime, HOLDOUT_START, None)
        yearly = _yearly(trades, xau.datetime)
        sel_m = _metrics(sel)

        row.update(
            status="evaluated",
            total=_metrics(trades),
            development=_metrics(dev),
            selection=sel_m,
            holdout=_metrics(hold),
            yearly=yearly,
            selection_p_value=_positive_edge_pvalue(sel),
            selection_score=_selection_score(sel_m, yearly),
        )
        rows.append(row)
        print(
            f"[{ordinal:02d}/20] {spec.strategy_id} {spec.name}: "
            f"selection n={sel_m['trades']} avgR={sel_m['avg_r']:.4f} "
            f"PF={sel_m['profit_factor']:.3f}"
        )

    evaluated = [r for r in rows if r.get("status") == "evaluated"]
    _bh_qvalues(evaluated)
    for row in evaluated:
        sel = row["selection"]
        row["selection_gate"] = bool(
            int(sel["trades"]) >= MIN_SELECTION_TRADES
            and float(sel["avg_r"]) > 0
            and float(sel["profit_factor"]) > 1.0
            and float(row.get("selection_bh_q", 1.0)) <= MAX_SELECTION_BH_Q
        )

    ranked = sorted(
        evaluated,
        key=lambda r: (
            bool(r.get("selection_gate")),
            float(r["selection_score"]),
            float(r["selection"]["avg_r"]),
            int(r["selection"]["trades"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["selection_rank"] = rank

    report = {
        "schema_version": 1,
        "purpose": "Independent mathematical comparison of 20 historically supported XAUUSD/commodity strategy families.",
        "important_limit": "This study does not claim these are the world's 20 most profitable strategies. Proprietary global strategy returns are not observable. They are high-priority objective candidates selected from documented trend, momentum, breakout and mean-reversion families.",
        "methodology": {
            "strategy_count": 20,
            "parameter_trials_per_strategy": 1,
            "selection_period": f"{SELECTION_START.isoformat()} to {HOLDOUT_START.isoformat()}",
            "holdout_period": f"{HOLDOUT_START.isoformat()} onward",
            "rank_uses_holdout": False,
            "entry_timing": "signal after M15 close; entry at next M15 open",
            "spread_bps": BACKTESTER.spread_bps,
            "slippage_bps": BACKTESTER.slippage_bps,
            "stop_atr": BACKTESTER.stop_atr,
            "reward_risk": BACKTESTER.reward_risk,
            "same_bar_stop_target_collision": "stop first",
            "anti_lookahead": "prefix invariance at 60% and 78% of history for every strategy",
            "multiple_testing": "one-sided positive-edge p-value with Benjamini-Hochberg q-values across the 20 selection tests",
        },
        "data": {
            "source_repository": SOURCE_REPO,
            "file": XAU_FILE,
            "rows": int(len(xau)),
            "start": pd.Timestamp(xau.datetime.iloc[0]).isoformat(),
            "end": pd.Timestamp(xau.datetime.iloc[-1]).isoformat(),
        },
        "counts": {
            "evaluated": len(evaluated),
            "lookahead_rejections": sum(r.get("status") == "rejected_lookahead" for r in rows),
            "selection_gate_passes": sum(bool(r.get("selection_gate")) for r in evaluated),
        },
        "ranking_selection_only": [
            {
                "rank": r["selection_rank"],
                "strategy_id": r["strategy_id"],
                "name": r["name"],
                "selection_gate": r["selection_gate"],
                "selection_score": r["selection_score"],
                "selection": r["selection"],
                "selection_bh_q": r.get("selection_bh_q"),
                "holdout": r["holdout"],
            }
            for r in ranked
        ],
        "strategies": rows,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
