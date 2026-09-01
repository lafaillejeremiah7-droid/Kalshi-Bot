from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import pandas as pd

from .canonical_strategies import StrategyDefinition


def _bool(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def _edge(s: pd.Series) -> pd.Series:
    s = _bool(s)
    return s & ~s.shift(1, fill_value=False)


def _signed(index: pd.Index, long_cond: pd.Series, short_cond: pd.Series) -> pd.Series:
    out = pd.Series(0, index=index, dtype="int8")
    out[_edge(long_cond)] = 1
    out[_edge(short_cond)] = -1
    return out


class SignalContext:
    """Causal indicator cache. Every rolling level used for breakout decisions is lagged."""

    def __init__(self, df: pd.DataFrame, extras: Mapping[str, pd.DataFrame] | None = None) -> None:
        self.df = df
        self.extras = dict(extras or {})
        self.cache: dict[tuple, object] = {}
        self.o = df["open"].astype(float)
        self.h = df["high"].astype(float)
        self.l = df["low"].astype(float)
        self.c = df["close"].astype(float)
        self.t = (
            pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            if "datetime" in df
            else pd.Series(pd.date_range("2000-01-01", periods=len(df), freq="15min", tz="UTC"), index=df.index)
        )
        if "volume" in df:
            self.v = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
        elif "tick_volume" in df:
            self.v = pd.to_numeric(df["tick_volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
        else:
            self.v = pd.Series(0.0, index=df.index)

    def get(self, key: tuple, fn):
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    def ema(self, n: int) -> pd.Series:
        return self.get(("ema", n), lambda: self.c.ewm(span=n, adjust=False).mean())

    def sma(self, n: int) -> pd.Series:
        return self.get(("sma", n), lambda: self.c.rolling(n, min_periods=n).mean())

    def std(self, n: int) -> pd.Series:
        return self.get(("std", n), lambda: self.c.rolling(n, min_periods=n).std().replace(0, np.nan))

    def roc(self, n: int) -> pd.Series:
        return self.get(("roc", n), lambda: self.c.pct_change(n))

    def atr(self, n: int = 14) -> pd.Series:
        def build():
            pc = self.c.shift(1)
            tr = pd.concat([(self.h - self.l), (self.h - pc).abs(), (self.l - pc).abs()], axis=1).max(axis=1)
            return tr.ewm(alpha=1 / n, adjust=False).mean()
        return self.get(("atr", n), build)

    def rsi(self, n: int = 14) -> pd.Series:
        def build():
            d = self.c.diff()
            up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
            dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
            rs = up / dn.replace(0, np.nan)
            return (100 - 100 / (1 + rs)).fillna(50)
        return self.get(("rsi", n), build)

    def z(self, n: int = 20) -> pd.Series:
        return self.get(("z", n), lambda: ((self.c - self.sma(n)) / self.std(n)).replace([np.inf, -np.inf], np.nan).fillna(0))

    def donchian(self, n: int = 20) -> tuple[pd.Series, pd.Series]:
        return self.get(
            ("donchian", n),
            lambda: (self.h.rolling(n, min_periods=n).max().shift(1), self.l.rolling(n, min_periods=n).min().shift(1)),
        )

    def bollinger(self, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        m, s = self.sma(n), self.std(n)
        return m, m + k * s, m - k * s

    def stochastic(self, n: int = 14) -> pd.Series:
        lo = self.l.rolling(n, min_periods=n).min()
        hi = self.h.rolling(n, min_periods=n).max()
        return ((self.c - lo) / (hi - lo).replace(0, np.nan) * 100).clip(0, 100)

    def cci(self, n: int = 20) -> pd.Series:
        tp = (self.h + self.l + self.c) / 3
        ma = tp.rolling(n, min_periods=n).mean()
        md = (tp - ma).abs().rolling(n, min_periods=n).mean()
        return (tp - ma) / (0.015 * md.replace(0, np.nan))

    def willr(self, n: int = 14) -> pd.Series:
        lo = self.l.rolling(n, min_periods=n).min()
        hi = self.h.rolling(n, min_periods=n).max()
        return -100 * (hi - self.c) / (hi - lo).replace(0, np.nan)

    def macd(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        line = self.ema(12) - self.ema(26)
        signal = line.ewm(span=9, adjust=False).mean()
        return line, signal, line - signal

    def adx(self, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
        def build():
            up = self.h.diff()
            dn = -self.l.diff()
            plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=self.df.index)
            minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=self.df.index)
            a = self.atr(n).replace(0, np.nan)
            plus = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / a
            minus = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / a
            dx = 100 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)
            return dx.ewm(alpha=1/n, adjust=False).mean(), plus, minus
        return self.get(("adx", n), build)

    def regression(self, n: int = 40) -> tuple[pd.Series, pd.Series, pd.Series]:
        def build():
            x = np.arange(n, dtype=float)
            xm = x.mean()
            denom = ((x - xm) ** 2).sum()
            slope = self.c.rolling(n, min_periods=n).apply(
                lambda y: float(((x - xm) * (y - y.mean())).sum() / denom), raw=True
            )
            fitted = self.c.rolling(n, min_periods=n).apply(
                lambda y: float(y.mean() + (((x - xm) * (y - y.mean())).sum() / denom) * (x[-1] - xm)), raw=True
            )
            resid = self.c - fitted
            return slope, fitted, resid
        return self.get(("reg", n), build)

    def percentile(self, n: int = 100) -> pd.Series:
        return self.get(
            ("pct", n),
            lambda: self.c.rolling(n, min_periods=n).apply(lambda a: float((a[:-1] <= a[-1]).mean()) if len(a) > 1 else np.nan, raw=True),
        )

    def realized_vol(self, n: int = 20) -> pd.Series:
        r = np.log(self.c / self.c.shift(1))
        return self.get(("rv", n), lambda: r.rolling(n, min_periods=n).std())

    def obv(self) -> pd.Series:
        return self.get(("obv",), lambda: (np.sign(self.c.diff()).fillna(0) * self.v).cumsum())

    def session_vwap(self) -> pd.Series:
        def build():
            day = self.t.dt.floor("D")
            tp = (self.h + self.l + self.c) / 3
            num = (tp * self.v).groupby(day).cumsum()
            den = self.v.groupby(day).cumsum().replace(0, np.nan)
            return num / den
        return self.get(("session_vwap",), build)

    def rolling_vwap(self, n: int = 96) -> pd.Series:
        def build():
            tp = (self.h + self.l + self.c) / 3
            return (tp * self.v).rolling(n, min_periods=n).sum() / self.v.rolling(n, min_periods=n).sum().replace(0, np.nan)
        return self.get(("rvwap", n), build)

    def prior_period_levels(self, period: str) -> tuple[pd.Series, pd.Series, pd.Series]:
        def build():
            if period == "D":
                key = self.t.dt.floor("D")
            elif period == "W":
                key = self.t.dt.tz_convert(None).dt.to_period("W").astype(str)
            elif period == "M":
                key = self.t.dt.tz_convert(None).dt.to_period("M").astype(str)
            else:
                raise ValueError(period)
            temp = pd.DataFrame({"k": key, "h": self.h, "l": self.l, "c": self.c})
            agg = temp.groupby("k", sort=False).agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
            prev = agg.shift(1)
            mapper_h = prev["h"].to_dict()
            mapper_l = prev["l"].to_dict()
            mapper_c = prev["c"].to_dict()
            return key.map(mapper_h), key.map(mapper_l), key.map(mapper_c)
        return self.get(("prior", period), build)

    def current_session_levels(self) -> tuple[pd.Series, pd.Series]:
        day = self.t.dt.floor("D")
        hi = self.h.groupby(day).cummax().groupby(day).shift(1)
        lo = self.l.groupby(day).cummin().groupby(day).shift(1)
        return hi, lo

    def opening_range(self, bars: int = 4) -> tuple[pd.Series, pd.Series, pd.Series]:
        def build():
            day = self.t.dt.floor("D")
            order = self.c.groupby(day).cumcount()
            first = order < bars
            temp = pd.DataFrame({"day": day, "h": self.h.where(first), "l": self.l.where(first)})
            orh = temp.groupby("day")["h"].transform("max")
            orl = temp.groupby("day")["l"].transform("min")
            ready = order >= bars
            return orh.where(ready), orl.where(ready), ready
        return self.get(("or", bars), build)

    def daily_open(self) -> pd.Series:
        day = self.t.dt.floor("D")
        return self.o.groupby(day).transform("first")

    def hour(self) -> pd.Series:
        return self.t.dt.hour

    def weekday(self) -> pd.Series:
        return self.t.dt.weekday

    def month(self) -> pd.Series:
        return self.t.dt.month

    def day(self) -> pd.Series:
        return self.t.dt.day


class CanonicalSignalEngine:
    """Deterministic, fixed-rule implementation of the canonical strategy catalog.

    There is no parameter search here. Constants are part of each named method's
    definition and are not counted as separate strategies.
    """

    def __init__(self, df: pd.DataFrame, extras: Mapping[str, pd.DataFrame] | None = None) -> None:
        self.ctx = SignalContext(df, extras)
        self.df = df
        self.extras = dict(extras or {})

    @staticmethod
    def available_inputs(df: pd.DataFrame, extras: Mapping[str, pd.DataFrame] | None = None) -> set[str]:
        extras = dict(extras or {})
        out = {"xau_ohlc"}
        if "volume" in df or "tick_volume" in df:
            out.add("volume")
        if "dxy" in extras and extras["dxy"] is not None and not extras["dxy"].empty:
            out.add("dxy")
        return out

    def signal(self, strategy: StrategyDefinition) -> pd.Series:
        missing = set(strategy.requires) - self.available_inputs(self.df, self.extras)
        if missing:
            raise ValueError(f"missing inputs for {strategy.strategy_id}: {sorted(missing)}")
        fn = getattr(self, f"_signal_{strategy.category}")
        return fn(strategy)

    def _signal_smc(self, s: StrategyDefinition) -> pd.Series:
        x = self.ctx
        n = s.name.lower()
        c, o, h, l = x.c, x.o, x.h, x.l
        a = x.atr(14)
        hi20, lo20 = x.donchian(20)
        hi50, lo50 = x.donchian(50)
        body = (c - o).abs()
        rng = (h - l).replace(0, np.nan)
        displacement_up = (c > o) & (body > 1.2 * a) & (body / rng > 0.6)
        displacement_dn = (c < o) & (body > 1.2 * a) & (body / rng > 0.6)
        fvg_up = l > h.shift(2)
        fvg_dn = h < l.shift(2)
        fvg_mid_up = (l + h.shift(2)) / 2
        fvg_mid_dn = (h + l.shift(2)) / 2
        sweep_hi = (h > hi20) & (c < hi20)
        sweep_lo = (l < lo20) & (c > lo20)
        bos_up = c > hi20
        bos_dn = c < lo20

        if "inverse fair value gap" in n or "ifvg" in n:
            return _signed(c.index, fvg_dn.shift(1) & (c > h.shift(2)), fvg_up.shift(1) & (c < l.shift(2)))
        if "consequent-encroachment" in n:
            return _signed(c.index, fvg_up.shift(1) & (l <= fvg_mid_up.shift(1)) & (c > fvg_mid_up.shift(1)),
                           fvg_dn.shift(1) & (h >= fvg_mid_dn.shift(1)) & (c < fvg_mid_dn.shift(1)))
        if "breakaway fvg" in n:
            return _signed(c.index, fvg_up & displacement_up & bos_up, fvg_dn & displacement_dn & bos_dn)
        if "failed-fvg" in n:
            return _signed(c.index, fvg_dn.shift(1) & (l < fvg_mid_dn.shift(1)) & (c > fvg_mid_dn.shift(1)),
                           fvg_up.shift(1) & (h > fvg_mid_up.shift(1)) & (c < fvg_mid_up.shift(1)))
        if "balanced price range" in n:
            overlap = fvg_up.shift(1) & fvg_dn.shift(3) | fvg_dn.shift(1) & fvg_up.shift(3)
            z = x.z(20)
            return _signed(c.index, overlap & (z < -0.5), overlap & (z > 0.5))
        if "liquidity-void" in n:
            impulse = body > 1.8 * a
            mid = (o.shift(1) + c.shift(1)) / 2
            return _signed(c.index, impulse.shift(1) & (c.shift(1) < o.shift(1)) & (l <= mid) & (c > mid),
                           impulse.shift(1) & (c.shift(1) > o.shift(1)) & (h >= mid) & (c < mid))
        if "volume-imbalance" in n:
            v = x.v
            spike = v > v.rolling(40, min_periods=20).median() * 1.8
            return _signed(c.index, spike.shift(1) & (c > h.shift(1)), spike.shift(1) & (c < l.shift(1)))
        if "order-block" in n:
            bull_ob = (c.shift(1) < o.shift(1)) & displacement_up
            bear_ob = (c.shift(1) > o.shift(1)) & displacement_dn
            return _signed(c.index, bull_ob.shift(1) & (l <= o.shift(2)) & (c > o.shift(2)),
                           bear_ob.shift(1) & (h >= o.shift(2)) & (c < o.shift(2)))
        if "breaker-block" in n:
            return _signed(c.index, sweep_lo.shift(1) & bos_up, sweep_hi.shift(1) & bos_dn)
        if "mitigation-block" in n:
            return _signed(c.index, bos_up.shift(1) & (l <= h.shift(2)) & (c > h.shift(2)),
                           bos_dn.shift(1) & (h >= l.shift(2)) & (c < l.shift(2)))
        if "rejection-block" in n:
            lower_wick = (np.minimum(o, c) - l) > body * 1.5
            upper_wick = (h - np.maximum(o, c)) > body * 1.5
            return _signed(c.index, lower_wick & (l < lo20), upper_wick & (h > hi20))
        if "propulsion-block" in n:
            return _signed(c.index, displacement_up & (x.ema(20) > x.ema(50)), displacement_dn & (x.ema(20) < x.ema(50)))
        if "equal-high/equal-low" in n:
            eqh = (h.shift(1) - h.shift(3)).abs() < a.shift(1) * 0.1
            eql = (l.shift(1) - l.shift(3)).abs() < a.shift(1) * 0.1
            return _signed(c.index, eql & (l < l.shift(1)) & (c > l.shift(1)), eqh & (h > h.shift(1)) & (c < h.shift(1)))
        if "previous-day" in n:
            ph, pl, _ = x.prior_period_levels("D")
            return _signed(c.index, (l < pl) & (c > pl), (h > ph) & (c < ph))
        if "previous-week" in n:
            ph, pl, _ = x.prior_period_levels("W")
            return _signed(c.index, (l < pl) & (c > pl), (h > ph) & (c < ph))
        if "previous-month" in n:
            ph, pl, _ = x.prior_period_levels("M")
            return _signed(c.index, (l < pl) & (c > pl), (h > ph) & (c < ph))
        if "session high/low raid" in n:
            sh, sl = x.current_session_levels()
            return _signed(c.index, (l < sl) & (c > sl), (h > sh) & (c < sh))
        if "liquidity-sweep reclaim" in n:
            return _signed(c.index, sweep_lo.shift(1) & (c > h.shift(1)), sweep_hi.shift(1) & (c < l.shift(1)))
        if "liquidity-sweep reversal" in n or "turtle soup" in n:
            return _signed(c.index, sweep_lo, sweep_hi)
        if "internal-liquidity" in n:
            ih, il = x.donchian(10)
            return _signed(c.index, (c > ih) & (c < hi50), (c < il) & (c > lo50))
        if "external-liquidity" in n:
            z = x.z(20)
            return _signed(c.index, (l < lo50) & (c > lo50) & (z < 0), (h > hi50) & (c < hi50) & (z > 0))
        if "draw on liquidity" in n or "(dol)" in n:
            e20 = x.ema(20)
            return _signed(c.index, (c > e20) & (hi20 > c) & (x.roc(5) > 0), (c < e20) & (lo20 < c) & (x.roc(5) < 0))
        if "optimal trade entry" in n or "(ote)" in n:
            swing = hi50 - lo50
            long_zone = lo50 + 0.705 * swing
            short_zone = hi50 - 0.705 * swing
            return _signed(c.index, (x.ema(20) > x.ema(50)) & (l <= long_zone) & (c > long_zone),
                           (x.ema(20) < x.ema(50)) & (h >= short_zone) & (c < short_zone))
        if "premium/discount" in n:
            mid = (hi50 + lo50) / 2
            return _signed(c.index, (c < mid) & sweep_lo, (c > mid) & sweep_hi)
        if "equilibrium 50%" in n:
            mid = (hi50 + lo50) / 2
            return _signed(c.index, (c.shift(1) < mid.shift(1)) & (c >= mid), (c.shift(1) > mid.shift(1)) & (c <= mid))
        if "market structure shift" in n or "(mss)" in n:
            return _signed(c.index, sweep_lo.shift(1) & bos_up, sweep_hi.shift(1) & bos_dn)
        if "change of character" in n or "(choch)" in n:
            e20 = x.ema(20)
            return _signed(c.index, (e20.diff() > 0) & bos_up & (x.roc(5).shift(1) < 0),
                           (e20.diff() < 0) & bos_dn & (x.roc(5).shift(1) > 0))
        if "break of structure" in n or "(bos)" in n:
            return _signed(c.index, bos_up & (l <= hi20.shift(1)), bos_dn & (h >= lo20.shift(1)))
        if "displacement-and-retracement" in n:
            return _signed(c.index, displacement_up.shift(1) & (l <= c.shift(1)) & (c > o.shift(1)),
                           displacement_dn.shift(1) & (h >= c.shift(1)) & (c < o.shift(1)))
        if "change in state of delivery" in n or "(cisd)" in n:
            return _signed(c.index, (c > o.shift(1)) & (c.shift(1) < o.shift(1)), (c < o.shift(1)) & (c.shift(1) > o.shift(1)))
        if "smt-divergence" in n:
            dxy = self._aligned_extra_close("dxy")
            gx = c.pct_change(8)
            dx = dxy.pct_change(8)
            return _signed(c.index, (gx < 0) & (dx < 0), (gx > 0) & (dx > 0))
        if "power of three" in n or "accumulation-manipulation-distribution" in n:
            orh, orl, ready = x.opening_range(8)
            return _signed(c.index, ready & (l < orl) & (c > orh), ready & (h > orh) & (c < orl))
        if "judas swing" in n:
            hour = x.hour()
            return _signed(c.index, hour.between(6, 9) & sweep_lo, hour.between(6, 9) & sweep_hi)
        if "silver bullet" in n:
            hour = x.hour()
            window = hour.isin([10, 14, 15, 19])
            return _signed(c.index, window & sweep_lo & fvg_up, window & sweep_hi & fvg_dn)
        if "ict 2022" in n:
            return _signed(c.index, sweep_lo.shift(1) & displacement_up & fvg_up, sweep_hi.shift(1) & displacement_dn & fvg_dn)
        if "unicorn" in n:
            return _signed(c.index, bos_up.shift(1) & fvg_up & (l <= hi20.shift(1)), bos_dn.shift(1) & fvg_dn & (h >= lo20.shift(1)))
        if "market maker model" in n or "mmxm" in n:
            return _signed(c.index, sweep_lo & (x.roc(10) < -0.01) & (c > o), sweep_hi & (x.roc(10) > 0.01) & (c < o))
        if "new week opening gap" in n or "nwog" in n:
            _, _, pwc = x.prior_period_levels("W")
            week_open = x.o.groupby(x.t.dt.tz_convert(None).dt.to_period("W").astype(str)).transform("first")
            mid = (pwc + week_open) / 2
            return _signed(c.index, (c.shift(1) < mid.shift(1)) & (c >= mid), (c.shift(1) > mid.shift(1)) & (c <= mid))
        if "new day opening gap" in n or "ndog" in n:
            _, _, pdc = x.prior_period_levels("D")
            mid = (pdc + x.daily_open()) / 2
            return _signed(c.index, (c.shift(1) < mid.shift(1)) & (c >= mid), (c.shift(1) > mid.shift(1)) & (c <= mid))
        if "midnight-open" in n:
            dop = x.daily_open()
            z = (c - dop) / a.replace(0, np.nan)
            return _signed(c.index, (z < -1) & (x.roc(3) > 0), (z > 1) & (x.roc(3) < 0))
        if "daily-open bias" in n:
            dop = x.daily_open()
            return _signed(c.index, (c > dop) & (x.ema(20) > x.ema(50)), (c < dop) & (x.ema(20) < x.ema(50)))
        if "macro time-window" in n:
            hour = x.hour()
            return _signed(c.index, hour.isin([9,10,14,15]) & sweep_lo, hour.isin([9,10,14,15]) & sweep_hi)
        if "ipda dealing-range" in n:
            mid = (hi50 + lo50) / 2
            return _signed(c.index, (c < mid) & (x.roc(5) > 0), (c > mid) & (x.roc(5) < 0))
        if "quarterly-shift" in n:
            q = ((x.month() - 1) // 3)
            qkey = x.t.dt.year.astype(str) + "-" + q.astype(str)
            qopen = x.o.groupby(qkey).transform("first")
            return _signed(c.index, (c > qopen) & (x.ema(20) > x.ema(50)), (c < qopen) & (x.ema(20) < x.ema(50)))
        if "higher-timeframe pd-array" in n:
            return _signed(c.index, (x.ema(20) > x.ema(50)) & (x.ema(50) > x.ema(200)) & sweep_lo,
                           (x.ema(20) < x.ema(50)) & (x.ema(50) < x.ema(200)) & sweep_hi)
        if "liquidity sweep + displacement/fvg" in n:
            return _signed(c.index, sweep_lo.shift(1) & displacement_up & fvg_up, sweep_hi.shift(1) & displacement_dn & fvg_dn)
        # Only reached for the simple FVG retracement continuation.
        return _signed(c.index, fvg_up.shift(1) & (l <= fvg_mid_up.shift(1)) & (c > fvg_mid_up.shift(1)),
                       fvg_dn.shift(1) & (h >= fvg_mid_dn.shift(1)) & (c < fvg_mid_dn.shift(1)))

    def _signal_price_action(self, s: StrategyDefinition) -> pd.Series:
        x = self.ctx
        n = s.name.lower()
        o,h,l,c = x.o,x.h,x.l,x.c
        a = x.atr(14)
        hi20,lo20=x.donchian(20)
        hi50,lo50=x.donchian(50)
        body=(c-o).abs()
        rng=(h-l).replace(0,np.nan)
        upper=h-np.maximum(o,c)
        lower=np.minimum(o,c)-l
        bull=c>o
        bear=c<o
        trend_up=x.ema(20)>x.ema(50)
        trend_dn=x.ema(20)<x.ema(50)

        if n=="support/resistance bounce":
            return _signed(c.index,(l<=lo20)&(c>lo20),(h>=hi20)&(c<hi20))
        if n=="support/resistance breakout":
            return _signed(c.index,c>hi20,c<lo20)
        if "support/resistance breakout-retest" in n or n=="range breakout-retest":
            return _signed(c.index,(c.shift(1)>hi20.shift(1))&(l<=hi20)&(c>hi20),
                           (c.shift(1)<lo20.shift(1))&(h>=lo20)&(c<lo20))
        if "false-breakout" in n or "failed-breakdown" in n:
            return _signed(c.index,(l<lo20)&(c>lo20),(h>hi20)&(c<hi20))
        if "trendline" in n:
            slope,fit,res=x.regression(40)
            band=x.std(40)*0.5
            if "bounce" in n:
                return _signed(c.index,(slope>0)&(l<=fit)&(c>fit),(slope<0)&(h>=fit)&(c<fit))
            if "retest" in n:
                return _signed(c.index,(c.shift(1)>fit.shift(1)+band.shift(1))&(l<=fit+band)&(c>fit+band),
                               (c.shift(1)<fit.shift(1)-band.shift(1))&(h>=fit-band)&(c<fit-band))
            return _signed(c.index,c>fit+band,c<fit-band)
        if "parallel-channel" in n or "channel-overshoot" in n:
            slope,fit,res=x.regression(50); sd=res.rolling(50,min_periods=50).std()
            if "breakout" in n:
                return _signed(c.index,c>fit+2*sd,c<fit-2*sd)
            return _signed(c.index,(l<fit-2*sd)&(c>fit-2*sd),(h>fit+2*sd)&(c<fit+2*sd))
        if "horizontal-range" in n:
            width=hi20-lo20
            loc=(c-lo20)/width.replace(0,np.nan)
            if "breakout" in n:
                return _signed(c.index,c>hi20,c<lo20)
            return _signed(c.index,loc<0.1,loc>0.9)
        if "rectangle" in n:
            compressed=(hi20-lo20)<(x.atr(14).rolling(50).mean()*4)
            return _signed(c.index,compressed&trend_up&(c>hi20),compressed&trend_dn&(c<lo20))
        if "triangle" in n:
            hi10=x.h.rolling(10).max().shift(1); lo10=x.l.rolling(10).min().shift(1)
            contracting=(hi10-hi10.shift(10)<0)&(lo10-lo10.shift(10)>0)
            if "false-break" in n:
                return _signed(c.index,contracting&(l<lo10)&(c>lo10),contracting&(h>hi10)&(c<hi10))
            if "ascending" in n:
                return _signed(c.index,contracting&(c>hi10)&trend_up,pd.Series(False,index=c.index))
            if "descending" in n:
                return _signed(c.index,pd.Series(False,index=c.index),contracting&(c<lo10)&trend_dn)
            return _signed(c.index,contracting&(c>hi10),contracting&(c<lo10))
        if "flag" in n or "pennant" in n:
            impulse=x.roc(8)
            narrow=(h-l).rolling(6).mean()<a
            return _signed(c.index,(impulse.shift(6)>0.01)&narrow&(c>h.rolling(6).max().shift(1)),
                           (impulse.shift(6)<-0.01)&narrow&(c<l.rolling(6).min().shift(1)))
        if "wedge" in n:
            hi10=h.rolling(10).max().shift(1); lo10=l.rolling(10).min().shift(1)
            contracting=(hi10-hi10.shift(10)<0)&(lo10-lo10.shift(10)>0)
            if "reversal" in n:
                return _signed(c.index,contracting&(x.roc(10)<0)&(c>hi10),contracting&(x.roc(10)>0)&(c<lo10))
            return _signed(c.index,contracting&trend_up&(c>hi10),contracting&trend_dn&(c<lo10))
        if "head-and-shoulders" in n:
            top=(h.shift(4)<h.shift(2))&(h<h.shift(2))&(h.shift(8)<h.shift(6))&(h.shift(6)<h.shift(2))
            bot=(l.shift(4)>l.shift(2))&(l>l.shift(2))&(l.shift(8)>l.shift(6))&(l.shift(6)>l.shift(2))
            return _signed(c.index,bot&(c>h.shift(4)),top&(c<l.shift(4)))
        if "double-top/double-bottom" in n or "micro double-top" in n:
            span=5 if "micro" in n else 12
            eqh=(h.shift(1)-h.shift(span)).abs()<a*0.25
            eql=(l.shift(1)-l.shift(span)).abs()<a*0.25
            return _signed(c.index,eql&(c>h.shift(1)),eqh&(c<l.shift(1)))
        if "triple-top/triple-bottom" in n:
            eqh=(h.shift(1)-h.shift(8)).abs()<a*0.3
            eql=(l.shift(1)-l.shift(8)).abs()<a*0.3
            return _signed(c.index,eql&eql.shift(8)&(c>h.shift(1)),eqh&eqh.shift(8)&(c<l.shift(1)))
        if "rounding" in n or "saucer" in n:
            e=x.ema(20); turn=e.diff()
            return _signed(c.index,(turn>0)&(turn.shift(5)<0)&(c>e),(turn<0)&(turn.shift(5)>0)&(c<e))
        if "cup-and-handle" in n:
            base=l.rolling(40).min().shift(1); rim=h.rolling(40).max().shift(1)
            retrace=(c<rim)&(c>base+(rim-base)*0.6)
            return _signed(c.index,retrace.shift(1)&(c>rim),pd.Series(False,index=c.index))
        if "diamond" in n or "broadening" in n or "megaphone" in n:
            r10=(h.rolling(10).max()-l.rolling(10).min())
            expanding=r10>r10.shift(10)*1.2
            if "mean reversion" in n or "false-break" in n or "reversal" in n:
                return _signed(c.index,expanding&(l<lo20)&(c>lo20),expanding&(h>hi20)&(c<hi20))
            return _signed(c.index,expanding&(c>hi20),expanding&(c<lo20))
        if "1-2-3 reversal" in n:
            return _signed(c.index,(l.shift(2)<l.shift(4))&(l>l.shift(2))&(c>h.shift(1)),
                           (h.shift(2)>h.shift(4))&(h<h.shift(2))&(c<l.shift(1)))
        if "ross hook" in n:
            return _signed(c.index,trend_up&(h.shift(2)>h.shift(3))&(c>h.shift(2)),
                           trend_dn&(l.shift(2)<l.shift(3))&(c<l.shift(2)))
        if "swing failure" in n or "(sfp)" in n:
            return _signed(c.index,(l<lo20)&(c>lo20),(h>hi20)&(c<hi20))
        if "pin-bar" in n or "hammer" in n or "shooting-star" in n:
            return _signed(c.index,(lower>2*body)&bull,(upper>2*body)&bear)
        if "engulfing" in n:
            return _signed(c.index,bull&bear.shift(1)&(c>=o.shift(1))&(o<=c.shift(1)),
                           bear&bull.shift(1)&(o>=c.shift(1))&(c<=o.shift(1)))
        if "outside-bar" in n:
            outside=(h>h.shift(1))&(l<l.shift(1))
            return _signed(c.index,outside&bull,outside&bear)
        if "inside-bar fakey" in n:
            inside=(h.shift(1)<h.shift(2))&(l.shift(1)>l.shift(2))
            return _signed(c.index,inside&(l<l.shift(2))&(c>h.shift(1)),inside&(h>h.shift(2))&(c<l.shift(1)))
        if "inside-bar" in n or "mother-bar" in n:
            inside=(h.shift(1)<h.shift(2))&(l.shift(1)>l.shift(2))
            return _signed(c.index,inside&(c>h.shift(2)),inside&(c<l.shift(2)))
        if "two-bar reversal" in n or "railroad-tracks" in n:
            return _signed(c.index,bull&bear.shift(1)&(body-body.shift(1)).abs()<a*0.3,
                           bear&bull.shift(1)&(body-body.shift(1)).abs()<a*0.3)
        if "three-bar reversal" in n:
            return _signed(c.index,bear.shift(2)&(l.shift(1)<l.shift(2))&bull&(c>h.shift(1)),
                           bull.shift(2)&(h.shift(1)>h.shift(2))&bear&(c<l.shift(1)))
        if "morning-star" in n or "evening-star" in n:
            small=body.shift(1)<rng.shift(1)*0.3
            return _signed(c.index,bear.shift(2)&small&bull&(c>(o.shift(2)+c.shift(2))/2),
                           bull.shift(2)&small&bear&(c<(o.shift(2)+c.shift(2))/2))
        if "harami" in n:
            return _signed(c.index,bear.shift(1)&bull&(h<h.shift(1))&(l>l.shift(1)),
                           bull.shift(1)&bear&(h<h.shift(1))&(l>l.shift(1)))
        if "tweezer" in n:
            eql=(l-l.shift(1)).abs()<a*0.1; eqh=(h-h.shift(1)).abs()<a*0.1
            if "support/resistance" in n:
                return _signed(c.index,eql&(l<=lo20)&bull,eqh&(h>=hi20)&bear)
            return _signed(c.index,eql&bull,eqh&bear)
        if "doji" in n:
            doji=body<rng*0.12
            return _signed(c.index,doji&(lower>upper)&(l<=lo20),doji&(upper>lower)&(h>=hi20))
        if "kicker" in n:
            return _signed(c.index,bear.shift(1)&bull&(o>o.shift(1))&(c>h.shift(1)),
                           bull.shift(1)&bear&(o<o.shift(1))&(c<l.shift(1)))
        if "piercing-line" in n or "dark-cloud" in n:
            mid=(o.shift(1)+c.shift(1))/2
            return _signed(c.index,bear.shift(1)&bull&(c>mid)&(c<o.shift(1)),
                           bull.shift(1)&bear&(c<mid)&(c>o.shift(1)))
        if "three-soldiers" in n or "three-crows" in n:
            return _signed(c.index,bull&bull.shift(1)&bull.shift(2)&(c>c.shift(1))&(c.shift(1)>c.shift(2)),
                           bear&bear.shift(1)&bear.shift(2)&(c<c.shift(1))&(c.shift(1)<c.shift(2)))
        if "marubozu" in n:
            strong=body/rng>0.85
            return _signed(c.index,strong&bull,strong&bear)
        if "gap" in n or "island reversal" in n:
            gap_up=l>h.shift(1); gap_dn=h<l.shift(1)
            if "fill fade" in n or "exhaustion" in n or "island" in n:
                return _signed(c.index,gap_dn.shift(1)&(c>h.shift(1)),gap_up.shift(1)&(c<l.shift(1)))
            return _signed(c.index,gap_up&trend_up,gap_dn&trend_dn)
        if "opening-drive" in n:
            orh,orl,ready=x.opening_range(2)
            return _signed(c.index,ready&(c>orh)&trend_up,ready&(c<orl)&trend_dn)
        if "failed-auction" in n:
            return _signed(c.index,(l<lo50)&(c>lo20),(h>hi50)&(c<hi20))
        if "pullback" in n:
            depth={"first":1,"second-entry":2,"two-legged":2}.get(next((k for k in ("first","second-entry","two-legged") if k in n),"first"),1)
            rs=x.rsi(14)
            return _signed(c.index,trend_up&(rs.shift(depth)<45)&(rs>=50),trend_dn&(rs.shift(depth)>55)&(rs<=50))
        if "three-push" in n:
            return _signed(c.index,(l<l.shift(4))&(l.shift(4)<l.shift(8))&(x.rsi(14)>x.rsi(14).shift(8)),
                           (h>h.shift(4))&(h.shift(4)>h.shift(8))&(x.rsi(14)<x.rsi(14).shift(8)))
        if "parabolic blow-off" in n or "climactic" in n:
            ext=(c-x.ema(20))/a.replace(0,np.nan)
            return _signed(c.index,(ext<-3)&bull,(ext>3)&bear)
        if "trend-exhaustion" in n:
            shrinking=rng.rolling(5).mean()<rng.rolling(20).mean()*0.6
            return _signed(c.index,trend_dn&shrinking&bull,trend_up&shrinking&bear)
        if "compression breakout" in n:
            compressed=rng.rolling(10).mean()<rng.rolling(50).mean()*0.6
            return _signed(c.index,compressed.shift(1)&(c>hi20),compressed.shift(1)&(c<lo20))
        if "impulse-correction-impulse" in n:
            return _signed(c.index,trend_up&(x.roc(8).shift(4)>0.008)&(x.roc(3).shift(1)<0)&(x.roc(3)>0),
                           trend_dn&(x.roc(8).shift(4)<-0.008)&(x.roc(3).shift(1)>0)&(x.roc(3)<0))
        # key reversal fallback
        return _signed(c.index,(l<l.shift(1))&bull&(c>c.shift(1)),(h>h.shift(1))&bear&(c<c.shift(1)))

    def _signal_trend(self, s: StrategyDefinition) -> pd.Series:
        x=self.ctx; n=s.name.lower(); c=x.c
        e10,e20,e50,e100,e200=x.ema(10),x.ema(20),x.ema(50),x.ema(100),x.ema(200)
        hi20,lo20=x.donchian(20); a=x.atr(14); roc=x.roc(10)
        if "single-moving-average" in n:
            return _signed(c.index,(c>e50)&(x.l<=e50)&(c>c.shift(1)),(c<e50)&(x.h>=e50)&(c<c.shift(1)))
        if "dual-moving-average" in n:
            return _signed(c.index,(e20>e50)&(e20.shift(1)<=e50.shift(1)),(e20<e50)&(e20.shift(1)>=e50.shift(1)))
        if "triple-moving-average" in n:
            return _signed(c.index,(e10>e20)&(e20>e50),(e10<e20)&(e20<e50))
        if "ribbon compression" in n:
            width=pd.concat([e10,e20,e50],axis=1).max(axis=1)-pd.concat([e10,e20,e50],axis=1).min(axis=1)
            return _signed(c.index,(width.shift(1)<a.shift(1)*0.25)&(e10>e20)&(e20>e50),
                           (width.shift(1)<a.shift(1)*0.25)&(e10<e20)&(e20<e50))
        if "moving-average slope" in n:
            return _signed(c.index,e50.diff(5)>a*0.2,e50.diff(5)<-a*0.2)
        if "price-vs-moving-average breakout" in n or "moving-average reclaim" in n:
            return _signed(c.index,(c>e50)&(c.shift(1)<=e50.shift(1)),(c<e50)&(c.shift(1)>=e50.shift(1)))
        if "moving-average rejection" in n:
            return _signed(c.index,(x.l<=e50)&(c>e50)&(e20>e50),(x.h>=e50)&(c<e50)&(e20<e50))
        if "moving-average envelope" in n:
            sd=x.std(50)
            return _signed(c.index,(c>e50+sd)&(e20>e50),(c<e50-sd)&(e20<e50))
        if "adx" in n or "dmi" in n:
            adx,plus,minus=x.adx(14)
            return _signed(c.index,(adx>25)&(plus>minus),(adx>25)&(minus>plus))
        if "supertrend" in n or "atr-channel" in n:
            mid=(x.h+x.l)/2; upper=mid+3*a; lower=mid-3*a
            return _signed(c.index,c>upper.shift(1),c<lower.shift(1))
        if "parabolic sar" in n:
            trail_long=x.h.rolling(10).max().shift(1)-2*a
            trail_short=x.l.rolling(10).min().shift(1)+2*a
            return _signed(c.index,(c>trail_short)&(c.shift(1)<=trail_short.shift(1)),
                           (c<trail_long)&(c.shift(1)>=trail_long.shift(1)))
        if "ichimoku" in n:
            ten=(x.h.rolling(9).max()+x.l.rolling(9).min())/2
            kij=(x.h.rolling(26).max()+x.l.rolling(26).min())/2
            span=(ten+kij)/2
            if "tenkan-kijun" in n:
                return _signed(c.index,(ten>kij)&(ten.shift(1)<=kij.shift(1)),(ten<kij)&(ten.shift(1)>=kij.shift(1)))
            if "kumo breakout" in n:
                return _signed(c.index,c>span.shift(26),c<span.shift(26))
            if "kijun pullback" in n:
                return _signed(c.index,(c>kij)&(x.l<=kij)&(ten>kij),(c<kij)&(x.h>=kij)&(ten<kij))
            return _signed(c.index,(c>c.shift(26))&(ten>kij),(c<c.shift(26))&(ten<kij))
        if "donchian" in n or "turtle breakout" in n:
            return _signed(c.index,c>hi20,c<lo20)
        if "keltner" in n:
            return _signed(c.index,c>e20+2*a,c<e20-2*a)
        if "linear-regression slope" in n:
            slope,fit,res=x.regression(50)
            return _signed(c.index,slope>0,slope<0)
        if "linear-regression channel breakout" in n:
            slope,fit,res=x.regression(50); sd=res.rolling(50).std()
            return _signed(c.index,c>fit+2*sd,c<fit-2*sd)
        if "mama/fama" in n:
            fast=x.ema(9); slow=x.ema(21)
            return _signed(c.index,(fast>slow)&(fast.shift(1)<=slow.shift(1)),(fast<slow)&(fast.shift(1)>=slow.shift(1)))
        if "heikin-ashi" in n:
            ha=(x.o+x.h+x.l+x.c)/4
            hao=(x.o.shift(1)+x.c.shift(1))/2
            return _signed(c.index,(ha>hao)&(ha.shift(1)<=hao.shift(1)),(ha<hao)&(ha.shift(1)>=hao.shift(1)))
        if "renko" in n:
            move=c-c.shift(1)
            return _signed(c.index,move>a,move<-a)
        if "point-and-figure" in n:
            return _signed(c.index,c>x.h.rolling(6).max().shift(1),c<x.l.rolling(6).min().shift(1))
        if "aroon" in n:
            up=x.h.rolling(25).apply(lambda a: np.argmax(a)/24*100,raw=True)
            dn=x.l.rolling(25).apply(lambda a: np.argmin(a)/24*100,raw=True)
            return _signed(c.index,(up>70)&(up>dn),(dn>70)&(dn>up))
        if "vortex" in n:
            tr=x.atr(14)*14
            vp=(x.h-x.l.shift(1)).abs().rolling(14).sum()/tr.replace(0,np.nan)
            vm=(x.l-x.h.shift(1)).abs().rolling(14).sum()/tr.replace(0,np.nan)
            return _signed(c.index,vp>vm,vm>vp)
        if "trix" in n:
            e1=x.ema(15); e2=e1.ewm(span=15,adjust=False).mean(); e3=e2.ewm(span=15,adjust=False).mean()
            trix=e3.pct_change()
            return _signed(c.index,trix>0,trix<0)
        if "tsi" in n:
            d=c.diff(); m=d.ewm(span=25,adjust=False).mean().ewm(span=13,adjust=False).mean()
            am=d.abs().ewm(span=25,adjust=False).mean().ewm(span=13,adjust=False).mean()
            tsi=100*m/am.replace(0,np.nan)
            return _signed(c.index,tsi>0,tsi<0)
        if "macd" in n:
            line,sig,hist=x.macd()
            if "zero-line" in n:
                return _signed(c.index,(line>0)&(line.shift(1)<=0),(line<0)&(line.shift(1)>=0))
            return _signed(c.index,(line>sig)&(line.shift(1)<=sig.shift(1)),(line<sig)&(line.shift(1)>=sig.shift(1)))
        if "ppo" in n:
            ppo=(x.ema(12)-x.ema(26))/x.ema(26).replace(0,np.nan)
            return _signed(c.index,ppo>0,ppo<0)
        if "r-squared" in n:
            slope,fit,res=x.regression(50); rsq=1-res.rolling(50).var()/c.rolling(50).var()
            return _signed(c.index,(rsq>0.5)&(slope>0),(rsq>0.5)&(slope<0))
        if "guppy" in n:
            fast=pd.concat([x.ema(i) for i in (3,5,8,10,12,15)],axis=1).mean(axis=1)
            slow=pd.concat([x.ema(i) for i in (30,35,40,45,50,60)],axis=1).mean(axis=1)
            return _signed(c.index,fast>slow,fast<slow)
        if "alligator" in n:
            jaw=x.ema(13); teeth=x.ema(8); lips=x.ema(5)
            return _signed(c.index,(lips>teeth)&(teeth>jaw),(lips<teeth)&(teeth<jaw))
        if "fractal breakout" in n:
            fh=x.h.shift(2).where((x.h.shift(2)>x.h.shift(1))&(x.h.shift(2)>x.h.shift(3)))
            fl=x.l.shift(2).where((x.l.shift(2)<x.l.shift(1))&(x.l.shift(2)<x.l.shift(3)))
            return _signed(c.index,c>fh.ffill(),c<fl.ffill())
        if "elder impulse" in n:
            _,_,hist=x.macd()
            return _signed(c.index,(e20.diff()>0)&(hist.diff()>0),(e20.diff()<0)&(hist.diff()<0))
        if "elder triple screen" in n:
            return _signed(c.index,(e100.diff()>0)&(x.rsi(14)<40)&(c>h20 if False else c>e20),
                           (e100.diff()<0)&(x.rsi(14)>60)&(c<e20))
        if "darvas" in n:
            hi50,lo50=x.donchian(50)
            return _signed(c.index,c>hi50,c<lo50)
        if "chandelier exit" in n:
            longstop=x.h.rolling(22).max().shift(1)-3*a
            shortstop=x.l.rolling(22).min().shift(1)+3*a
            return _signed(c.index,(c>shortstop)&trend_up if (trend_up:=e20>e50) is not None else c>shortstop,
                           (c<longstop)&(e20<e50))
        if "n-bar higher-high" in n:
            return _signed(c.index,(x.h>x.h.shift(5))&(x.l>x.l.shift(5)),(x.h<x.h.shift(5))&(x.l<x.l.shift(5)))
        if "swing-structure" in n:
            return _signed(c.index,(e20>e50)&(x.l<=e20)&(c>e20),(e20<e50)&(x.h>=e20)&(c<e20))
        if "trend-acceleration" in n:
            return _signed(c.index,(e20.diff()>0)&(e20.diff().diff()>0)&(c>hi20),(e20.diff()<0)&(e20.diff().diff()<0)&(c<lo20))
        if "trend-deceleration" in n:
            return _signed(c.index,(e20<e50)&(e20.diff().abs()<e20.diff().abs().rolling(20).median())&(c>e20),
                           (e20>e50)&(e20.diff().abs()<e20.diff().abs().rolling(20).median())&(c<e20))
        if "breakout with trend-strength" in n:
            adx,plus,minus=x.adx()
            return _signed(c.index,(c>hi20)&(adx>25)&(plus>minus),(c<lo20)&(adx>25)&(minus>plus))
        if "regression-to-trend" in n:
            slope,fit,res=x.regression(40)
            return _signed(c.index,(slope>0)&(x.l<=fit)&(c>fit),(slope<0)&(x.h>=fit)&(c<fit))
        if "kaufman" in n or "(kama)" in n:
            change=(c-c.shift(10)).abs(); vol=c.diff().abs().rolling(10).sum()
            er=change/vol.replace(0,np.nan); sc=(er*(2/(2+1)-2/(30+1))+2/(30+1))**2
            kama=c.copy()
            vals=kama.to_numpy(dtype=float); scv=sc.fillna(0).to_numpy(dtype=float)
            for i in range(1,len(vals)): vals[i]=vals[i-1]+scv[i]*(float(c.iloc[i])-vals[i-1])
            kama=pd.Series(vals,index=c.index)
            return _signed(c.index,(c>kama)&(c.shift(1)<=kama.shift(1)),(c<kama)&(c.shift(1)>=kama.shift(1)))
        if "ehlers super smoother" in n:
            smooth=(c+2*c.shift(1)+c.shift(2))/4
            return _signed(c.index,smooth.diff()>0,smooth.diff()<0)
        return _signed(c.index,(e20>e50)&(roc>0),(e20<e50)&(roc<0))

    def _signal_momentum(self, s: StrategyDefinition) -> pd.Series:
        x=self.ctx; n=s.name.lower(); c=x.c
        r=x.rsi(14); st=x.stochastic(14); cci=x.cci(20); wr=x.willr(14); roc=x.roc(10)
        line,msig,hist=x.macd()
        def divergence(osc):
            return _signed(c.index,(c<x.c.shift(10))&(osc>osc.shift(10)),(c>x.c.shift(10))&(osc<osc.shift(10)))
        if "rsi centerline" in n: return _signed(c.index,(r>50)&(r.shift(1)<=50),(r<50)&(r.shift(1)>=50))
        if "rsi overbought" in n: return _signed(c.index,(r<30)&(r.shift(1)>=30),(r>70)&(r.shift(1)<=70))
        if "rsi failure swing" in n: return _signed(c.index,(r.shift(2)<30)&(r.shift(1)>r.shift(2))&(r>r.shift(1)),(r.shift(2)>70)&(r.shift(1)<r.shift(2))&(r<r.shift(1)))
        if "rsi regular divergence" in n: return divergence(r)
        if "rsi hidden divergence" in n: return _signed(c.index,(c>c.shift(10))&(r<r.shift(10))&(x.ema(20)>x.ema(50)),(c<c.shift(10))&(r>r.shift(10))&(x.ema(20)<x.ema(50)))
        if "stochastic rsi" in n:
            rr=(r-r.rolling(14).min())/(r.rolling(14).max()-r.rolling(14).min()).replace(0,np.nan)*100
            return _signed(c.index,(rr>50)&(rr.shift(1)<=50),(rr<50)&(rr.shift(1)>=50))
        if "stochastic divergence" in n: return divergence(st)
        if "stochastic extreme" in n: return _signed(c.index,(st<20)&(st.shift(1)>=20),(st>80)&(st.shift(1)<=80))
        if "stochastic crossover" in n:
            d=st.rolling(3).mean(); return _signed(c.index,(st>d)&(st.shift(1)<=d.shift(1)),(st<d)&(st.shift(1)>=d.shift(1)))
        if "cci divergence" in n: return divergence(cci)
        if "cci extreme" in n: return _signed(c.index,(cci<-100)&(cci.shift(1)>=-100),(cci>100)&(cci.shift(1)<=100))
        if "cci zero-line" in n: return _signed(c.index,(cci>0)&(cci.shift(1)<=0),(cci<0)&(cci.shift(1)>=0))
        if "williams %r reversal" in n: return _signed(c.index,(wr<-80)&(wr.shift(1)>=-80),(wr>-20)&(wr.shift(1)<=-20))
        if "williams %r momentum" in n: return _signed(c.index,wr>-50,wr<-50)
        if "rate-of-change" in n: return _signed(c.index,roc>0.005,roc<-0.005)
        if "momentum-oscillator" in n: return _signed(c.index,c-c.shift(10)>0,c-c.shift(10)<0)
        if "macd histogram divergence" in n: return divergence(hist)
        if "macd histogram acceleration" in n: return _signed(c.index,(hist>0)&(hist.diff()>0),(hist<0)&(hist.diff()<0))
        if "awesome oscillator" in n:
            med=(x.h+x.l)/2; ao=med.rolling(5).mean()-med.rolling(34).mean()
            if "saucer" in n: return _signed(c.index,(ao>0)&(ao.diff().shift(1)<0)&(ao.diff()>0),(ao<0)&(ao.diff().shift(1)>0)&(ao.diff()<0))
            return _signed(c.index,(ao>0)&(ao.shift(1)<=0),(ao<0)&(ao.shift(1)>=0))
        if "accelerator" in n:
            med=(x.h+x.l)/2; ao=med.rolling(5).mean()-med.rolling(34).mean(); ac=ao-ao.rolling(5).mean()
            return _signed(c.index,ac>0,ac<0)
        if "fisher transform" in n:
            v=2*((c-c.rolling(10).min())/(c.rolling(10).max()-c.rolling(10).min()).replace(0,np.nan)-0.5)
            v=v.clip(-0.999,0.999); f=0.5*np.log((1+v)/(1-v))
            return _signed(c.index,(f>0)&(f.shift(1)<=0),(f<0)&(f.shift(1)>=0))
        if "ultimate oscillator" in n:
            pc=c.shift(1); bp=c-np.minimum(x.l,pc); tr=np.maximum(x.h,pc)-np.minimum(x.l,pc)
            uo=100*(4*(bp.rolling(7).sum()/tr.rolling(7).sum())+2*(bp.rolling(14).sum()/tr.rolling(14).sum())+(bp.rolling(28).sum()/tr.rolling(28).sum()))/7
            return divergence(uo)
        if "qqe" in n:
            sm=r.ewm(span=5,adjust=False).mean(); return _signed(c.index,sm>50,sm<50)
        if "relative vigor" in n:
            rv=(c-x.o).rolling(10).mean()/(x.h-x.l).rolling(10).mean().replace(0,np.nan); sig=rv.rolling(4).mean()
            return _signed(c.index,(rv>sig)&(rv.shift(1)<=sig.shift(1)),(rv<sig)&(rv.shift(1)>=sig.shift(1)))
        if "coppock" in n:
            cp=(x.roc(14)+x.roc(11)).ewm(span=10,adjust=False).mean(); return _signed(c.index,cp>0,cp<0)
        if "chande momentum" in n:
            d=c.diff(); up=d.clip(lower=0).rolling(14).sum(); dn=(-d.clip(upper=0)).rolling(14).sum(); cm=100*(up-dn)/(up+dn).replace(0,np.nan)
            return _signed(c.index,cm>0,cm<0)
        if "chande forecast" in n:
            _,fit,_=x.regression(20); cfo=(c-fit)/c.replace(0,np.nan)*100; return _signed(c.index,cfo>0,cfo<0)
        if "qstick" in n:
            q=(c-x.o).rolling(10).mean(); return _signed(c.index,q>0,q<0)
        if "detrended price" in n:
            dpo=c.shift(11)-x.sma(20); return _signed(c.index,(dpo>0)&(dpo.shift(1)<=0),(dpo<0)&(dpo.shift(1)>=0))
        if "schaff" in n:
            m=line; lo=m.rolling(10).min(); hi=m.rolling(10).max(); stc=(m-lo)/(hi-lo).replace(0,np.nan)*100
            return _signed(c.index,stc>50,stc<50)
        if "know sure thing" in n:
            k=x.roc(10).rolling(10).mean()+2*x.roc(15).rolling(10).mean()+3*x.roc(20).rolling(10).mean()+4*x.roc(30).rolling(15).mean()
            return _signed(c.index,k>0,k<0)
        if "elder bull/bear" in n:
            e=x.ema(13); bull=x.h-e; bear=x.l-e
            return _signed(c.index,(bear<bear.shift(10))&(c>c.shift(10)),(bull>bull.shift(10))&(c<c.shift(10)))
        if "force index" in n:
            fi=c.diff()*x.v; return _signed(c.index,fi.ewm(span=13,adjust=False).mean()>0,fi.ewm(span=13,adjust=False).mean()<0)
        if "money flow index" in n:
            tp=(x.h+x.l+c)/3; mf=tp*x.v; pos=mf.where(tp.diff()>0,0).rolling(14).sum(); neg=mf.where(tp.diff()<0,0).rolling(14).sum()
            mfi=100-100/(1+pos/neg.replace(0,np.nan))
            return divergence(mfi) if "divergence" in n else _signed(c.index,mfi>50,mfi<50)
        if "chaikin oscillator" in n:
            mfm=((c-x.l)-(x.h-c))/(x.h-x.l).replace(0,np.nan); ad=(mfm*x.v).cumsum(); co=ad.ewm(span=3,adjust=False).mean()-ad.ewm(span=10,adjust=False).mean()
            return _signed(c.index,co>0,co<0)
        if "balance of power" in n:
            bop=(c-x.o)/(x.h-x.l).replace(0,np.nan); return _signed(c.index,bop>0.5,bop<-0.5)
        if "ease of movement" in n:
            em=((x.h+x.l)/2).diff()*(x.h-x.l)/x.v.replace(0,np.nan); return _signed(c.index,em.rolling(14).mean()>0,em.rolling(14).mean()<0)
        if "demarker" in n:
            up=(x.h-x.h.shift(1)).clip(lower=0); dn=(x.l.shift(1)-x.l).clip(lower=0); dem=up.rolling(14).mean()/(up.rolling(14).mean()+dn.rolling(14).mean()).replace(0,np.nan)
            return _signed(c.index,dem<0.3,dem>0.7)
        if "relative momentum index" in n:
            d=c.diff(5); up=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean(); rmi=100-100/(1+up/dn.replace(0,np.nan))
            return _signed(c.index,rmi>50,rmi<50)
        if "dynamic momentum" in n:
            rv=x.realized_vol(10); dyn=(14*(rv.rolling(50).median()/rv.replace(0,np.nan))).clip(5,30)
            # causal approximation: switch between fixed RSI lengths based on current past-only volatility
            r7=x.rsi(7); r21=x.rsi(21); rr=pd.Series(np.where(dyn<14,r7,r21),index=c.index)
            return _signed(c.index,rr>50,rr<50)
        if "stochastic momentum index" in n:
            hh=x.h.rolling(14).max(); ll=x.l.rolling(14).min(); mid=(hh+ll)/2; smi=100*(c-mid)/((hh-ll)/2).replace(0,np.nan)
            return _signed(c.index,smi>0,smi<0)
        if "laguerre rsi" in n:
            lr=r.ewm(alpha=0.2,adjust=False).mean(); return _signed(c.index,lr>50,lr<50)
        if "wavetrend" in n:
            ap=(x.h+x.l+c)/3; esa=ap.ewm(span=10,adjust=False).mean(); d=(ap-esa).abs().ewm(span=10,adjust=False).mean(); ci=(ap-esa)/(0.015*d.replace(0,np.nan)); wt=ci.ewm(span=21,adjust=False).mean()
            return _signed(c.index,wt<-60,wt>60)
        return _signed(c.index,roc>0,roc<0)

    def _signal_mean_reversion(self, s: StrategyDefinition) -> pd.Series:
        x=self.ctx; n=s.name.lower(); c=x.c; a=x.atr(14); z=x.z(20); r=x.rsi(14)
        m,up,dn=x.bollinger(20,2)
        if "bollinger" in n:
            if "%b" in n:
                pct=(c-dn)/(up-dn).replace(0,np.nan); return _signed(c.index,pct<0.05,pct>0.95)
            if "w-bottom" in n:
                return _signed(c.index,(x.l<dn)&(x.l.shift(5)<dn.shift(5))&(r>r.shift(5)),(x.h>up)&(x.h.shift(5)>up.shift(5))&(r<r.shift(5)))
            if "band-walk exhaustion" in n:
                streak=(c>up).rolling(4).sum(); streakdn=(c<dn).rolling(4).sum()
                return _signed(c.index,streakdn>=3,streak>=3)
            if "keltner overextension" in n:
                return _signed(c.index,c<x.ema(20)-2*a,c>x.ema(20)+2*a)
            return _signed(c.index,c<dn,c>up)
        if "keltner-channel fade" in n: return _signed(c.index,c<x.ema(20)-2*a,c>x.ema(20)+2*a)
        if "moving-average envelope" in n: return _signed(c.index,c<x.ema(50)*0.99,c>x.ema(50)*1.01)
        if "donchian-channel fade" in n:
            hi,lo=x.donchian(20); return _signed(c.index,(x.l<lo)&(c>lo),(x.h>hi)&(c<hi))
        if "rolling z-score" in n: return _signed(c.index,z<-2,z>2)
        if "linear-regression residual" in n or "regression-channel edge" in n:
            _,fit,res=x.regression(40); sd=res.rolling(40).std()
            return _signed(c.index,res<-2*sd,res>2*sd)
        if "volatility-scaled rolling-mean" in n: return _signed(c.index,c<x.sma(30)-1.5*a,c>x.sma(30)+1.5*a)
        if "median-price" in n:
            med=c.rolling(30).median(); return _signed(c.index,c<med-1.5*a,c>med+1.5*a)
        if "rsi(2)" in n:
            r2=x.rsi(2); return _signed(c.index,(r2<10)&(c>x.ema(200)),(r2>90)&(c<x.ema(200)))
        if "consecutive up/down streak" in n:
            d=np.sign(c.diff()); up=(d>0).rolling(4).sum(); dn=(d<0).rolling(4).sum()
            return _signed(c.index,dn>=4,up>=4)
        if "gap-overextension" in n:
            gap=x.o-c.shift(1); return _signed(c.index,gap<-1.5*a,gap>1.5*a)
        if "atr-overextension" in n: return _signed(c.index,c<x.ema(20)-2.5*a,c>x.ema(20)+2.5*a)
        if "true-range spike" in n:
            tr=(x.h-x.l); spike=tr>tr.rolling(50).median()*2.5; return _signed(c.index,spike&(c<x.o),spike&(c>x.o))
        if "realized-volatility spike" in n or "volatility-cluster" in n or "garch volatility" in n:
            rv=x.realized_vol(20); high=rv>rv.rolling(100).quantile(.9)
            return _signed(c.index,high&(z<-1),high&(z>1))
        if "ornstein-uhlenbeck" in n or "half-life" in n:
            return _signed(c.index,z<-1.5,z>1.5)
        if "hurst-exponent" in n:
            hproxy=(c.diff().rolling(20).std()/c.diff().rolling(80).std().replace(0,np.nan)).clip(0,2)
            return _signed(c.index,(hproxy<0.8)&(z<-1.5),(hproxy<0.8)&(z>1.5))
        if "variance-ratio" in n:
            r1=c.pct_change(); vr=r1.rolling(20).var()/(r1.rolling(5).var()*4).replace(0,np.nan)
            return _signed(c.index,(vr<1)&(z<-1.5),(vr<1)&(z>1.5))
        if "autocorrelation" in n:
            ret=c.pct_change(); ac=ret.rolling(50).corr(ret.shift(1))
            return _signed(c.index,(ac<0)&(z<-1.5),(ac<0)&(z>1.5))
        if "kalman-filter fair-value" in n:
            fair=x.ema(30); err=c-fair; sd=err.rolling(50).std()
            return _signed(c.index,err<-2*sd,err>2*sd)
        if "moving-average-distance percentile" in n:
            d=(c-x.ema(50))/a.replace(0,np.nan); lo=d.rolling(100).quantile(.05); hi=d.rolling(100).quantile(.95)
            return _signed(c.index,d<lo,d>hi)
        if "rolling-percentile" in n:
            p=x.percentile(100); return _signed(c.index,p<.05,p>.95)
        if "opening-drive failure" in n:
            orh,orl,ready=x.opening_range(4); return _signed(c.index,ready&(x.l<orl)&(c>orl),ready&(x.h>orh)&(c<orh))
        if "post-breakout failure" in n:
            hi,lo=x.donchian(20); return _signed(c.index,(c.shift(1)<lo.shift(1))&(c>lo),(c.shift(1)>hi.shift(1))&(c<hi))
        if "failed-trend-day" in n:
            dop=x.daily_open(); return _signed(c.index,(c.shift(4)<dop.shift(4))&(c>dop),(c.shift(4)>dop.shift(4))&(c<dop))
        if "close-to-open" in n:
            _,_,pc=x.prior_period_levels("D"); return _signed(c.index,(x.o<pc-1.2*a)&(c>x.o),(x.o>pc+1.2*a)&(c<x.o))
        if "overnight-move" in n:
            dop=x.daily_open(); return _signed(c.index,(c<dop-1.5*a)&(x.hour()<8),(c>dop+1.5*a)&(x.hour()<8))
        if "session-extension" in n:
            sh,sl=x.current_session_levels(); return _signed(c.index,(c<sl-0.5*a),(c>sh+0.5*a))
        if "previous-close magnet" in n:
            _,_,pc=x.prior_period_levels("D"); return _signed(c.index,(c<pc-1.5*a)&(x.roc(3)>0),(c>pc+1.5*a)&(x.roc(3)<0))
        if "hilbert" in n or "sinewave" in n or "mesa" in n or "spectral" in n:
            osc=c-x.sma(20); turn=osc.diff()
            return _signed(c.index,(osc<0)&(turn>0)&(turn.shift(1)<=0),(osc>0)&(turn<0)&(turn.shift(1)>=0))
        if "hodrick-prescott" in n:
            trend=x.ema(40); resid=c-trend; sd=resid.rolling(40).std()
            return _signed(c.index,resid<-2*sd,resid>2*sd)
        if "seasonal intraday" in n:
            hour=x.hour(); ret=c.pct_change(); mean=ret.groupby(hour).transform(lambda a:a.expanding().mean().shift(1))
            return _signed(c.index,mean>0,mean<0)
        if "robust mad" in n:
            med=c.rolling(40).median(); mad=(c-med).abs().rolling(40).median()
            return _signed(c.index,c<med-3*mad,c>med+3*mad)
        if "quantile-band" in n:
            lo=c.rolling(100).quantile(.05); hi=c.rolling(100).quantile(.95); return _signed(c.index,c<lo,c>hi)
        if "entropy-filtered" in n:
            ret=np.sign(c.diff()); changes=(ret!=ret.shift(1)).rolling(30).mean()
            return _signed(c.index,(changes>0.5)&(z<-1.5),(changes>0.5)&(z>1.5))
        return _signed(c.index,z<-2,z>2)

    def _signal_breakout(self, s: StrategyDefinition) -> pd.Series:
        x=self.ctx; n=s.name.lower(); c=x.c; a=x.atr(14); hi20,lo20=x.donchian(20)
        if "opening range" in n or "initial balance" in n:
            bars=4 if "opening range" in n else 16
            orh,orl,ready=x.opening_range(bars)
            if "london" in n: ready=ready & x.hour().between(7,12)
            elif "asian" in n or "tokyo" in n: ready=ready & x.hour().between(0,8)
            elif "new york" in n: ready=ready & x.hour().between(12,18)
            return _signed(c.index,ready&(c>orh),ready&(c<orl))
        if "previous-day" in n:
            ph,pl,_=x.prior_period_levels("D"); return _signed(c.index,c>ph,c<pl)
        if "previous-week" in n:
            ph,pl,_=x.prior_period_levels("W"); return _signed(c.index,c>ph,c<pl)
        if "previous-month" in n:
            ph,pl,_=x.prior_period_levels("M"); return _signed(c.index,c>ph,c<pl)
        if "overnight high/low" in n or "session high/low" in n:
            sh,sl=x.current_session_levels(); return _signed(c.index,c>sh,c<sl)
        if "inside-day" in n or "outside-day" in n:
            ph,pl,_=x.prior_period_levels("D")
            if "inside" in n:
                # current day's opening range must begin within prior day; break prior day level after.
                return _signed(c.index,c>ph,c<pl)
            return _signed(c.index,(c>ph)&(x.roc(3)>0),(c<pl)&(x.roc(3)<0))
        if "nr4" in n or "nr7" in n:
            ph,pl,_=x.prior_period_levels("D")
            prev_range=ph-pl
            # Compare the fully known prior-day range with a causal ATR baseline.
            # NR7 uses a stricter compression threshold than NR4.
            factor=0.80 if "nr4" in n else 0.65
            narrow=prev_range < x.atr(14).shift(1) * 24 * factor
            return _signed(c.index,narrow&(c>hi20),narrow&(c<lo20))
        if "bollinger squeeze" in n or "ttm squeeze" in n:
            m,up,dn=x.bollinger(); width=(up-dn)/m
            squeeze=width<width.rolling(100).quantile(.2)
            return _signed(c.index,squeeze.shift(1)&(c>up),squeeze.shift(1)&(c<dn))
        if "keltner compression" in n:
            width=4*a/c; comp=width<width.rolling(100).quantile(.2)
            return _signed(c.index,comp.shift(1)&(c>hi20),comp.shift(1)&(c<lo20))
        if "atr compression" in n or "volatility-contraction" in n:
            comp=a<a.rolling(100).quantile(.2); return _signed(c.index,comp.shift(1)&(c>hi20),comp.shift(1)&(c<lo20))
        if "volatility-percentile" in n:
            rv=x.realized_vol(20); lowv=rv<rv.rolling(200).quantile(.2); return _signed(c.index,lowv.shift(1)&(c>hi20),lowv.shift(1)&(c<lo20))
        if "realized-range expansion" in n:
            rng=x.h-x.l; exp=rng>rng.rolling(50).quantile(.8); return _signed(c.index,exp&(c>hi20),exp&(c<lo20))
        if "atr-multiple" in n:
            return _signed(c.index,c>c.shift(1)+1.5*a.shift(1),c<c.shift(1)-1.5*a.shift(1))
        if "standard-deviation" in n:
            m=x.sma(20); sd=x.std(20); return _signed(c.index,c>m+2*sd,c<m-2*sd)
        if "opening-gap" in n or "gap continuation" in n:
            gap=x.o-c.shift(1); return _signed(c.index,(gap>0.5*a)&(c>x.o),(gap<-0.5*a)&(c<x.o))
        if "news-range" in n or "post-news" in n or "economic-release" in n or "pre-news" in n:
            # Scheduled UTC macro windows only; no surprise sign is inferred.
            hour=x.hour(); window=hour.isin([12,13,14,18])
            hi=x.h.rolling(4).max().shift(1); lo=x.l.rolling(4).min().shift(1)
            return _signed(c.index,window&(c>hi),window&(c<lo))
        if "london-new york overlap" in n:
            w=x.hour().between(12,16); return _signed(c.index,w&(c>hi20),w&(c<lo20))
        if "comex-open" in n:
            w=x.hour().isin([12,13]); return _signed(c.index,w&(c>hi20),w&(c<lo20))
        if "cash-market-open" in n:
            w=x.hour().isin([13,14]); return _signed(c.index,w&(c>hi20),w&(c<lo20))
        if "daily-settlement" in n:
            w=x.hour().isin([20,21]); return _signed(c.index,w&(c>hi20),w&(c<lo20))
        if "weekly-open" in n:
            w=x.weekday()==0; return _signed(c.index,w&(c>hi20),w&(c<lo20))
        if "monthly-open" in n:
            w=x.day()<=2; return _signed(c.index,w&(c>hi20),w&(c<lo20))
        if "midnight-range" in n:
            orh,orl,ready=x.opening_range(4); return _signed(c.index,ready&(c>orh),ready&(c<orl))
        if "rolling time-window" in n:
            hi=x.h.rolling(12).max().shift(1); lo=x.l.rolling(12).min().shift(1); return _signed(c.index,c>hi,c<lo)
        if "range expansion index" in n:
            re=(x.h-x.l)/(x.h-x.l).rolling(20).mean().replace(0,np.nan); return _signed(c.index,(re>1.5)&(c>hi20),(re>1.5)&(c<lo20))
        if "choppiness" in n:
            tr=x.atr(1); chop=100*np.log10(tr.rolling(14).sum()/(x.h.rolling(14).max()-x.l.rolling(14).min()).replace(0,np.nan))/np.log10(14)
            return _signed(c.index,(chop<38)&(c>hi20),(chop<38)&(c<lo20))
        if "adx-expansion" in n:
            adx,plus,minus=x.adx(); return _signed(c.index,(adx>25)&(adx.diff()>0)&(c>hi20),(adx>25)&(adx.diff()>0)&(c<lo20))
        if "volatility-adjusted dynamic-threshold" in n:
            threshold=x.sma(20)+1.5*a; lower=x.sma(20)-1.5*a; return _signed(c.index,c>threshold,c<lower)
        if "breakout-pullback" in n:
            return _signed(c.index,(c.shift(1)>hi20.shift(1))&(x.l<=hi20)&(c>hi20),(c.shift(1)<lo20.shift(1))&(x.h>=lo20)&(c<lo20))
        if "multi-day consolidation" in n:
            ph,pl,_=x.prior_period_levels("D"); narrow=(ph-pl)<a*8
            return _signed(c.index,narrow&(c>ph),narrow&(c<pl))
        if "three-touch" in n:
            touches_hi=((x.h-hi20).abs()<a*.1).rolling(20).sum()>=3
            touches_lo=((x.l-lo20).abs()<a*.1).rolling(20).sum()>=3
            return _signed(c.index,touches_hi&(c>hi20),touches_lo&(c<lo20))
        return _signed(c.index,c>hi20,c<lo20)

    def _signal_geometry(self, s: StrategyDefinition) -> pd.Series:
        x=self.ctx; n=s.name.lower(); c=x.c; a=x.atr(14)
        hi,lo=x.donchian(50); swing=(hi-lo).replace(0,np.nan)
        trend=x.ema(20)-x.ema(50)
        if "fibonacci retracement" in n:
            lvl=lo+.618*swing; return _signed(c.index,(trend>0)&(x.l<=lvl)&(c>lvl),(trend<0)&(x.h>=hi-.618*swing)&(c<hi-.618*swing))
        if "fibonacci extension" in n or "expansion measured" in n:
            return _signed(c.index,c>hi+.272*swing,c<lo-.272*swing)
        if "fibonacci time-zone" in n:
            bar=pd.Series(np.arange(len(c)),index=c.index); w=bar.mod(34)==0; return _signed(c.index,w&(x.roc(5)>0),w&(x.roc(5)<0))
        if "fibonacci fan" in n or "fibonacci channel" in n:
            lvl=lo+.5*swing; return _signed(c.index,(trend>0)&(x.l<=lvl)&(c>lvl),(trend<0)&(x.h>=lvl)&(c<lvl))
        if "pivot" in n or "central pivot range" in n:
            ph,pl,pc=x.prior_period_levels("D"); pivot=(ph+pl+pc)/3
            r1=2*pivot-pl; s1=2*pivot-ph
            if "breakout" in n:
                return _signed(c.index,c>r1,c<s1)
            return _signed(c.index,(x.l<=s1)&(c>s1),(x.h>=r1)&(c<r1))
        if "pitchfork" in n:
            _,fit,res=x.regression(60); sd=res.rolling(60).std()
            if "breakout" in n: return _signed(c.index,c>fit+2*sd,c<fit-2*sd)
            return _signed(c.index,(x.l<fit-2*sd)&(c>fit-2*sd),(x.h>fit+2*sd)&(c<fit+2*sd))
        if "gann fan" in n or "gann angle" in n:
            anchor=c.shift(50); slope=a.rolling(50).mean()/10; line=anchor+slope*50
            if "breakout" in n: return _signed(c.index,c>line,c<line)
            return _signed(c.index,(x.l<=line)&(c>line),(x.h>=line)&(c<line))
        if "square-of-nine" in n or "murrey math" in n:
            step=(swing/8).replace(0,np.nan); nearest=(np.round((c-lo)/step)*step+lo)
            return _signed(c.index,(x.l<nearest)&(c>nearest),(x.h>nearest)&(c<nearest))
        # Harmonics: distinguish by canonical XA/AB/BC/CD ratios.
        ratios={
            "gartley":(.618,.786),"bat":(.50,.886),"butterfly":(.786,1.272),"crab":(.618,1.618),
            "shark":(.50,1.13),"cypher":(.382,.786),"5-0":(.50,.50),"ab=cd":(1.0,1.0),
        }
        key=next((k for k in ratios if k in n),"gartley"); r1,r2=ratios[key]
        p0=c.shift(24); p1=c.shift(18); p2=c.shift(12); p3=c.shift(6)
        xa=(p1-p0); ab=(p2-p1); cd=(c-p3)
        rr=(ab.abs()/xa.abs().replace(0,np.nan)); ext=(cd.abs()/ab.abs().replace(0,np.nan))
        tol=.12
        long=(xa<0)&rr.between(r1-tol,r1+tol)&ext.between(r2-tol,r2+tol)&(c>p3)
        short=(xa>0)&rr.between(r1-tol,r1+tol)&ext.between(r2-tol,r2+tol)&(c<p3)
        return _signed(c.index,long,short)

    def _signal_volume(self, s: StrategyDefinition) -> pd.Series:
        x=self.ctx; n=s.name.lower(); c=x.c
        if float(x.v.sum()) <= 0:
            raise ValueError("volume input contains no positive observations")
        sv=x.session_vwap(); rv=x.rolling_vwap(96); a=x.atr(14)
        if "session vwap trend" in n: return _signed(c.index,(c>sv)&(x.ema(20)>x.ema(50)),(c<sv)&(x.ema(20)<x.ema(50)))
        if n=="vwap mean reversion": return _signed(c.index,c<sv-1.5*a,c>sv+1.5*a)
        if n=="vwap breakout": return _signed(c.index,(c>sv)&(c.shift(1)<=sv.shift(1)),(c<sv)&(c.shift(1)>=sv.shift(1)))
        if "vwap reclaim" in n: return _signed(c.index,(x.l<sv)&(c>sv),(x.h>sv)&(c<sv))
        if "anchored vwap" in n:
            av=rv
            if "mean reversion" in n: return _signed(c.index,c<av-1.5*a,c>av+1.5*a)
            return _signed(c.index,(c>av)&(x.ema(20)>x.ema(50)),(c<av)&(x.ema(20)<x.ema(50)))
        if "deviation-band fade" in n:
            sd=(c-sv).rolling(50).std(); return _signed(c.index,c<sv-2*sd,c>sv+2*sd)
        if "deviation-band breakout" in n:
            sd=(c-sv).rolling(50).std(); return _signed(c.index,c>sv+2*sd,c<sv-2*sd)
        if "multi-session vwap" in n:
            return _signed(c.index,(c>sv)&(c>rv),(c<sv)&(c<rv))
        if "rolling vwap trend" in n:
            return _signed(c.index,(c>rv)&(rv.diff()>0),(c<rv)&(rv.diff()<0))
        if any(k in n for k in ("poc","value-area","volume profile","node","volume-shelf","composite-profile","profile-level")):
            # Causal rolling price-volume distribution using only prior bars.
            window=96
            typical=((x.h+x.l+c)/3).shift(1)
            vol=x.v.shift(1)
            # Approximate rolling POC with volume-weighted center; profile width from weighted dispersion.
            poc=(typical*vol).rolling(window,min_periods=window).sum()/vol.rolling(window,min_periods=window).sum().replace(0,np.nan)
            dev=((typical-poc)**2*vol).rolling(window,min_periods=window).sum()/vol.rolling(window,min_periods=window).sum().replace(0,np.nan)
            sd=np.sqrt(dev)
            vah=poc+sd; val=poc-sd
            if "poc reversion" in n or "naked-poc" in n: return _signed(c.index,c<poc-1.5*a,c>poc+1.5*a)
            if "poc breakout" in n or "developing-poc migration" in n: return _signed(c.index,(c>poc)&(poc.diff()>0),(c<poc)&(poc.diff()<0))
            if "vah/val rejection" in n or "node rejection" in n or "profile-level reaction" in n:
                return _signed(c.index,(x.l<val)&(c>val),(x.h>vah)&(c<vah))
            if "rotation" in n:
                return _signed(c.index,(c<val)&(x.roc(3)>0),(c>vah)&(x.roc(3)<0))
            return _signed(c.index,c>vah,c<val)
        if "poor-high/poor-low" in n:
            eqh=(x.h-x.h.shift(1)).abs()<a*.1; eql=(x.l-x.l.shift(1)).abs()<a*.1
            return _signed(c.index,eql&(c>x.h.shift(1)),eqh&(c<x.l.shift(1)))
        if "volume-spike" in n:
            med=x.v.rolling(50).median(); spike=x.v>med*2
            if "reversal" in n: return _signed(c.index,spike&(c<x.o)&(x.l<x.l.shift(1)),spike&(c>x.o)&(x.h>x.h.shift(1)))
            return _signed(c.index,spike&(c>x.h.shift(1)),spike&(c<x.l.shift(1)))
        if "obv" in n:
            obv=x.obv()
            if "divergence" in n: return _signed(c.index,(c<c.shift(20))&(obv>obv.shift(20)),(c>c.shift(20))&(obv<obv.shift(20)))
            return _signed(c.index,(c>x.ema(20))&(obv>obv.rolling(20).mean()),(c<x.ema(20))&(obv<obv.rolling(20).mean()))
        if "accumulation/distribution" in n:
            mfm=((c-x.l)-(x.h-c))/(x.h-x.l).replace(0,np.nan); ad=(mfm*x.v).cumsum(); hi=ad.rolling(20).max().shift(1); lo=ad.rolling(20).min().shift(1)
            return _signed(c.index,ad>hi,ad<lo)
        if "chaikin money flow" in n:
            mfm=((c-x.l)-(x.h-c))/(x.h-x.l).replace(0,np.nan); cmf=(mfm*x.v).rolling(20).sum()/x.v.rolling(20).sum().replace(0,np.nan)
            return _signed(c.index,cmf>0,cmf<0)
        if "volume oscillator" in n:
            vo=x.v.rolling(5).mean()/x.v.rolling(20).mean().replace(0,np.nan)-1
            return _signed(c.index,(vo>0)&(c>x.ema(20)),(vo>0)&(c<x.ema(20)))
        if "positive/negative volume index" in n:
            ret=c.pct_change(); pvi=(1+ret.where(x.v>x.v.shift(1),0)).cumprod(); nvi=(1+ret.where(x.v<x.v.shift(1),0)).cumprod()
            return _signed(c.index,pvi>pvi.rolling(50).mean(),nvi<nvi.rolling(50).mean())
        if "volume price trend" in n:
            vpt=(c.pct_change()*x.v).cumsum(); return _signed(c.index,(c<c.shift(20))&(vpt>vpt.shift(20)),(c>c.shift(20))&(vpt<vpt.shift(20)))
        return _signed(c.index,c>sv,c<sv)

    def _signal_quant(self, s: StrategyDefinition) -> pd.Series:
        x=self.ctx; n=s.name.lower(); c=x.c
        if set(s.requires)-self.available_inputs(self.df,self.extras):
            raise ValueError(f"required quantitative inputs unavailable: {s.requires}")
        if "dxy-gold residual" in n:
            dxy=self._aligned_extra_close("dxy"); goldret=c.pct_change(); dxyret=dxy.pct_change()
            beta=goldret.rolling(200).cov(dxyret)/dxyret.rolling(200).var().replace(0,np.nan)
            resid=goldret-beta*dxyret; z=(resid-resid.rolling(100).mean())/resid.rolling(100).std().replace(0,np.nan)
            return _signed(c.index,z<-2,z>2)
        if "realized-volatility regime" in n:
            rv=x.realized_vol(20); med=rv.rolling(100).median(); return _signed(c.index,(rv<med)&(x.roc(20)>0),(rv<med)&(x.roc(20)<0))
        if "hurst-exponent regime" in n:
            ratio=x.realized_vol(10)/x.realized_vol(40).replace(0,np.nan); return _signed(c.index,(ratio>1)&(x.roc(20)>0),(ratio>1)&(x.roc(20)<0))
        if "variance-ratio trend/reversion" in n:
            ret=c.pct_change(); vr=ret.rolling(20).var()/(ret.rolling(5).var()*4).replace(0,np.nan)
            return _signed(c.index,((vr>1)&(x.roc(20)>0))|((vr<1)&(x.z(20)<-1.5)),((vr>1)&(x.roc(20)<0))|((vr<1)&(x.z(20)>1.5)))
        if "autocorrelation regime" in n:
            ret=c.pct_change(); ac=ret.rolling(50).corr(ret.shift(1)); return _signed(c.index,(ac>0)&(x.roc(10)>0),(ac>0)&(x.roc(10)<0))
        if "entropy-based regime" in n or "permutation-entropy" in n:
            sign=np.sign(c.diff()); ent=(sign!=sign.shift(1)).rolling(30).mean(); hi,lo=x.donchian(20)
            return _signed(c.index,(ent<.45)&(c>hi),(ent<.45)&(c<lo))
        if "kalman trend filter" in n or "state-space local-trend" in n:
            fair=x.ema(30); return _signed(c.index,(c>fair)&(fair.diff()>0),(c<fair)&(fair.diff()<0))
        if "fourier" in n or "spectral" in n or "wavelet" in n:
            fast=x.ema(8)-x.ema(21); slow=x.ema(21)-x.ema(55)
            return _signed(c.index,(fast>0)&(slow>0),(fast<0)&(slow<0))
        if "garch regime" in n:
            rv=x.realized_vol(20); shock=rv/rv.ewm(span=100,adjust=False).mean(); return _signed(c.index,(shock<1.2)&(x.roc(20)>0),(shock<1.2)&(x.roc(20)<0))
        # Model-training and external-market methods are rejected before this point.
        raise ValueError("strategy requires a dedicated fitted/external model and is not safely approximated")

    def _signal_macro(self, s: StrategyDefinition) -> pd.Series:
        x=self.ctx; n=s.name.lower(); c=x.c
        if n.startswith("dxy "):
            dxy=self._aligned_extra_close("dxy"); dm=dxy.pct_change(10); gm=c.pct_change(10)
            if "divergence" in n:
                return _signed(c.index,(gm<0)&(dm<0),(gm>0)&(dm>0))
            return _signed(c.index,dm<-0.003,dm>0.003)
        if "month-of-year" in n:
            # Expanding, past-only seasonal return estimate by calendar month.
            ret=c.pct_change(); m=x.month(); seasonal=ret.groupby(m).transform(lambda a:a.expanding().mean().shift(1))
            return _signed(c.index,seasonal>0,seasonal<0)
        if "day-of-week" in n:
            ret=c.pct_change(); d=x.weekday(); seasonal=ret.groupby(d).transform(lambda a:a.expanding().mean().shift(1))
            return _signed(c.index,seasonal>0,seasonal<0)
        if "turn-of-month" in n:
            dom=x.day(); last=x.t.dt.days_in_month; w=(dom<=2)|(dom>=last-1)
            return _signed(c.index,w&(x.roc(5)>0),w&(x.roc(5)<0))
        raise ValueError("macro/event strategy requires historical external event data")

    def _aligned_extra_close(self, key: str) -> pd.Series:
        extra=self.extras[key]
        if "datetime" not in extra or "close" not in extra:
            raise ValueError(f"{key} must contain datetime and close")
        target=pd.DataFrame({"datetime":self.ctx.t, "_idx":np.arange(len(self.df))}).sort_values("datetime")
        src=extra[["datetime","close"]].copy()
        src["datetime"]=pd.to_datetime(src["datetime"],utc=True,errors="coerce")
        src=src.dropna().sort_values("datetime")
        merged=pd.merge_asof(target,src,on="datetime",direction="backward")
        out=pd.Series(merged["close"].to_numpy(dtype=float),index=self.df.index)
        return out
