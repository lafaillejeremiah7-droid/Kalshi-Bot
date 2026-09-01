from __future__ import annotations

import heapq
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from xau_company.adaptive_research import AdaptiveStrategyResearchAgent
from xau_company.indicators import atr, donchian, ema, roc, rsi, rolling_zscore
from xau_company.selector import StrategyPick, StrategySelectorAgent


SOURCE_REPO = "simom1/XAUUSD-history"
RAW_ROOT = f"https://raw.githubusercontent.com/{SOURCE_REPO}/main"
FILES = {
    "5min": "Gold-Cash/XAUUSD/XAUUSD_M5_2010_2026.csv",
    "15min": "Gold-Cash/XAUUSD/XAUUSD_M15_2010_2026.csv",
    "1h": "Gold-Cash/XAUUSD/XAUUSD_H1_2010_2026.csv",
    "4h": "Gold-Cash/XAUUSD/XAUUSD_H4_2010_2026.csv",
    "dxy_1h": "Index-Cash/USDX/USDX_H1.csv",
}
TF_MINUTES = {"5min": 5, "15min": 15, "1h": 60, "4h": 240}
TF_CONTEXT_WEIGHTS = {"5min": 0.90, "15min": 1.00, "1h": 1.08, "4h": 1.12}
TF_SELECTOR_WEIGHTS = {"5min": 0.80, "15min": 1.00, "1h": 1.30, "4h": 1.55}
REGIMES = ("trend_up", "trend_down", "range", "volatile")
REGIME_TO_INDEX = {name: idx for idx, name in enumerate(REGIMES)}

MAX_CANDIDATES = 20_000
CATALOG_SIZE = 600
TRAIN_ROWS = 3_000  # exact live research history budget
MIN_CONFIDENCE = 0.72
MIN_CONSENSUS = 3
SPREAD_BPS = 1.5
SLIPPAGE_BPS = 0.5
STOP_ATR = 1.20
REWARD_RISK = 1.70
MAX_TRADES_PER_DAY = 2
CALIBRATION_BIN = 0.05
CALIBRATION_PRIOR = 20.0


def _download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1024:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, 5):
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
            if path.exists():
                path.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def _load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace("<", "").replace(">", "") for c in df.columns]
    if "time" in df.columns:
        stamp = df["time"]
    elif "datetime" in df.columns:
        stamp = df["datetime"]
    elif "date" in df.columns and "time" in df.columns:
        stamp = df["date"].astype(str) + " " + df["time"].astype(str)
    else:
        raise RuntimeError(f"No time column in {path}; columns={list(df.columns)}")

    out = pd.DataFrame({"datetime": pd.to_datetime(stamp, utc=True, errors="coerce")})
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise RuntimeError(f"Missing {col} in {path}; columns={list(df.columns)}")
        out[col] = pd.to_numeric(df[col], errors="coerce")
    out = (
        out.dropna(subset=["datetime", "open", "high", "low", "close"])
        .drop_duplicates(subset=["datetime"], keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    sane = (
        (out["open"] > 0)
        & (out["high"] >= out[["open", "close"]].max(axis=1))
        & (out["low"] <= out[["open", "close"]].min(axis=1))
        & (out["high"] >= out["low"])
    )
    out = out[sane].reset_index(drop=True)
    if len(out) < 500:
        raise RuntimeError(f"Insufficient usable bars in {path}: {len(out)}")
    return out


def _coverage(df: pd.DataFrame) -> dict[str, object]:
    start = pd.Timestamp(df["datetime"].iloc[0])
    end = pd.Timestamp(df["datetime"].iloc[-1])
    return {
        "rows": int(len(df)),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "calendar_years": float((end - start).total_seconds() / (365.2425 * 86400)),
    }


def _clip_common(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.Timestamp, pd.Timestamp]:
    xau = {k: v for k, v in frames.items() if k in TF_MINUTES}
    common_start = max(pd.Timestamp(v.datetime.iloc[0]) for v in xau.values())
    common_end = min(
        pd.Timestamp(v.datetime.iloc[-1]) + pd.Timedelta(minutes=TF_MINUTES[k])
        for k, v in xau.items()
    )
    clipped: dict[str, pd.DataFrame] = {}
    for key, df in frames.items():
        if key in TF_MINUTES:
            minutes = TF_MINUTES[key]
            mask = (df.datetime >= common_start) & (df.datetime + pd.Timedelta(minutes=minutes) <= common_end)
        else:
            mask = (df.datetime >= common_start - pd.Timedelta(days=20)) & (df.datetime <= common_end)
        clipped[key] = df.loc[mask].reset_index(drop=True)
    return clipped, common_start, common_end


def _regime_arrays(df: pd.DataFrame) -> tuple[np.ndarray, pd.Series, pd.Series]:
    close = df.close
    a14 = atr(df, 14)
    e20, e50 = ema(close, 20), ema(close, 50)
    strength = (e20 - e50).abs() / a14.replace(0, np.nan)
    vol_ratio = a14 / close.replace(0, np.nan)
    vol_med = vol_ratio.rolling(120).median()

    labels = np.full(len(df), REGIME_TO_INDEX["range"], dtype=np.int8)
    trending = strength.to_numpy(dtype=float) > 1.0
    labels[trending & (e20.to_numpy() > e50.to_numpy())] = REGIME_TO_INDEX["trend_up"]
    labels[trending & (e20.to_numpy() < e50.to_numpy())] = REGIME_TO_INDEX["trend_down"]
    labels[vol_ratio.to_numpy(dtype=float) > vol_med.to_numpy(dtype=float) * 1.5] = REGIME_TO_INDEX["volatile"]
    return labels, a14, vol_ratio


def _specialist_support(df: pd.DataFrame, regime_idx: np.ndarray, lab: AdaptiveStrategyResearchAgent):
    n = len(df)
    close, opn, high, low = df.close, df.open, df.high, df.low
    dirs = np.zeros((6, n), dtype=np.int8)
    conf = np.zeros((6, n), dtype=np.float32)

    e20, e50, e100 = ema(close, 20), ema(close, 50), ema(close, 100)
    bull = (e20 > e50) & (e50 > e100)
    bear = (e20 < e50) & (e50 < e100)
    dirs[0, bull.to_numpy()] = 1
    dirs[0, bear.to_numpy()] = -1
    q = lab.quality("trend")
    mult = np.where(np.isin(regime_idx, [0, 1]), 1.15, 0.80)
    conf[0] = np.clip(q * mult, 0.0, 0.95)

    hi20, lo20 = donchian(df, 20)
    m5 = roc(close, 5)
    bbuy = (close > hi20) & (m5 > 0)
    bsell = (close < lo20) & (m5 < 0)
    dirs[1, bbuy.to_numpy()] = 1
    dirs[1, bsell.to_numpy()] = -1
    q = lab.quality("breakout")
    boost = np.where(np.isin(regime_idx, [0, 1, 3]), 1.10, 0.85)
    conf[1] = np.clip(q * boost, 0.0, 0.95)

    rs = rsi(close, 14)
    z = rolling_zscore(close, 30)
    mbuy = (rs < 30) & (z < -1.0)
    msell = (rs > 70) & (z > 1.0)
    dirs[2, mbuy.to_numpy()] = 1
    dirs[2, msell.to_numpy()] = -1
    q = lab.quality("mean_reversion")
    boost = np.where(regime_idx == REGIME_TO_INDEX["range"], 1.15, 0.65)
    conf[2] = np.clip(q * boost, 0.0, 0.92)

    r3, r10 = roc(close, 3), roc(close, 10)
    mbuy2 = (r3 > 0) & (r10 > 0)
    msell2 = (r3 < 0) & (r10 < 0)
    dirs[3, mbuy2.to_numpy()] = 1
    dirs[3, msell2.to_numpy()] = -1
    q = lab.quality("momentum")
    boost = np.where(regime_idx != REGIME_TO_INDEX["range"], 1.08, 0.85)
    conf[3] = np.clip(q * boost, 0.0, 0.93)

    prev_open, prev_close = opn.shift(1), close.shift(1)
    body = (close - opn).abs()
    candle_range = (high - low).clip(lower=1e-9)
    body_ratio = body / candle_range
    bullish_engulf = (close > opn) & (prev_close < prev_open) & (close >= prev_open) & (opn <= prev_close) & (body_ratio > 0.55)
    bearish_engulf = (close < opn) & (prev_close > prev_open) & (opn >= prev_close) & (close <= prev_open) & (body_ratio > 0.55)
    dirs[4, bullish_engulf.to_numpy()] = 1
    dirs[4, bearish_engulf.to_numpy()] = -1
    conf[4].fill(0.62)

    h5, l5 = high.rolling(5).max(), low.rolling(5).min()
    higher = (h5 > h5.shift(5)) & (l5 > l5.shift(5))
    lower = (h5 < h5.shift(5)) & (l5 < l5.shift(5))
    dirs[5, higher.to_numpy()] = 1
    dirs[5, lower.to_numpy()] = -1
    conf[5].fill(0.61)

    def side(direction: int):
        support_mask = dirs == direction
        opposition_mask = dirs == -direction
        support_n = support_mask.sum(axis=0).astype(np.int8)
        opposition_n = opposition_mask.sum(axis=0).astype(np.int8)
        support = (conf * support_mask).sum(axis=0) / np.maximum(1, support_n)
        opposition = (conf * opposition_mask).sum(axis=0) / np.maximum(1, opposition_n)
        return support.astype(np.float32), opposition.astype(np.float32), support_n, opposition_n

    return side(1), side(-1)


def _frame_votes(df: pd.DataFrame, timeframe: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minutes = TF_MINUTES[timeframe]
    fast_span, slow_span = (9, 21) if timeframe == "5min" else (20, 50)
    close = df.close
    fast, slow = ema(close, fast_span), ema(close, slow_span)
    momentum = roc(close, 5)
    a = atr(df, 14)
    separation = (fast - slow).abs() / a.replace(0, np.nan)
    strength = 0.50 + separation * 0.12 + np.minimum(momentum.abs() * 30, 0.12)
    strength = np.clip(strength, 0.50, 0.82) * TF_CONTEXT_WEIGHTS[timeframe]
    strength = np.clip(strength, 0.0, 0.90)

    direction = np.zeros(len(df), dtype=np.int8)
    bullish = (fast > slow) & (momentum > 0)
    bearish = (fast < slow) & (momentum < 0)
    direction[bullish.to_numpy()] = 1
    direction[bearish.to_numpy()] = -1
    confidence = np.where(direction == 0, 0.42, strength.to_numpy(dtype=float)).astype(np.float32)
    confidence[:59] = 0.30
    direction[:59] = 0
    ends = (df.datetime + pd.Timedelta(minutes=minutes)).astype("int64").to_numpy()
    return ends, direction, confidence


def _align_votes(target_end_ns: np.ndarray, source_ends: np.ndarray, direction: np.ndarray, confidence: np.ndarray):
    idx = np.searchsorted(source_ends, target_end_ns, side="right") - 1
    out_dir = np.zeros(len(target_end_ns), dtype=np.int8)
    out_conf = np.full(len(target_end_ns), 0.30, dtype=np.float32)
    valid = idx >= 0
    out_dir[valid] = direction[idx[valid]]
    out_conf[valid] = confidence[idx[valid]]
    return out_dir, out_conf


def _timeframe_alignment(frames: dict[str, pd.DataFrame], target_end_ns: np.ndarray, direction: int) -> np.ndarray:
    signed = np.zeros(len(target_end_ns), dtype=np.float32)
    total = 0.0
    for timeframe in ("5min", "15min", "1h", "4h"):
        ends, src_dir, src_conf = _frame_votes(frames[timeframe], timeframe)
        aligned_dir, aligned_conf = _align_votes(target_end_ns, ends, src_dir, src_conf)
        weight = TF_SELECTOR_WEIGHTS[timeframe]
        total += weight
        signed += np.where(
            aligned_dir == direction,
            weight * aligned_conf,
            np.where(aligned_dir != 0, -weight * aligned_conf, 0.0),
        )
    return np.clip(0.5 + signed / max(total, 1e-9) * 0.5, 0.0, 1.0).astype(np.float32)


def _macro_alignment(dxy: pd.DataFrame | None, target_end_ns: np.ndarray, direction: int) -> np.ndarray:
    if dxy is None or dxy.empty:
        return np.full(len(target_end_ns), 0.5, dtype=np.float32)
    close = dxy.close
    short, long = ema(close, 10), ema(close, 30)
    move = roc(close, 5)
    magnitude = np.minimum(move.abs() * 35, 0.15)
    confidence = np.clip(0.52 + magnitude, 0.52, 0.70).to_numpy(dtype=np.float32)
    src_dir = np.zeros(len(dxy), dtype=np.int8)
    src_dir[((short > long) & (move > 0)).to_numpy()] = -1
    src_dir[((short < long) & (move < 0)).to_numpy()] = 1
    confidence[src_dir == 0] = 0.40
    src_dir[:34] = 0
    confidence[:34] = 0.30
    ends = (dxy.datetime + pd.Timedelta(hours=1)).astype("int64").to_numpy()
    aligned_dir, aligned_conf = _align_votes(target_end_ns, ends, src_dir, confidence)
    signed = np.where(aligned_dir == direction, aligned_conf, np.where(aligned_dir != 0, -aligned_conf, 0.0))
    return np.clip(0.5 + signed * 0.5, 0.0, 1.0).astype(np.float32)


class OnlineCalibrator:
    def __init__(self) -> None:
        self.global_stats: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        self.context_stats: dict[tuple[int, str, str], list[int]] = defaultdict(lambda: [0, 0])
        self.brier_sum = 0.0
        self.brier_n = 0

    @staticmethod
    def bucket(probability: float) -> int:
        p = float(np.clip(probability, 0.0, 0.999999))
        return int(math.floor(p / CALIBRATION_BIN))

    def resolve(self, raw_probability: float, calibrated_probability: float, family: str, regime: str, won: bool) -> None:
        bucket = self.bucket(raw_probability)
        for key in (bucket,):
            stats = self.global_stats[key]
            stats[0] += 1
            stats[1] += int(won)
        ctx = self.context_stats[(bucket, family, regime)]
        ctx[0] += 1
        ctx[1] += int(won)
        self.brier_sum += (float(calibrated_probability) - (1.0 if won else 0.0)) ** 2
        self.brier_n += 1

    def calibrate(self, raw_probability: float, family: str, regime: str) -> tuple[float, int]:
        raw = float(np.clip(raw_probability, 0.01, 0.99))
        bucket = self.bucket(raw)
        n, wins = self.global_stats[bucket]
        if n == 0:
            return raw, 0
        global_posterior = (wins + raw * CALIBRATION_PRIOR) / (n + CALIBRATION_PRIOR)
        posterior = global_posterior
        used_n = n
        context_n, context_wins = self.context_stats[(bucket, family, regime)]
        if context_n >= 8:
            context_posterior = (context_wins + raw * CALIBRATION_PRIOR) / (context_n + CALIBRATION_PRIOR)
            context_weight = min(0.65, context_n / (context_n + 20.0))
            posterior = global_posterior * (1.0 - context_weight) + context_posterior * context_weight
            used_n = context_n
        return float(np.clip(posterior, 0.05, 0.95)), used_n

    @property
    def brier(self) -> float | None:
        return self.brier_sum / self.brier_n if self.brier_n else None


def _candidate_static(selector: StrategySelectorAgent, catalog, regime: str) -> np.ndarray:
    values: list[float] = []
    for result in catalog:
        family_fit = selector._family_regime_fit(result.candidate.family, regime)
        regime_history, _ = selector._historical_regime_fit(result, regime)
        regime_fit = family_fit * 0.35 + regime_history * 0.65
        wf_hit = result.walk_forward_hit_rate if result.walk_forward_hit_rate > 0 else result.valid_hit_rate
        stability_gap = abs(result.train_hit_rate - result.valid_hit_rate)
        stability = float(np.clip(1.0 - stability_gap - result.walk_forward_std * 1.5, 0.0, 1.0))
        pf_score = result.profit_factor / (1.0 + max(0.0, result.profit_factor))
        lifecycle = selector._lifecycle_quality(result)
        sample_n = selector._oos_sample_size(result)
        sample_trust = float(np.clip(np.log1p(sample_n) / np.log(401), 0.0, 1.0))
        base = (
            wf_hit * 0.20
            + result.score * 0.13
            + regime_fit * 0.18
            + pf_score * 0.05
            + stability * 0.07
            + lifecycle * 0.11
            - (1.0 - sample_trust) * 0.08
        )
        values.append(float(base))
    return np.asarray(values, dtype=np.float32)


def _label(result) -> str:
    pick = StrategyPick(
        score=result,
        direction=None,  # type: ignore[arg-type]
        probability_score=0.0,
        analyst_agreement=0,
        analyst_opposition=0,
        regime_fit=0.5,
    )
    return pick.label


def _resolve_trade(
    m5: pd.DataFrame,
    m5_ns: np.ndarray,
    signal_end: pd.Timestamp,
    direction: int,
    known_atr: float,
    holding_minutes: int,
) -> dict[str, object] | None:
    signal_ns = int(signal_end.value)
    entry_idx = int(np.searchsorted(m5_ns, signal_ns, side="left"))
    if entry_idx >= len(m5):
        return None
    # Do not invent a fill across an extended data gap.
    if int(m5_ns[entry_idx]) - signal_ns > int(pd.Timedelta(minutes=15).value):
        return None

    side_cost = (SPREAD_BPS / 2.0 + SLIPPAGE_BPS) / 10_000.0
    raw_open = float(m5.open.iloc[entry_idx])
    entry = raw_open * (1.0 + side_cost) if direction > 0 else raw_open * (1.0 - side_cost)
    risk_distance = float(known_atr) * STOP_ATR
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(risk_distance) or risk_distance <= 0:
        return None
    if direction > 0:
        stop, target = entry - risk_distance, entry + risk_distance * REWARD_RISK
    else:
        stop, target = entry + risk_distance, entry - risk_distance * REWARD_RISK

    expiry = signal_end + pd.Timedelta(minutes=holding_minutes)
    expiry_ns = int(expiry.value)
    last_exclusive = int(np.searchsorted(m5_ns, expiry_ns, side="left"))
    if last_exclusive <= entry_idx:
        return None
    last_idx = min(len(m5) - 1, last_exclusive - 1)
    exit_idx = last_idx
    raw_exit = float(m5.close.iloc[last_idx])
    reason = "timeout"

    for bar_idx in range(entry_idx, last_idx + 1):
        hi = float(m5.high.iloc[bar_idx])
        lo = float(m5.low.iloc[bar_idx])
        stop_hit = lo <= stop if direction > 0 else hi >= stop
        target_hit = hi >= target if direction > 0 else lo <= target
        if stop_hit:  # conservative ambiguous-bar ordering
            exit_idx, raw_exit, reason = bar_idx, stop, "stop"
            break
        if target_hit:
            exit_idx, raw_exit, reason = bar_idx, target, "target"
            break

    exit_fill = raw_exit * (1.0 - side_cost) if direction > 0 else raw_exit * (1.0 + side_cost)
    pnl = (exit_fill - entry) * direction
    r_multiple = pnl / max(risk_distance, 1e-12)
    resolution_time = pd.Timestamp(m5.datetime.iloc[exit_idx]) + pd.Timedelta(minutes=5)
    return {
        "entry": float(entry),
        "exit": float(exit_fill),
        "stop": float(stop),
        "target": float(target),
        "r": float(r_multiple),
        "won": bool(pnl > 0),
        "reason": reason,
        "resolution_time": resolution_time,
    }


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    path = np.concatenate(([0.0], np.cumsum(np.asarray(values, dtype=float))))
    peaks = np.maximum.accumulate(path)
    return float(np.max(peaks - path))


def _max_loss_streak(wins: list[bool]) -> int:
    worst = current = 0
    for won in wins:
        if won:
            current = 0
        else:
            current += 1
            worst = max(worst, current)
    return worst


def main() -> int:
    out_path = Path(os.getenv("BACKTEST_REPORT", "full-company-backtest-report.json"))
    cache_dir = Path(os.getenv("BACKTEST_CACHE", "/tmp/xau-full-company-backtest"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    original_coverage: dict[str, object] = {}
    for key, rel in FILES.items():
        local = cache_dir / Path(rel).name
        _download(f"{RAW_ROOT}/{rel}", local)
        frame = _load_ohlc(local)
        frames[key] = frame
        original_coverage[key] = _coverage(frame)
        print(f"loaded {key}: {len(frame):,} rows {frame.datetime.iloc[0]} -> {frame.datetime.iloc[-1]}", flush=True)

    frames, common_start, common_end = _clip_common(frames)
    core = frames["15min"].copy()
    if len(core) <= TRAIN_ROWS + 500:
        raise RuntimeError(f"Not enough 15m history after common clipping: {len(core)}")

    train = core.iloc[:TRAIN_ROWS].copy().reset_index(drop=True)
    train_end = pd.Timestamp(train.datetime.iloc[-1]) + pd.Timedelta(minutes=15)
    test_start = train_end
    test_end = common_end

    with tempfile.TemporaryDirectory(prefix="xau-adaptive-lab-") as tmp:
        lab = AdaptiveStrategyResearchAgent(
            max_candidates=MAX_CANDIDATES,
            spread_bps=SPREAD_BPS,
            walk_forward_folds=4,
            catalog_size=CATALOG_SIZE,
            min_walk_forward_folds=2,
            slippage_bps=SLIPPAGE_BPS,
            backtest_stop_atr=STOP_ATR,
            backtest_reward_risk=REWARD_RISK,
            enable_evolution=True,
            strategy_library_path=str(Path(tmp) / "evolved.json"),
            discoveries_per_cycle=250,
            discovery_library_size=5_000,
            enable_invention=True,
            invention_library_path=str(Path(tmp) / "invented.json"),
            invented_families_per_cycle=6,
            invented_variants_per_family=8,
            invention_library_size=4_000,
        )
        print("research cycle 1/2", flush=True)
        lab.run(train)
        cycle1 = {
            "universe": int(lab.last_universe_size),
            "evaluated": int(lab.last_evaluated),
            "catalog": int(len(lab.catalog)),
            "discovered": int(lab.last_discovered),
            "invented_families": int(lab.last_invented_families),
            "invented_variants": int(lab.last_invented_variants),
        }
        print("research cycle 2/2 (audits evolved/invented variants)", flush=True)
        lab.run(train)
        cycle2 = {
            "universe": int(lab.last_universe_size),
            "evaluated": int(lab.last_evaluated),
            "catalog": int(len(lab.catalog)),
            "seed_audited": int(lab.last_seed_audited),
            "seed_rejected": int(lab.last_seed_overfit_rejected),
            "seed_live_eligible": int(lab.last_seed_live_eligible),
            "evolution_promoted": int(lab.last_promoted),
            "evolution_quarantined": int(lab.last_quarantined),
            "invention_promoted": int(lab.last_invention_promoted),
            "invention_quarantined": int(lab.last_invention_quarantined),
            "dynamic_library": int(lab.dynamic_library_size),
            "invention_library": int(lab.invention_library_size),
        }

        catalog = list(lab.catalog)
        if not catalog:
            raise RuntimeError("Production overfit gates produced no live research catalog on calibration window")
        selector = StrategySelectorAgent(MIN_CONFIDENCE, MIN_CONSENSUS)

        regime_idx, a14, vol_ratio = _regime_arrays(core)
        vol_baseline = vol_ratio.rolling(200).median().to_numpy(dtype=float)
        vol_veto = vol_ratio.to_numpy(dtype=float) > vol_baseline * 2.2
        (buy_support, buy_opp, buy_n, buy_opp_n), (sell_support, sell_opp, sell_n, sell_opp_n) = _specialist_support(core, regime_idx, lab)

        target_end = core.datetime + pd.Timedelta(minutes=15)
        target_end_ns = target_end.astype("int64").to_numpy()
        tf_buy = _timeframe_alignment(frames, target_end_ns, 1)
        tf_sell = _timeframe_alignment(frames, target_end_ns, -1)
        macro_buy = _macro_alignment(frames.get("dxy_1h"), target_end_ns, 1)
        macro_sell = _macro_alignment(frames.get("dxy_1h"), target_end_ns, -1)

        static = np.stack([_candidate_static(selector, catalog, regime) for regime in REGIMES], axis=0)
        print(f"precomputing live directions for {len(catalog)} audited strategies across {len(core):,} 15m bars", flush=True)
        signal_matrix = np.zeros((len(catalog), len(core)), dtype=np.int8)
        signal_cache: dict[tuple, pd.Series | tuple[pd.Series, pd.Series]] = {}
        for j, result in enumerate(catalog):
            signal_matrix[j] = lab._signal(core, result.candidate, signal_cache).to_numpy(dtype=np.int8)
            if (j + 1) % 50 == 0 or j + 1 == len(catalog):
                print(f"  strategy signals {j + 1}/{len(catalog)}", flush=True)

        labels = [_label(result) for result in catalog]
        families = [result.candidate.family for result in catalog]
        horizons = [int(lab.HORIZONS.get(result.candidate.family, 4)) * 15 for result in catalog]

        calibrator = OnlineCalibrator()
        pending: list[tuple[int, int, dict[str, object]]] = []
        pending_serial = 0
        daily_counts: Counter[str] = Counter()
        selected_counter: Counter[str] = Counter()
        family_counter: Counter[str] = Counter()
        regime_counter: Counter[str] = Counter()
        trades: list[dict[str, object]] = []
        counters = Counter()

        m5 = frames["5min"]
        m5_ns = m5.datetime.astype("int64").to_numpy()
        oos_start_idx = int(np.searchsorted(core.datetime.astype("int64").to_numpy(), int(test_start.value), side="left"))

        def resolve_due(now_ns: int) -> None:
            while pending and pending[0][0] <= now_ns:
                _, _, event = heapq.heappop(pending)
                calibrator.resolve(
                    float(event["raw_confidence"]),
                    float(event["calibrated_confidence"]),
                    str(event["family"]),
                    str(event["regime"]),
                    bool(event["won"]),
                )

        print(f"strict forward replay: {test_start} -> {test_end}", flush=True)
        for i in range(max(oos_start_idx, 220), len(core)):
            candle_start = pd.Timestamp(core.datetime.iloc[i])
            decision_time = candle_start + pd.Timedelta(minutes=15)
            if decision_time > test_end:
                break
            resolve_due(int(decision_time.value))

            # Exact Session Desk window uses research-candle start hour.
            if not (7 <= int(candle_start.hour) <= 17):
                counters["session_veto"] += 1
                continue
            if bool(vol_veto[i]):
                counters["volatility_veto"] += 1
                continue
            counters["decision_bars"] += 1

            current = signal_matrix[:, i]
            if not np.any(current):
                counters["no_strategy_direction"] += 1
                continue

            regime = REGIMES[int(regime_idx[i])]
            base = static[int(regime_idx[i])]
            probabilities = np.full(len(catalog), -1.0, dtype=np.float32)

            if int(buy_n[i]) >= MIN_CONSENSUS:
                buy_dynamic = buy_support[i] * 0.09 + tf_buy[i] * 0.13 + macro_buy[i] * 0.04 - buy_opp[i] * 0.08
                buy_mask = current == 1
                probabilities[buy_mask] = base[buy_mask] + buy_dynamic
            if int(sell_n[i]) >= MIN_CONSENSUS:
                sell_dynamic = sell_support[i] * 0.09 + tf_sell[i] * 0.13 + macro_sell[i] * 0.04 - sell_opp[i] * 0.08
                sell_mask = current == -1
                probabilities[sell_mask] = base[sell_mask] + sell_dynamic

            probabilities = np.clip(probabilities, -1.0, 0.97)
            best_idx = int(np.argmax(probabilities))
            raw_conf = float(probabilities[best_idx])
            if raw_conf < MIN_CONFIDENCE:
                counters["selector_veto"] += 1
                continue
            counters["selector_pass"] += 1

            direction = int(current[best_idx])
            family = families[best_idx]
            calibrated_conf, calibration_samples = calibrator.calibrate(raw_conf, family, regime)
            if calibrated_conf < MIN_CONFIDENCE:
                counters["calibration_veto"] += 1
                continue

            local_date = decision_time.tz_convert("America/Chicago").date().isoformat()
            if daily_counts[local_date] >= MAX_TRADES_PER_DAY:
                counters["frequency_veto"] += 1
                continue

            outcome = _resolve_trade(
                m5,
                m5_ns,
                decision_time,
                direction,
                float(a14.iloc[i]),
                horizons[best_idx],
            )
            if outcome is None:
                counters["execution_data_veto"] += 1
                continue

            daily_counts[local_date] += 1
            counters["accepted"] += 1
            counters["buy" if direction > 0 else "sell"] += 1
            selected_counter[labels[best_idx]] += 1
            family_counter[family] += 1
            regime_counter[regime] += 1

            trade = {
                "setup_at": candle_start.isoformat(),
                "decision_at": decision_time.isoformat(),
                "direction": "BUY" if direction > 0 else "SELL",
                "strategy": labels[best_idx],
                "family": family,
                "regime": regime,
                "raw_confidence": raw_conf,
                "calibrated_confidence": calibrated_conf,
                "calibration_samples": int(calibration_samples),
                **{k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in outcome.items()},
            }
            trades.append(trade)

            pending_serial += 1
            pending_event = {
                "raw_confidence": raw_conf,
                "calibrated_confidence": calibrated_conf,
                "family": family,
                "regime": regime,
                "won": bool(outcome["won"]),
            }
            heapq.heappush(
                pending,
                (int(pd.Timestamp(outcome["resolution_time"]).value), pending_serial, pending_event),
            )

        resolve_due(int(test_end.value) + int(pd.Timedelta(days=10).value))

        resolved_trades = [t for t in trades if pd.Timestamp(t["resolution_time"]) <= test_end]
        wins = [bool(t["won"]) for t in resolved_trades]
        r_values = [float(t["r"]) for t in resolved_trades]
        gross_win = sum(x for x in r_values if x > 0)
        gross_loss = -sum(x for x in r_values if x < 0)
        profit_factor_r = gross_win / gross_loss if gross_loss > 1e-12 else (4.0 if gross_win > 0 else 1.0)

        yearly: dict[str, dict[str, object]] = {}
        for year, rows in pd.DataFrame(resolved_trades).groupby(pd.to_datetime(pd.DataFrame(resolved_trades)["decision_at"], utc=True).dt.year) if resolved_trades else []:
            records = rows.to_dict("records")
            year_wins = sum(bool(r["won"]) for r in records)
            year_r = [float(r["r"]) for r in records]
            yearly[str(int(year))] = {
                "trades": len(records),
                "wins": year_wins,
                "losses": len(records) - year_wins,
                "win_rate": year_wins / len(records) if records else None,
                "total_r": float(sum(year_r)),
                "avg_r": float(np.mean(year_r)) if year_r else None,
            }

        top_research = []
        for result in lab.top[:25]:
            top_research.append(
                {
                    "strategy": _label(result),
                    "family": result.candidate.family,
                    "research_score": float(result.score),
                    "walk_forward_hit_rate": float(result.walk_forward_hit_rate),
                    "walk_forward_std": float(result.walk_forward_std),
                    "profit_factor": float(result.profit_factor),
                    "avg_r": float(result.avg_r_multiple),
                    "max_drawdown_r": float(result.max_drawdown_r),
                    "max_loss_streak": int(result.max_loss_streak),
                    "trades": int(result.trades),
                    "folds": int(result.folds),
                }
            )

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "git_sha": os.getenv("GITHUB_SHA"),
            "status": "complete",
            "dataset": {
                "source_repo": SOURCE_REPO,
                "source_type": "third-party MT5 OHLCV history, timestamps documented as UTC",
                "files": FILES,
                "original_coverage": original_coverage,
                "common_xau_start": common_start.isoformat(),
                "common_xau_end": common_end.isoformat(),
                "common_calendar_years": float((common_end - common_start).total_seconds() / (365.2425 * 86400)),
                "execution_timeframe": "5min",
                "research_timeframe": "15min",
                "context_timeframes": ["5min", "15min", "1h", "4h"],
                "dxy_context_coverage": _coverage(frames["dxy_1h"]) if not frames["dxy_1h"].empty else None,
            },
            "configuration": {
                "max_candidates": MAX_CANDIDATES,
                "catalog_size": CATALOG_SIZE,
                "train_rows": TRAIN_ROWS,
                "train_start": pd.Timestamp(train.datetime.iloc[0]).isoformat(),
                "train_end": train_end.isoformat(),
                "strict_forward_start": test_start.isoformat(),
                "strict_forward_end": test_end.isoformat(),
                "strict_forward_calendar_years": float((test_end - test_start).total_seconds() / (365.2425 * 86400)),
                "min_confidence": MIN_CONFIDENCE,
                "min_consensus": MIN_CONSENSUS,
                "spread_bps": SPREAD_BPS,
                "slippage_bps": SLIPPAGE_BPS,
                "stop_atr": STOP_ATR,
                "reward_risk": REWARD_RISK,
                "max_trades_per_day": MAX_TRADES_PER_DAY,
                "trade_timezone": "America/Chicago",
                "evolution_enabled": True,
                "invention_enabled": True,
                "news_event_blackouts": "none, matching current default empty HIGH_IMPACT_EVENTS_UTC",
                "yield_context": "neutral because no continuous historical Treasury-yield feed was supplied",
                "one_minute_context": "omitted; live quality logic explicitly permits 5min as execution fallback",
            },
            "research": {
                "cycle_1": cycle1,
                "cycle_2": cycle2,
                "final_live_catalog": len(catalog),
                "top_25": top_research,
            },
            "forward_replay": {
                "counters": dict(counters),
                "resolved_trades": len(resolved_trades),
                "wins": int(sum(wins)),
                "losses": int(len(wins) - sum(wins)),
                "win_rate": float(np.mean(wins)) if wins else None,
                "total_r": float(sum(r_values)),
                "avg_r": float(np.mean(r_values)) if r_values else None,
                "median_r": float(np.median(r_values)) if r_values else None,
                "profit_factor_r": float(profit_factor_r),
                "max_drawdown_r": _max_drawdown(r_values),
                "max_loss_streak": _max_loss_streak(wins),
                "calibration_brier": calibrator.brier,
                "selected_strategy_counts": dict(selected_counter.most_common(20)),
                "family_counts": dict(family_counter),
                "regime_counts": dict(regime_counter),
                "by_year": yearly,
            },
            "methodology": [
                "The adaptive research company runs two initial cycles on the first 3,000 completed 15m bars, matching the live research history budget; cycle two can audit variants proposed/invented by cycle one.",
                "The resulting audited live catalog is then frozen for the strict forward replay so no future price data can alter strategy selection.",
                "Every later 15m decision recreates the live Regime, Trend, Breakout, Mean Reversion, Momentum, Price Action, Structure, Volatility, Session, multi-timeframe, USD macro, Strategy Selector, calibration, and frequency logic.",
                "Entry is the first available 5m open after the 15m setup closes. Spread/slippage are charged on both entry and exit; ATR stop/target geometry matches the research backtester; ambiguous stop/target bars assume stop first.",
                "Calibration is updated only when an earlier accepted trade has actually resolved, preventing future outcomes from leaking backward.",
                "No dollar PnL is reported because the company has no historical position-size/account-equity series in this test; performance is reported in R multiples.",
            ],
            "limitations": [
                "The source is third-party broker/MT5 history, not the live Dukascopy feed, so broker-specific candles can differ.",
                "Historical Treasury-yield context is neutral and no historical news blackout calendar is fabricated.",
                "The research catalog is intentionally frozen after the initial two cycles to maintain a clean long out-of-sample test. Live operation periodically researches again, so this is conservative but not a perfect simulation of future adaptive retraining.",
                "5m execution is used for the entire span because continuous 1m history is not available for all 2010-2026; the live system explicitly supports 5m fallback.",
            ],
        }

    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_path), "summary": report["forward_replay"]}, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
