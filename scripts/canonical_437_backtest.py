from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from xau_company.backtest import TradeLifecycleBacktester, TradeOutcome
from xau_company.canonical_strategies import STRATEGIES
from xau_company.canonical_strategy_engine import CanonicalSignalEngine


SOURCE_REPO = "simom1/XAUUSD-history"
RAW_ROOT = f"https://raw.githubusercontent.com/{SOURCE_REPO}/main"
XAU_FILE = "Gold-Cash/XAUUSD/XAUUSD_M15_2010_2026.csv"
DXY_FILE = "Index-Cash/USDX/USDX_H1.csv"

REPORT_PATH = Path(os.getenv("CANONICAL_437_REPORT", "canonical-437-backtest-report.json"))
SURVIVOR_PATH = Path(os.getenv("CANONICAL_437_SURVIVORS", "xau_company/surviving_strategies.json"))
CACHE_ROOT = Path(os.getenv("BACKTEST_CACHE", "/tmp/xau-canonical-437"))
MAX_SURVIVORS = math.floor(len(STRATEGIES) * 0.25)  # 109; never exceed 25%.
SELECTION_START = pd.Timestamp("2016-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2022-01-01", tz="UTC")
MIN_SELECTION_TRADES = 30
MIN_HOLDOUT_TRADES = 15

BACKTESTER = TradeLifecycleBacktester(
    spread_bps=1.5,
    slippage_bps=0.5,
    stop_atr=1.20,
    reward_risk=1.70,
)


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


def _load_ohlcv(path: Path) -> pd.DataFrame:
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

    volume_col = next(
        (col for col in ("volume", "tick_volume", "tickvol", "tick_volume_", "vol") if col in raw),
        None,
    )
    if volume_col:
        out["volume"] = pd.to_numeric(raw[volume_col], errors="coerce").fillna(0.0).clip(lower=0.0)

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


def _coverage(df: pd.DataFrame) -> dict[str, object]:
    start = pd.Timestamp(df.datetime.iloc[0])
    end = pd.Timestamp(df.datetime.iloc[-1])
    return {
        "rows": int(len(df)),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "calendar_years": float((end - start).total_seconds() / (365.2425 * 86400)),
        "has_volume": bool("volume" in df and float(df["volume"].sum()) > 0),
    }


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df.close.shift(1)
    tr = pd.concat(
        [df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


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
        }
    rs = np.asarray([t.r_multiple for t in trades], dtype=float)
    return {
        "trades": int(len(trades)),
        "win_rate": float(np.mean([t.won for t in trades])),
        "avg_r": float(np.mean(rs)),
        "profit_factor": float(BACKTESTER.profit_factor(trades)),
        "max_drawdown_r": float(BACKTESTER.max_drawdown_r(trades)),
        "max_loss_streak": int(BACKTESTER.max_loss_streak(trades)),
        "total_r": float(rs.sum()),
    }


def _subset(trades: list[TradeOutcome], times: pd.Series, start: pd.Timestamp | None, end: pd.Timestamp | None) -> list[TradeOutcome]:
    out = []
    for t in trades:
        ts = pd.Timestamp(times.iloc[t.signal_index])
        if start is not None and ts < start:
            continue
        if end is not None and ts >= end:
            continue
        out.append(t)
    return out


def _yearly(trades: list[TradeOutcome], times: pd.Series) -> dict[str, dict[str, float | int]]:
    buckets: dict[int, list[TradeOutcome]] = {}
    for t in trades:
        year = int(pd.Timestamp(times.iloc[t.signal_index]).year)
        buckets.setdefault(year, []).append(t)
    return {str(year): _metrics(rows) for year, rows in sorted(buckets.items())}


def _selection_score(m: dict[str, float | int], yearly: dict[str, dict[str, float | int]]) -> float:
    n = int(m["trades"])
    if n <= 0:
        return -1e9
    avg_r = float(m["avg_r"])
    pf = float(m["profit_factor"])
    wr = float(m["win_rate"])
    dd = float(m["max_drawdown_r"])
    years = [v for y, v in yearly.items() if 2016 <= int(y) < 2022 and int(v["trades"]) >= 3]
    positive_years = sum(float(v["avg_r"]) > 0 for v in years)
    consistency = positive_years / max(1, len(years))
    sample = min(1.0, math.log1p(n) / math.log(301))
    expectancy_score = 0.5 + 0.5 * math.tanh(avg_r / 0.35)
    pf_score = min(1.0, max(0.0, (pf - 0.75) / 1.50))
    dd_score = math.exp(-max(0.0, dd) / 12.0)
    return float(
        0.30 * expectancy_score
        + 0.20 * pf_score
        + 0.12 * wr
        + 0.13 * sample
        + 0.15 * consistency
        + 0.10 * dd_score
    )


def _normal_positive_edge_pvalue(trades: list[TradeOutcome]) -> float:
    if len(trades) < 3:
        return 1.0
    a = np.asarray([t.r_multiple for t in trades], dtype=float)
    sd = float(np.std(a, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        return 0.0 if float(np.mean(a)) > 0 else 1.0
    z = float(np.mean(a)) / (sd / math.sqrt(len(a)))
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def _bh_qvalues(rows: list[dict[str, object]]) -> None:
    valid = [(i, float(r.get("selection_p_value", 1.0))) for i, r in enumerate(rows) if r.get("status") == "evaluated"]
    valid.sort(key=lambda x: x[1])
    m = len(valid)
    q = [1.0] * m
    running = 1.0
    for pos in range(m - 1, -1, -1):
        _, p = valid[pos]
        rank = pos + 1
        running = min(running, p * m / rank)
        q[pos] = min(1.0, running)
    for pos, (idx, _) in enumerate(valid):
        rows[idx]["selection_bh_q"] = q[pos]


def _lookahead_check(strategy, full_engine: CanonicalSignalEngine, df: pd.DataFrame, extras: dict[str, pd.DataFrame]) -> tuple[bool, str | None]:
    cut = int(len(df) * 0.70)
    if cut < 1000:
        return False, "dataset too short for anti-lookahead test"
    try:
        full = full_engine.signal(strategy).iloc[:cut].reset_index(drop=True)
        prefix_df = df.iloc[:cut].copy().reset_index(drop=True)
        prefix_end = pd.Timestamp(prefix_df.datetime.iloc[-1])
        prefix_extras = {
            k: v.loc[pd.to_datetime(v["datetime"], utc=True, errors="coerce") <= prefix_end].copy().reset_index(drop=True)
            for k, v in extras.items()
        }
        prefix = CanonicalSignalEngine(prefix_df, prefix_extras).signal(strategy).reset_index(drop=True)
    except Exception as exc:
        return False, f"anti-lookahead signal build failed: {exc}"
    # Ignore the first 250 warmup bars; compare every available pre-cut signal.
    a = full.iloc[250:].to_numpy(dtype=np.int8)
    b = prefix.iloc[250:].to_numpy(dtype=np.int8)
    if len(a) != len(b):
        return False, "anti-lookahead signal lengths differ"
    if not np.array_equal(a, b):
        mismatch = int(np.flatnonzero(a != b)[0]) + 250
        return False, f"future-data sensitivity at bar {mismatch}"
    return True, None


def main() -> None:
    if len(STRATEGIES) != 437:
        raise RuntimeError(f"expected 437 canonical strategies, got {len(STRATEGIES)}")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    xau_path = CACHE_ROOT / "xau_m15.csv"
    dxy_path = CACHE_ROOT / "dxy_h1.csv"
    _download(f"{RAW_ROOT}/{XAU_FILE}", xau_path)
    _download(f"{RAW_ROOT}/{DXY_FILE}", dxy_path)

    xau = _load_ohlcv(xau_path)
    dxy = _load_ohlcv(dxy_path)
    extras = {"dxy": dxy}
    available = CanonicalSignalEngine.available_inputs(xau, extras)
    engine = CanonicalSignalEngine(xau, extras)
    atr_values = _atr(xau, 14)

    rows: list[dict[str, object]] = []
    print(f"Loaded XAU M15: {_coverage(xau)}")
    print(f"Loaded DXY H1: {_coverage(dxy)}")
    print(f"Canonical strategies: {len(STRATEGIES)}; max survivors: {MAX_SURVIVORS}")

    for ordinal, strategy in enumerate(STRATEGIES, start=1):
        row: dict[str, object] = {
            "strategy_id": strategy.strategy_id,
            "name": strategy.name,
            "category": strategy.category,
            "requires": list(strategy.requires),
            "parameter_trials": 1,
            "status": "pending",
        }
        missing = sorted(set(strategy.requires) - available)
        if missing:
            row.update(status="rejected_missing_inputs", rejection_reason=f"missing {missing}")
            rows.append(row)
            print(f"[{ordinal:03d}/437] {strategy.strategy_id} rejected: missing {missing}")
            continue

        passed, reason = _lookahead_check(strategy, engine, xau, extras)
        row["lookahead_check"] = "passed" if passed else "failed"
        if not passed:
            row.update(status="rejected_lookahead", rejection_reason=reason)
            rows.append(row)
            print(f"[{ordinal:03d}/437] {strategy.strategy_id} rejected: {reason}")
            continue

        try:
            signal = engine.signal(strategy)
            trades = BACKTESTER.simulate(xau, signal, atr_values, strategy.horizon)
        except Exception as exc:
            row.update(status="rejected_execution", rejection_reason=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            print(f"[{ordinal:03d}/437] {strategy.strategy_id} rejected: execution {exc}")
            continue

        development = _subset(trades, xau.datetime, None, SELECTION_START)
        selection = _subset(trades, xau.datetime, SELECTION_START, HOLDOUT_START)
        holdout = _subset(trades, xau.datetime, HOLDOUT_START, None)
        yearly = _yearly(trades, xau.datetime)
        dev_m, sel_m, hold_m = _metrics(development), _metrics(selection), _metrics(holdout)
        row.update(
            status="evaluated",
            total=_metrics(trades),
            development=dev_m,
            selection=sel_m,
            holdout=hold_m,
            yearly=yearly,
            selection_score=_selection_score(sel_m, yearly),
            selection_p_value=_normal_positive_edge_pvalue(selection),
        )
        rows.append(row)
        print(
            f"[{ordinal:03d}/437] {strategy.strategy_id} {strategy.name}: "
            f"sel={sel_m['trades']} avgR={sel_m['avg_r']:.3f} PF={sel_m['profit_factor']:.2f}; "
            f"hold={hold_m['trades']} avgR={hold_m['avg_r']:.3f}"
        )

    _bh_qvalues(rows)

    evaluated = [r for r in rows if r["status"] == "evaluated"]
    for r in evaluated:
        sel = r["selection"]
        r["selection_gate"] = bool(
            int(sel["trades"]) >= MIN_SELECTION_TRADES
            and float(sel["avg_r"]) > 0.0
            and float(sel["profit_factor"]) > 1.0
        )

    ranked = sorted(
        [r for r in evaluated if r.get("selection_gate")],
        key=lambda r: (float(r["selection_score"]), float(r["selection"]["avg_r"]), int(r["selection"]["trades"])),
        reverse=True,
    )
    frozen_top = ranked[:MAX_SURVIVORS]
    top_ids = {str(r["strategy_id"]) for r in frozen_top}
    for rank, r in enumerate(ranked, start=1):
        r["selection_rank"] = rank
        r["top_25_selection"] = str(r["strategy_id"]) in top_ids

    survivors: list[dict[str, object]] = []
    for r in frozen_top:
        hold = r["holdout"]
        holdout_gate = bool(
            int(hold["trades"]) >= MIN_HOLDOUT_TRADES
            and float(hold["avg_r"]) > 0.0
            and float(hold["profit_factor"]) > 1.0
            and float(hold["max_drawdown_r"]) <= 25.0
        )
        r["holdout_gate"] = holdout_gate
        if holdout_gate:
            r["status"] = "survived"
            survivors.append(
                {
                    "strategy_id": r["strategy_id"],
                    "name": r["name"],
                    "category": r["category"],
                    "selection_rank": r["selection_rank"],
                    "selection_score": r["selection_score"],
                    "selection": r["selection"],
                    "holdout": r["holdout"],
                }
            )
        else:
            r["status"] = "rejected_holdout"

    # Strategies outside the frozen top quartile cannot be rescued by the holdout.
    for r in evaluated:
        if r.get("selection_gate") and not r.get("top_25_selection"):
            r["status"] = "rejected_below_top_quartile"
        elif not r.get("selection_gate"):
            r["status"] = "rejected_selection_gate"

    report = {
        "schema_version": 1,
        "methodology": {
            "canonical_strategy_count": 437,
            "parameter_grid_variants": 0,
            "parameter_trials_per_strategy": 1,
            "maximum_survivors": MAX_SURVIVORS,
            "survivor_fraction_cap": MAX_SURVIVORS / 437,
            "signal_timing": "closed-bar signal; next-bar-open entry",
            "intrabar_collision": "stop assumed first when stop and target both touched",
            "spread_bps": 1.5,
            "slippage_bps": 0.5,
            "stop_atr": 1.20,
            "reward_risk": 1.70,
            "development_period": f"before {SELECTION_START.isoformat()}",
            "selection_period": f"{SELECTION_START.isoformat()} to {HOLDOUT_START.isoformat()}",
            "untouched_holdout_period": f"{HOLDOUT_START.isoformat()} onward",
            "selection_policy": "rank on selection only; freeze top 109; holdout may reject but never replace",
            "anti_lookahead": "for every executable strategy, compare full-history pre-cut signals with prefix-only signals",
            "multiple_testing": "one-sided positive-edge p-values with Benjamini-Hochberg q-values reported",
        },
        "data": {
            "xau_m15": _coverage(xau),
            "dxy_h1": _coverage(dxy),
            "source_repository": SOURCE_REPO,
            "available_inputs": sorted(available),
        },
        "counts": {
            "canonical": 437,
            "evaluated": sum(r.get("status") in {"evaluated", "survived", "rejected_holdout", "rejected_below_top_quartile", "rejected_selection_gate"} for r in rows),
            "rejected_missing_inputs": sum(r.get("status") == "rejected_missing_inputs" for r in rows),
            "rejected_lookahead": sum(r.get("status") == "rejected_lookahead" for r in rows),
            "rejected_execution": sum(r.get("status") == "rejected_execution" for r in rows),
            "survivors": len(survivors),
        },
        "survivors": survivors,
        "strategies": rows,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    SURVIVOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    SURVIVOR_PATH.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": str(REPORT_PATH),
                "canonical_count": 437,
                "max_survivors": MAX_SURVIVORS,
                "survivor_count": len(survivors),
                "survivors": survivors,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Completed: {len(survivors)} survivors (cap={MAX_SURVIVORS})")
    print(f"Report: {REPORT_PATH}")
    print(f"Survivors: {SURVIVOR_PATH}")


if __name__ == "__main__":
    main()
