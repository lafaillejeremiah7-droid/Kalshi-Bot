from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from xau_company.adaptive_research import AdaptiveStrategyResearchAgent


DATA_URL = (
    "https://raw.githubusercontent.com/getdata-finance/"
    "xauusd-5m-ohlcv-metals-historical-data/main/XAUUSD_5m.csv"
)
REPORT_PATH = Path("real-data-overfit-audit.json")


def load_real_xauusd() -> pd.DataFrame:
    response = requests.get(DATA_URL, timeout=45)
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text))
    required = {"datetime", "open", "high", "low", "close"}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"Historical dataset missing columns: {sorted(missing)}")

    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["datetime", "open", "high", "low", "close"])
    raw = raw.sort_values("datetime").drop_duplicates("datetime", keep="last")
    raw = raw.set_index("datetime")

    # Production default research interval is 15 minutes. Build true 15m OHLC
    # from the public 5m feed so this audit exercises the same horizon class.
    df = raw.resample("15min", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            **({"volume": "sum"} if "volume" in raw.columns else {}),
        }
    )
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index()
    if len(df) < 3000:
        raise RuntimeError(f"Not enough 15m history for audit: {len(df)} rows")
    return df


def main() -> None:
    df = load_real_xauusd()
    lab = AdaptiveStrategyResearchAgent(
        max_candidates=4_000,
        catalog_size=600,
        walk_forward_folds=4,
        min_walk_forward_folds=2,
        spread_bps=1.5,
        slippage_bps=0.5,
        backtest_stop_atr=1.20,
        backtest_reward_risk=1.70,
        enable_evolution=False,
        enable_invention=False,
    )
    top = lab.run(df)

    leaked = []
    for score in lab.catalog:
        audit = lab.overfit_auditor.audit(score, tested_trials=max(1, lab.last_evaluated))
        if not audit.passed:
            leaked.append(
                {
                    "family": score.candidate.family,
                    "params": repr(score.candidate.params),
                    "reasons": list(audit.reasons),
                }
            )

    report = {
        "source": DATA_URL,
        "research_interval": "15min",
        "rows": len(df),
        "start": df["datetime"].iloc[0].isoformat(),
        "end": df["datetime"].iloc[-1].isoformat(),
        "universe_size": lab.last_universe_size,
        "evaluated": lab.last_evaluated,
        "scored_and_audited": lab.last_seed_audited,
        "overfit_risk_rejected": lab.last_seed_overfit_rejected,
        "live_eligible_before_catalog_cap": lab.last_seed_live_eligible,
        "live_catalog": len(lab.catalog),
        "top_count": len(top),
        "overfit_gate_leaks": len(leaked),
        "top": [
            {
                "family": s.candidate.family,
                "params": repr(s.candidate.params),
                "score": round(float(s.score), 6),
                "valid_hit_rate": round(float(s.valid_hit_rate), 6),
                "profit_factor": round(float(s.profit_factor), 6),
                "avg_r": round(float(s.avg_r_multiple), 6),
                "oos_trades": lab.overfit_auditor.oos_trade_count(s),
            }
            for s in top[:10]
        ],
        "leaked": leaked[:20],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    if lab.last_seed_audited == 0:
        raise SystemExit("FAIL: no strategies produced auditable walk-forward scores")
    if leaked:
        raise SystemExit(f"FAIL: {len(leaked)} rejected strategies leaked into live catalog")


if __name__ == "__main__":
    main()
