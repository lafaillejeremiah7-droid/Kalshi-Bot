# XAU/USD Multi-Agent Signal Company

A research-first XAU/USD signal engine built around a fixed catalog of **437 distinct strategy methodologies**. The company does not treat parameter changes as new strategies. Historical selection is chronological and fail-closed: a strategy must survive the canonical XAUUSD audit before it can appear in the live research catalog.

The framework produces research signals only. It does not guarantee profit.

## Canonical strategy policy

- Exactly **437** canonical strategy IDs: `S001` through `S437`.
- One ID represents one methodology.
- Parameter combinations never create additional strategy IDs.
- The historical audit may retain at most **109 strategies** (25% of 437, rounded down).
- Strategies requiring unavailable data are rejected rather than tested with fabricated proxy inputs.
- Live research reads only `xau_company/surviving_strategies.json`; an empty or missing survivor set means no strategy is live-eligible.

The canonical catalog is defined in `xau_company/canonical_strategies.py`. Deterministic signal logic is implemented in `xau_company/canonical_strategy_engine.py`.

## Historical audit

`scripts/canonical_437_backtest.py` evaluates the canonical catalog against historical XAUUSD data with these safeguards:

- closed-bar signals only;
- next-bar-open entry;
- spread and slippage charged;
- ATR-based stop and target geometry;
- stop-first handling when an OHLC bar touches stop and target;
- chronological development, selection, and untouched holdout periods;
- anti-lookahead prefix checks for every executable strategy;
- one statistical trial per canonical methodology;
- Benjamini-Hochberg multiple-testing diagnostics;
- top-quartile selection frozen before the holdout is inspected;
- the holdout may reject a selected strategy but can never promote a replacement.

The workflow `.github/workflows/canonical-437-backtest.yml` publishes:

- `canonical-437-backtest-report.json`
- `xau_company/surviving_strategies.json`

The survivor count can be lower than 109. It can never be higher.

## Signal pipeline

1. **Market Data Bot** fetches XAU/USD OHLC data, optional DXY/yield context, and a fresh reference price.
2. **Market Data Quality Guard** validates candle geometry/timestamps, drops still-forming candles, rejects stale required frames, and blocks closed/maintenance sessions.
3. **Strategy Research Lab** evaluates only historically surviving canonical strategies.
4. **Backtest/Overfit Auditor** applies chronological OOS quality gates to the survivor set.
5. **Regime, specialist, timeframe, macro, and risk desks** analyze current completed candles and context.
6. **Strategy Selector** scores only live-eligible canonical strategies.
7. **Boss** requires configured consensus and uses the same ATR stop multiplier and reward/risk assumptions used by research.
8. **Outcome & Calibration Bot** calibrates selection scores from confirmed forward outcomes.
9. **Trade Frequency Guard + outcome ledger** atomically reserve the setup and daily slot before delivery.
10. **Telegram or paper mode** receives the final authorized signal.

The dashboard is read-only telemetry and never authorizes a trade.

## Data synchronization and setup identity

All strategy, regime, and specialist logic operates on completed candles. Required `RESEARCH_INTERVAL`, 1h, and 4h context must be present and fresh, and at least one fresh 1m/5m execution frame must exist.

A setup is identified by **symbol + completed research-candle timestamp**. Only one authorization may be emitted for a research candle even if a later polling cycle changes the preferred strategy, quote, stop, or target.

Cached DXY/yield frames are revalidated on every decision cycle. Stale optional macro data is discarded rather than allowed to influence a decision.

## Research and live risk model

Research uses chronological walk-forward validation. The lifecycle simulator enforces:

- signal known only after a completed candle;
- entry on the next bar;
- spread and slippage;
- stop/target derived from information available at the signal candle;
- conservative stop-first same-bar collision handling;
- one simulated position at a time;
- strategy-specific maximum holding horizon.

The Boss uses the same `BACKTEST_STOP_ATR` and `BACKTEST_REWARD_RISK` values as research.

## Forward outcome and delivery safety

Forward outcome tracking is anchored to the end of the research setup candle. If neither TP nor SL is reached by the research holding horizon, the last usable close at or before the horizon is used consistently with the research lifecycle.

Before a signal is sent, SQLite atomically checks setup deduplication and the daily cap, then reserves the slot. Delivery states distinguish confirmed sends, definitive failures, and uncertain delivery so duplicate messages are not emitted after crashes.

Outcome recovery prefers completed 1-minute history and falls back to completed 5-minute history when needed.

## Trade frequency policy

The company never forces a trade.

- Monday-Friday accounting in `TRADE_TIMEZONE`.
- Default timezone: `America/Chicago`.
- Zero, one, or two qualified signals may be emitted in a local trading day.
- `MAX_TRADES_PER_DAY=2` is a hard maximum.

## Core settings

```text
RESEARCH_INTERVAL=15min
TIMEFRAMES=1min,5min,15min,1h,4h
OUTPUT_SIZE=3000
CONTEXT_OUTPUT_SIZE=500
POLL_SECONDS=60
MIN_CONFIDENCE=0.72
MIN_CONSENSUS=3
WALK_FORWARD_FOLDS=4
MIN_WALK_FORWARD_FOLDS=2
RESEARCH_EVERY_CYCLES=60

OVERFIT_MIN_ADJUSTED_SCORE=0.60
OVERFIT_MIN_PROFIT_FACTOR=1.15
OVERFIT_MIN_AVG_R=0.05
OVERFIT_MIN_TRADES=40
OVERFIT_MAX_WF_STD=0.12
OVERFIT_MAX_TRAIN_VALID_GAP=0.15
OVERFIT_MAX_DRAWDOWN_R=10.0
OVERFIT_MAX_LOSS_STREAK=7

SPREAD_BPS=1.5
SLIPPAGE_BPS=0.5
BACKTEST_STOP_ATR=1.20
BACKTEST_REWARD_RISK=1.70
OUTCOME_DB_PATH=data/xau_outcomes.sqlite3
OUTCOME_MAX_AGE_HOURS=72
CALIBRATION_BIN_WIDTH=0.05
CALIBRATION_PRIOR_STRENGTH=20
TRADE_TIMEZONE=America/Chicago
MAX_TRADES_PER_DAY=2
MAX_STALE_MULTIPLIER=4.0
MAX_SIGNAL_DELAY_MINUTES=5
PAPER_MODE=true
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

Run the company:

```bash
python main.py
```

Run the dashboard separately:

```bash
python dashboard_server.py
```

CI compiles the runtime and runs the complete pytest suite.

## Safety status

The live company should remain disabled while a canonical historical audit is being rebuilt or while the survivor file is empty. Historical results are evidence, not a guarantee of future performance.
