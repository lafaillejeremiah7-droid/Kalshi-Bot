# XAU/USD Multi-Agent Signal Company

A research-first XAU/USD signal engine. The company researches a large adaptive strategy universe, validates candidates with conservative lifecycle simulation and chronological walk-forward folds, diagnoses current market conditions, then lets the Selector/Boss authorize at most two qualified Telegram signals per local trading day.

The framework produces research signals only. It does not place brokerage orders and does not guarantee profit.

## Signal pipeline

1. **Market Data Bot** fetches XAU/USD OHLC data, optional DXY/yield context, and a fresh reference price.
2. **Market Data Quality Guard** validates candle geometry/timestamps, removes still-forming candles, rejects stale required frames, and blocks closed/maintenance sessions.
3. **Strategy Research Lab** evaluates up to `MAX_CANDIDATES` from the original 27,801 seed variants plus persistent evolved and invented strategies.
4. **Backtest Auditor** uses next-bar entry, spread/slippage, ATR stop/target geometry, non-overlapping trades and conservative same-bar TP/SL ordering.
5. **Strategy Discovery & Evolution Bot** combines validated strategies into cross-family ensembles. New ideas start `EXPERIMENTAL`.
6. **Strategy Invention Bot** creates genuinely new formula families from primitive market features, then creates parameter variants of those formulas. It has 1,980 structural family templates in its current grammar.
7. **Overfit Auditor** applies hard OOS quality gates plus a multiplicity penalty based on the persistent lifetime number of tested strategies. Evolution and Invention share the same gate.
8. **Regime / specialist / timeframe / macro / risk desks** analyze the current completed research candle and surrounding context.
9. **Strategy Selector** scores only live-eligible strategies. Experimental dynamic strategies are invisible until explicitly promoted.
10. **Boss** requires the configured consensus exactly, uses a fresh reference price for Entry, and uses the same ATR stop multiplier and R:R assumptions as the research backtester.
11. **Outcome & Calibration Bot** calibrates the Selector score from confirmed forward outcomes.
12. **Trade Frequency Guard + outcome ledger** atomically reserve the setup and daily slot before delivery.
13. **Telegram** receives the final authorized signal. Telegram is delivery infrastructure, not a trading employee.

The animated dashboard now represents **28 working employees**, including the separate Strategy Invention Bot. Dashboard telemetry remains read-only/fail-open and never participates in signal authorization.

## Data synchronization and setup identity

All strategy, regime and specialist logic works from completed candles only. Required `RESEARCH_INTERVAL`, 1h and 4h context must be present and fresh, and at least one fresh 1m/5m execution frame must exist.

A company setup is identified by **symbol + completed research-candle timestamp**. Only one authorization may be emitted for that research candle even if, during later polling, the preferred strategy, direction, quote, stop or target changes. `MAX_SIGNAL_DELAY_MINUTES` prevents sending a setup too long after its research candle closed.

The final Entry comes from Twelve Data's current-price endpoint rather than simply reusing the last candle close. The framework still labels the signal as research-only because a reference price is not a guaranteed executable broker fill.

Cached DXY/yield frames are revalidated on every decision cycle. The API call may be rate-limited to every five cycles, but stale cached macro data is discarded instead of continuing to influence a decision.

## Strategy universe

### Fixed seed universe

The original research engine contains **11 strategy families and 27,801 fixed parameterized variants**. The configured default evaluates up to 20,000 candidates per research cycle with balanced sampling and retains a bounded research catalog.

### Strategy Discovery & Evolution Bot

Evolution creates new cross-family ensembles from validated strategies. Current modes are:

- `confirm` — both parents must agree.
- `primary_filter` — the primary may fire only if the secondary does not oppose it.
- `consensus_or` — either may fire if the other does not contradict it.

Promoted invented strategies may be used as finite parents in an evolved strategy, but recursive ensemble-of-ensemble chains are rejected to keep evaluation cost and audit attribution bounded.

### Strategy Invention Bot

Invention is separate from Evolution. It does not merely recombine existing strategy outputs. It builds formulas from three directional feature blocks plus a volatility gate and decision logic.

Current directional feature primitives include:

- EMA gap and EMA slope;
- momentum;
- RSI trend and RSI reversion;
- z-score trend and z-score reversion;
- Donchian breakout and fade;
- candle impulse;
- range location.

Current gates are `none`, `atr_expansion`, `atr_normal`, and `atr_compression`. Current logic modes are `all`, `majority`, and `lead_confirm`. Choosing 3 of 11 features across 4 gates and 3 logic modes creates **1,980 distinct structural family templates** before parameter variants.

Defaults are:

- `INVENTED_FAMILIES_PER_CYCLE=6`
- `INVENTED_VARIANTS_PER_FAMILY=8`
- up to 48 newly persisted invented variants per research cycle while capacity remains.

Every invented variant starts `EXPERIMENTAL`, is backtested on a later research run, and must pass the shared Overfit Auditor before it can enter the Selector-visible catalog. A promoted invention that later deteriorates becomes `QUARANTINED`. The persistent invention cursor recovers from the library itself if metadata lags after a crash, preventing duplicate family regeneration.

## Promotion and multiple-testing safety

Only the original seed families are inherently eligible for the research live catalog. `ensemble` and `invented` candidates require a matching `PROMOTED` key from their persistent library.

The Overfit Auditor checks multiplicity-adjusted score, profit factor, average R, actual OOS trade count, walk-forward dispersion, train/OOS gap, drawdown, loss streak and minimum fold count.

A single persistent lifetime tested-trial ledger is used for both Evolution and Invention, so adding a new search source cannot reset the multiple-testing penalty.

Promoted and quarantined history is protected from experimental-library pruning. Experimental queues rotate old unpromoted entries when full so discovery can continue.

## Research and live risk model

Research uses expanding chronological walk-forward validation. A candidate must have enough usable OOS folds/trades to survive. Its score incorporates OOS hit rate, sample size, profit factor, average R, expectancy, drawdown, loss streak, fold coverage/stability and regime diversity.

Research lifecycle rules include:

- signal known only after a completed candle;
- entry on the next bar;
- spread and slippage charged;
- stop/target based on ATR information known at the signal candle;
- same candle touching TP and SL is treated as stop-first;
- only one simulated position at a time;
- family-specific maximum holding horizon.

The Boss uses the same `BACKTEST_STOP_ATR` and `BACKTEST_REWARD_RISK` values as research.

Forward outcome tracking is anchored to the **end of the research setup candle**, not Telegram send time. If neither TP nor SL is hit by the research holding horizon, the forward ledger uses the last completed close at/before the horizon and classifies the timeout as WIN/LOSS in the same directional way as the research backtester. `EXPIRED` is reserved for cases where sufficient close data is unavailable.

## Selector evidence

For every active live-eligible strategy, the Selector combines:

- walk-forward OOS hit rate;
- research score;
- current family/regime fit and regime-specific OOS history;
- specialist directional support/opposition;
- weighted 1m/5m/15m/1h/4h alignment;
- DXY/yield macro alignment;
- profit factor;
- walk-forward stability;
- lifecycle quality from average R, drawdown and loss streak;
- OOS sample-size trust.

Timeframe and macro votes are not counted again as generic analysts. Invented and ensemble strategies use neutral structural regime priors unless/until their own historical regime evidence differentiates them. The highest score remains a **heuristic selection score**, not a guaranteed win probability. Confirmed forward outcomes are used afterward for calibration.

## Forward outcome and delivery safety

Before a qualifying signal is sent, SQLite uses `BEGIN IMMEDIATE` to atomically check both setup deduplication and the daily signal cap, then reserves the slot.

Delivery states are:

- `RESERVED` — slot claimed immediately before delivery;
- `SENT` — Telegram/paper delivery confirmed;
- `UNKNOWN` — delivery may have succeeded but acknowledgement was lost; the slot remains consumed to prevent duplicate sends;
- `FAILED` — definitive delivery rejection; the reservation may be replaced after configuration is corrected.

On restart, abandoned reservations become `UNKNOWN`, which is conservative: they cannot be resent or reuse the daily slot, and they are excluded from probability calibration.

If the emission occurred inside a resolution OHLC candle that hit TP/SL, the result is marked `AMBIGUOUS` because OHLC cannot establish whether the hit happened before or after the actual send. Ambiguous and unknown-delivery rows do not train calibration.

Outcome recovery prefers completed **1-minute** history and automatically falls back to completed **5-minute** history if the 1-minute feed is unavailable/unhealthy. Output size is calculated from the configured outcome window and actual resolution interval, capped at the provider's 5,000-bar limit.

Telegram signal text is capped at 3,900 characters. Critical Action/Entry/TP/SL/confidence/risk fields are emitted before optional reasons, so verbose strategy labels or explanations cannot trigger a predictable Telegram message-length rejection.

## Trade frequency policy

The company never forces a trade.

- Monday-Friday accounting in `TRADE_TIMEZONE`.
- Default: `America/Chicago`.
- 0, 1 or 2 qualified signals can be emitted in a local day.
- `MAX_TRADES_PER_DAY=2` is a hard maximum.
- Atomic reservation prevents two processes sharing the same SQLite database from simultaneously claiming the same final slot.

## Core settings

```text
RESEARCH_INTERVAL=15min
TIMEFRAMES=1min,5min,15min,1h,4h
OUTPUT_SIZE=3000
CONTEXT_OUTPUT_SIZE=500
POLL_SECONDS=60
MIN_CONFIDENCE=0.72
MIN_CONSENSUS=3
MAX_CANDIDATES=20000
RESEARCH_CATALOG_SIZE=600
WALK_FORWARD_FOLDS=4
MIN_WALK_FORWARD_FOLDS=2
RESEARCH_EVERY_CYCLES=60

ENABLE_STRATEGY_EVOLUTION=true
STRATEGY_LIBRARY_PATH=data/discovered_strategies.json
DISCOVERIES_PER_CYCLE=250
DISCOVERY_LIBRARY_SIZE=5000

ENABLE_STRATEGY_INVENTION=true
INVENTION_LIBRARY_PATH=data/invented_strategies.json
INVENTED_FAMILIES_PER_CYCLE=6
INVENTED_VARIANTS_PER_FAMILY=8
INVENTION_LIBRARY_SIZE=4000

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

Fill in `TWELVE_DATA_API_KEY`, `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`. Keep `PAPER_MODE=true` until enough forward evidence exists.

Run the company:

```bash
python main.py
```

Run the animated operations floor separately:

```bash
python dashboard_server.py
```

Then open `http://127.0.0.1:8080`.

CI compiles `main.py`, `xau_company` and all tests before running the complete pytest suite.

## Remaining production work

- Automatic high-impact economic-calendar ingestion; current event blackout list is manually configured.
- Formal Deflated-Sharpe/PBO-style diagnostics and purged/embargoed validation.
- Persistent robust research catalog across restarts.
- Strategy correlation/agreement model to distinguish independent confirmation from redundant strategies.
- Optional source-backed research scout for public strategy concepts; current Invention Bot generates formulas internally rather than browsing external research.
- Verified durable hosting/service deployment and durable runtime storage before any claim of continuous operation.

This is a research framework, not a guarantee of profit.
