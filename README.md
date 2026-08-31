# XAU/USD Multi-Agent Signal Company

A research-first XAU/USD signal engine. The company researches an adaptive strategy universe, validates candidates with conservative lifecycle simulation and chronological walk-forward folds, diagnoses current market conditions, then lets the Selector/Boss authorize at most two qualified Telegram signals per local trading day.

The framework produces research signals only. It does not place brokerage orders and does not guarantee profit.

## Signal pipeline

1. **Market Data Bot** fetches XAU/USD OHLC data, optional DXY/yield context, and a fresh reference price.
2. **Market Data Quality Guard** validates candle geometry/timestamps, removes still-forming candles, rejects stale required frames, and blocks closed/maintenance sessions.
3. **Strategy Research Lab** evaluates up to `MAX_CANDIDATES` from the original 27k+ seed universe plus persistent discoveries.
4. **Backtest Auditor** uses next-bar entry, spread/slippage, ATR stop/target geometry, non-overlapping trades and conservative same-bar TP/SL ordering.
5. **Strategy Discovery & Evolution Bot** creates cross-family experiments after research. New ideas start `EXPERIMENTAL` and cannot trade immediately.
6. **Overfit Auditor** applies hard OOS quality gates plus a multiplicity penalty based on the persistent lifetime number of tested strategies.
7. **Regime / specialist / timeframe / macro / risk desks** analyze the current completed research candle and surrounding context.
8. **Strategy Selector** scores only live-eligible strategies. Generic analyst support excludes timeframe and macro votes because those have separate score terms.
9. **Boss** requires the configured consensus exactly, uses a fresh reference price for Entry, and uses the same ATR stop multiplier and R:R assumptions as the research backtester.
10. **Outcome & Calibration Bot** calibrates the Selector score from confirmed forward outcomes.
11. **Trade Frequency Guard + outcome ledger** atomically reserve the setup and daily slot before delivery.
12. **Telegram** receives the final authorized signal. Telegram is delivery infrastructure, not a trading employee.

## Data synchronization and setup identity

All strategy, regime and specialist logic works from completed candles only. Required `RESEARCH_INTERVAL`, 1h and 4h context must be present and fresh, and at least one fresh 1m/5m execution frame must exist.

A company setup is identified by **symbol + completed research-candle timestamp**. Only one authorization may be emitted for that research candle even if, during later polling, the preferred strategy, direction, quote, stop or target changes. `MAX_SIGNAL_DELAY_MINUTES` also prevents sending a setup too long after its research candle closed.

The final Entry comes from Twelve Data's current-price endpoint rather than simply reusing the last candle close. The framework still labels the signal as research-only because a reference price is not a guaranteed executable broker fill.

## Adaptive strategy discovery

The original 27k+ parameter universe is a stable seed library, not the ceiling.

After each research cycle, the Evolution Bot creates structurally new cross-family ensembles from strong live-eligible seed strategies. Current modes are:

- `confirm` — both parents must agree.
- `primary_filter` — the primary may fire only if the secondary does not oppose it.
- `consensus_or` — either may fire if the other does not contradict it.

New entries are generated after the current research run and stored as `EXPERIMENTAL`. A later run must backtest them and place them into the research staging catalog before the Overfit Auditor can consider promotion.

Only `PROMOTED` discoveries can enter the Boss-visible catalog. A promoted discovery that later fails its audit becomes `QUARANTINED` and immediately loses live eligibility. Promoted and quarantined history is protected from experimental-library pruning. When the experimental queue reaches capacity, old unpromoted experiments rotate out so discovery can continue.

The evolution library uses atomic file replacement and a lock on supported Unix/Linux hosts. Lifetime tested-trial count is persisted separately so the multiple-testing penalty grows as the search expands instead of resetting every research cycle.

The current Overfit Auditor is a transparent selection-bias guard, not a formal Deflated Sharpe Ratio or Probability-of-Backtest-Overfitting implementation.

## Research and live risk model

Research uses expanding chronological walk-forward validation. A candidate must have enough usable OOS folds/trades to survive. Its score incorporates OOS hit rate, sample size, profit factor, average R, expectancy, drawdown, loss streak, fold coverage/stability and regime diversity.

The live Selector's sample-size trust and evolved-strategy promotion gate use the **actual OOS trade sample**, not the larger total historical backtest trade count.

Research lifecycle rules include:

- signal known only after a completed candle;
- entry on the next bar;
- spread and slippage charged;
- stop/target based on ATR information known at the signal candle;
- same candle touching TP and SL is treated as stop-first;
- only one simulated position at a time;
- family-specific maximum holding horizon.

The Boss now uses the same `BACKTEST_STOP_ATR` and `BACKTEST_REWARD_RISK` values as research. The forward outcome ledger also receives the selected strategy's research holding horizon, so a trade cannot later be called a win after the corresponding backtest would already have timed out.

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

Timeframe and macro votes are not counted again as generic analysts. The highest score still represents a **heuristic selection score**, not a mathematically guaranteed win probability. Confirmed forward outcomes are used afterward for calibration.

## Forward outcome and delivery safety

Before a qualifying signal is sent, SQLite uses `BEGIN IMMEDIATE` to atomically check both setup deduplication and the daily signal cap, then reserves the slot.

Delivery states are:

- `RESERVED` — slot claimed immediately before delivery;
- `SENT` — Telegram/paper delivery confirmed;
- `UNKNOWN` — delivery may have succeeded but acknowledgement was lost; the slot remains consumed to prevent duplicate sends;
- `FAILED` — explicit reusable failed reservation state.

On restart, abandoned reservations become `UNKNOWN`, which is conservative: they cannot be resent or reuse the daily slot, and they are excluded from probability calibration.

The outcome bot resolves only confirmed `SENT` signals for calibration. If the emission occurred inside a 1-minute OHLC candle that hit TP/SL, the result is marked `AMBIGUOUS` because OHLC cannot tell whether the hit happened before or after the actual send. Ambiguous/expired/unknown-delivery rows do not train calibration.

Runtime fetches enough 1-minute history to reconstruct up to `OUTCOME_MAX_AGE_HOURS` after downtime; configuration restricts that setting to at most 72 hours under the current 5,000-bar retrieval limit.

## Trade frequency policy

The company never forces a trade.

- Monday-Friday accounting in `TRADE_TIMEZONE`.
- Default: `America/Chicago`.
- 0, 1 or 2 qualified signals can be emitted in a local day.
- `MAX_TRADES_PER_DAY=2` is a hard maximum.
- Atomic reservation prevents two processes from simultaneously claiming the same final slot in the same SQLite database.

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

Run:

```bash
python main.py
```

CI compiles `main.py`, `xau_company` and all tests before running the complete pytest suite.

## Remaining production work

- Source-backed research scout for new public strategy concepts; current evolution is internal recombination only.
- Automatic high-impact economic-calendar ingestion; current event blackout list is manually configured.
- Formal Deflated-Sharpe/PBO-style diagnostics and purged/embargoed validation.
- Persistent robust research catalog across restarts.
- Strategy correlation/agreement model to distinguish independent strategy confirmation from redundant signals.
- Verified durable hosting/service deployment and durable runtime storage before any claim of continuous operation.

This is a research framework, not a guarantee of profit.
