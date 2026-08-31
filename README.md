# XAU/USD Multi-Agent Signal Company

A research-first XAU/USD signal engine. The company researches a large and now **adaptive** strategy universe, validates candidates with conservative trade-lifecycle simulation and expanding walk-forward folds, diagnoses the current market across multiple timeframes and macro context, then lets a boss/risk layer authorize Telegram BUY/SELL signals with Entry, TP and SL.

## Company structure

1. **Market Data Bot** — pulls XAU/USD OHLC candles and optional macro series.
2. **Strategy Research Lab** — evaluates up to `MAX_CANDIDATES` each research cycle. The original 27k+ parameter grid is the permanent seed universe, not a ceiling.
3. **Strategy Discovery & Evolution Bot** — recombines strong, structurally different strategies into new experimental ensembles and stores them in a persistent strategy library.
4. **Backtest Auditor** — simulates next-bar entries, spread, slippage, ATR stops/targets, non-overlapping trades and conservative same-bar TP/SL ordering.
5. **Overfit Auditor** — applies a search-size/multiple-testing penalty plus hard walk-forward, PF, average-R, drawdown and loss-streak promotion gates to evolved strategies.
6. **Regime Bot** — classifies trend-up, trend-down, range or volatile conditions.
7. **Multi-Timeframe Team** — independently analyzes 1m, 5m, 15m, 1h and 4h horizons, with higher timeframes weighted more heavily.
8. **Trend / Momentum Team** — trend, triple-trend, RSI-trend, pullback and momentum evidence.
9. **Breakout Team** — Donchian, Bollinger and volatility-breakout evidence.
10. **Mean-Reversion Team** — RSI/z-score, Bollinger-reversion and range-fade evidence.
11. **Price/Structure Team** — candle and higher-high/lower-low confirmation.
12. **Macro Team** — USD and Treasury-yield context when those feeds are available.
13. **Risk Team** — volatility/session guards plus high-impact-news vetoes.
14. **Strategy Selector + Boss** — chooses the researched strategy best suited to the current market and computes final risk geometry.
15. **Outcome & Calibration Bot** — permanently records emitted signals, resolves later TP/SL outcomes, tracks forward win rate/Brier score and calibrates future release confidence.
16. **Trade Frequency Guard** — allows qualified setups Monday-Friday only and enforces a persistent maximum of two emitted trades/signals per local trading day.

## Adaptive strategy discovery

The initial 27k+ strategy universe is no longer the maximum size of the research space. It is the stable **seed library**.

After each research cycle, the Strategy Discovery & Evolution Bot takes strong live-eligible non-ensemble survivors and creates structurally new cross-family experiments. Current evolution modes include:

- `confirm` — both parent strategies must agree on direction.
- `primary_filter` — a primary strategy may fire only when the second strategy does not oppose it.
- `consensus_or` — either parent may trigger when the other does not contradict it.

Every new structure is stored as `EXPERIMENTAL` in `STRATEGY_LIBRARY_PATH`. New experiments are generated **after** the current research cycle, which prevents a freshly generated idea from immediately influencing a live decision.

On a later research cycle the experiment enters the normal candidate universe and can be backtested. Importantly, an experimental strategy is **not Boss-visible merely because it scored well enough to enter the research staging catalog**. The Overfit Auditor must separately approve it.

Promotion requires the strategy to survive the normal next-bar lifecycle simulation, transaction-cost model and chronological walk-forward folds, then pass additional evolved-strategy gates for:

- a research score after an extra penalty that grows with the number of variants tested,
- profit factor,
- average R,
- minimum executed-trade sample,
- walk-forward dispersion,
- train-vs-OOS stability gap,
- maximum drawdown,
- maximum losing streak,
- and sufficient walk-forward fold coverage.

Only strategies with `PROMOTED` status are allowed into the Boss-visible live catalog. `EXPERIMENTAL` strategies remain research-only. If a previously promoted strategy later fails its audit, it becomes `QUARANTINED` and immediately loses live eligibility until future evidence is strong enough to pass again.

The persistent library preserves parent provenance plus the latest audit metrics/reasons. Storage pruning protects promoted strategies from being evicted by a flood of fresh experiments.

The current Overfit Auditor is a transparent selection-bias guard; it does **not** claim to be a formal Deflated Sharpe Ratio or Probability of Backtest Overfitting implementation. Those remain useful future research upgrades.

## Research model

The system does **not** choose the strategy with the highest raw historical win rate. Searching thousands of variants can overfit history.

Each candidate is evaluated using expanding walk-forward validation and a realistic lifecycle model:

- The signal must exist on a completed candle.
- The simulated trade enters on the **next candle open**.
- Spread and configurable slippage are charged.
- Stop-loss and take-profit are based on ATR-known information from the signal candle.
- If the same OHLC candle touches both TP and SL, the backtester assumes **SL happened first**.
- A second trade cannot open while the first simulated trade is active.
- Trades crossing a walk-forward fold boundary are excluded from that fold's validation.

Strategies are ranked using out-of-sample hit rate, fold stability, executed-trade sample size, profit factor, average R-multiple, net expectancy, maximum drawdown, losing-streak behavior and performance by market regime.

The live selector combines those historical metrics with current regime fit, 1m/5m/15m/1h/4h alignment, specialist analyst confirmation and optional USD/yield context.

## Forward outcome calibration

Every paper/live signal that is actually emitted is written to a local SQLite ledger. On later cycles the Outcome & Calibration Bot checks completed OHLC candles after that signal and resolves it as:

- `WIN` when TP is reached first.
- `LOSS` when SL is reached first.
- `LOSS` when one candle contains both TP and SL because intrabar ordering is unknown.
- `EXPIRED` after `OUTCOME_MAX_AGE_HOURS`; expired signals are not counted as wins or losses.

The original selector score is stored as `Selection confidence`. The release layer then computes a **Forward-calibrated confidence** using resolved signals from the same confidence bucket, with strategy-family + regime evidence used after enough samples exist. Bayesian shrinkage keeps a tiny sample from moving confidence aggressively.

If forward-calibrated confidence falls below `MIN_CONFIDENCE`, the signal is vetoed even if the research selector originally approved it.

The ledger also prevents the same execution-candle signal from being resent after a restart. The SQLite file is deliberately excluded from Git so live/paper outcome history remains runtime data rather than source code.

## Trade frequency policy

Trading is setup-dependent; the company never forces a trade just to hit a quota.

- Monday-Friday only in `TRADE_TIMEZONE`.
- Default timezone: `America/Chicago`.
- The company may emit **0, 1, or 2** qualified trades/signals in a day.
- `MAX_TRADES_PER_DAY=2` is a hard cap; after the second trade, later setups are vetoed until the next local trading day.
- The count is read from the persistent outcome ledger, so restarting the process does not reset the daily limit.

## Main settings

```text
MAX_CANDIDATES=20000
RESEARCH_CATALOG_SIZE=600
WALK_FORWARD_FOLDS=4
MIN_WALK_FORWARD_FOLDS=2
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
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in:

- `TWELVE_DATA_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Keep `PAPER_MODE=true` until enough forward evidence exists. In paper mode the company logs the signal and still records/resolves it for calibration.

## Run

```bash
python main.py
```

## Telegram signal includes

```text
XAU COMPANY SIGNAL
Symbol: XAU/USD
Action: BUY
Strategy: <selected researched strategy>
Daily trade slot: 1/2
OOS validation: <walk-forward result>
Profit factor: <PF>
Avg R: <average R> / Max DD: <drawdown in R>
Entry: 0000.00
TP: 0000.00
SL: 0000.00
Selection confidence: 00.0%
Forward-calibrated confidence: 00.0% from N resolved outcomes
Calibration Brier score: 0.0000
Regime: trend_up
R:R: 0.00
```

## Remaining production work

- Add a source-backed research scout that can ingest new public strategy concepts into the experimental pipeline instead of relying only on internal recombination.
- Add stronger automatic economic-calendar ingestion instead of manually configured event timestamps.
- Add stale-candle/market-hours guards, API retry/backoff and health monitoring.
- Add formal Deflated-Sharpe/PBO-style diagnostics and walk-forward embargo/purge controls as the evolving library grows.
- Persist the researched robust catalog across restarts to reduce startup work.
- Deploy only after sustained paper/shadow validation.

This is a research framework, not a guarantee of profit.
