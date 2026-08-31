# XAU/USD Multi-Agent Signal Company

A research-first XAU/USD signal engine. The company researches a large strategy universe, validates strategies with conservative trade-lifecycle simulation and expanding walk-forward folds, diagnoses the current market across multiple timeframes and macro context, then lets a boss/risk layer authorize Telegram BUY/SELL signals with Entry, TP and SL.

## Company structure

1. **Market Data Bot** — pulls XAU/USD OHLC candles and optional macro series.
2. **Strategy Research Lab** — samples up to `MAX_CANDIDATES` from a 27k+ universe spanning 11 strategy families.
3. **Backtest Auditor** — simulates next-bar entries, spread, slippage, ATR stops/targets, non-overlapping trades and conservative same-bar TP/SL ordering.
4. **Regime Bot** — classifies trend-up, trend-down, range or volatile conditions.
5. **Multi-Timeframe Team** — independently analyzes 1m, 5m, 15m, 1h and 4h horizons, with higher timeframes weighted more heavily.
6. **Trend / Momentum Team** — trend, triple-trend, RSI-trend, pullback and momentum evidence.
7. **Breakout Team** — Donchian, Bollinger and volatility-breakout evidence.
8. **Mean-Reversion Team** — RSI/z-score, Bollinger-reversion and range-fade evidence.
9. **Price/Structure Team** — candle and higher-high/lower-low confirmation.
10. **Macro Team** — USD and Treasury-yield context when those feeds are available.
11. **Risk Team** — volatility/session guards plus high-impact-news vetoes.
12. **Strategy Selector + Boss** — chooses the researched strategy best suited to the current market and computes final risk geometry.
13. **Outcome & Calibration Bot** — permanently records emitted signals, resolves later TP/SL outcomes, tracks forward win rate/Brier score and calibrates future release confidence.

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

## Main settings

```text
MAX_CANDIDATES=20000
RESEARCH_CATALOG_SIZE=600
WALK_FORWARD_FOLDS=4
MIN_WALK_FORWARD_FOLDS=2
SPREAD_BPS=1.5
SLIPPAGE_BPS=0.5
BACKTEST_STOP_ATR=1.20
BACKTEST_REWARD_RISK=1.70
OUTCOME_DB_PATH=data/xau_outcomes.sqlite3
OUTCOME_MAX_AGE_HOURS=72
CALIBRATION_BIN_WIDTH=0.05
CALIBRATION_PRIOR_STRENGTH=20
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

- Add stronger automatic economic-calendar ingestion instead of manually configured event timestamps.
- Add stale-candle/market-hours guards, API retry/backoff and health monitoring.
- Add stronger multiple-testing/overfit controls for the 20k-candidate research process.
- Persist the researched strategy catalog across restarts to reduce startup work.
- Deploy only after sustained paper/shadow validation.

This is a research framework, not a guarantee of profit.
