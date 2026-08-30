# XAU/USD Multi-Agent Signal Company

A research-first, multi-agent XAU/USD signal engine. It evaluates thousands of parameterized strategy variants, detects the current market regime, asks specialized strategy desks for votes, lets risk/volatility desks veto bad conditions, and only emits a Telegram BUY/SELL signal when the boss agent sees enough agreement.

## The company (10 bots)

1. **Market Data Bot** — pulls XAU/USD OHLC candles from Twelve Data.
2. **Strategy Research Bot** — tests up to `MAX_CANDIDATES` strategy variants and ranks out-of-sample stability.
3. **Regime Bot** — classifies trend-up, trend-down, range, or volatile conditions.
4. **Trend Bot** — EMA alignment/trend-following vote.
5. **Breakout Bot** — Donchian breakout + momentum confirmation.
6. **Mean-Reversion Bot** — RSI + z-score exhaustion vote.
7. **Momentum Bot** — multi-horizon rate-of-change vote.
8. **Price/Structure Bot** — candle and higher-high/lower-low confirmation.
9. **Risk Guards** — volatility/session filters and veto logic.
10. **Boss Bot** — combines votes, applies consensus/confidence gates, computes ATR-based SL/TP, and authorizes Telegram delivery.

## Important design rule

The system does **not** select the strategy with the highest in-sample win rate. Searching thousands of variants can overfit history. The research bot uses a chronological train/validation split, penalizes train-vs-validation deterioration, trading friction, and tiny sample sizes, then feeds family-quality scores to the live desks.

This is a research framework, not a guarantee of profit. Run it in paper mode and validate it across enough unseen data before considering live use.

## Data

Twelve Data currently lists **Gold Spot / US Dollar (`XAU/USD`)** and supports intraday history. A request is capped at 5,000 data points, so the app caps `OUTPUT_SIZE` accordingly.

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

Keep `PAPER_MODE=true` initially. In paper mode the signal is printed instead of sent to Telegram.

## Run

```bash
python main.py
```

## Telegram signal format

```text
XAU COMPANY SIGNAL
Symbol: XAU/USD
Action: BUY
Entry: 0000.00
TP: 0000.00
SL: 0000.00
Confidence: 78.4%
Regime: trend_up
R:R: 1.80
```

## Next production upgrades

- Walk-forward/embargoed cross-validation and probability-of-backtest-overfitting metrics.
- Economic-calendar/news-event vetoes for CPI, NFP, FOMC and major geopolitical shocks.
- DXY, real-yield, ETF-flow, positioning and cross-asset features.
- Persistent results DB so strategy weights learn from real forward outcomes.
- Multi-timeframe confirmation (1m/5m/15m/1h/4h).
- Alert deduplication, health checks and deployment monitoring.
