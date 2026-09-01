# Research system blueprint

## Scope

This repository begins as a **research-only system** for Kalshi `KXBTC15M`. It has no live order-placement code and should not receive trading permissions. Its job is to create evidence: clean synchronized data, exact settlement reconstruction, deterministic replay, calibrated probability research, and realistic cost/execution simulation.

## Verified platform facts that affect design

As of the 2026 API/docs used to design this codebase:

1. Production Trade API: `https://external-api.kalshi.com/trade-api/v2`.
2. Production WebSocket: `wss://external-api-ws.kalshi.com/trade-api/ws/v2`.
3. Kalshi has migrated price and quantity APIs to fixed-point dollar/contract strings; research code must not assume integer cents or integer contract counts.
4. `orderbook_delta` sends a snapshot first and then sequenced incremental updates. Sequence gaps invalidate local book state until a fresh snapshot is obtained.
5. Older markets/trades/orders move behind historical endpoints; the target live window is about three months.
6. BTC crypto settlement uses CF Benchmarks RTI data. The documented BTC15m rule is a simple average of the 60 one-second BRTI observations immediately before expiration, with the official final value rounded to two decimals. YES wins when that value is at least the target.
7. Fee configuration can vary by series/time. The research store must persist fee metadata effective at the decision/fill time rather than hard-code one fee schedule.
8. Kalshi exposes order queue position for a user's resting orders, confirming price-time priority. Historical public queue position is not available, so maker replay requires explicit queue assumptions unless the research account captured its own queue data live.

## Why raw capture is mandatory

Historical REST trades/candles cannot reconstruct exact historical queue state or every L2 book mutation. Therefore the system must capture raw WebSocket snapshots/deltas prospectively. Every raw message should be archived unchanged before normalization so parser bugs can be corrected later without losing evidence.

## Data layers

### Layer 0 — immutable raw capture

Write original websocket/REST payloads with:
- local receive timestamp in nanoseconds
- source/connection id
- websocket subscription id and sequence where available
- content hash
- raw payload

Rotate by source/date/hour. Never rewrite these files.

### Layer 1 — canonical event log

Normalize into typed events with:
- `event_ts_ns`: source/exchange timestamp
- `recv_ts_ns`: when this process received it
- market ticker/source/kind
- fixed-point values parsed with `Decimal`

SQLite/WAL is the MVP local event log because it is dependency-light, transactional and deterministic. When capture volume grows, retain the same event schema while partitioning raw/canonical data to Parquet and querying with DuckDB/Polars.

### Layer 2 — synchronized research panels

Create decision-time rows only by as-of joins on **receive time**. Each row records freshness/age of each source. Never forward-fill indefinitely; stale features become missing and the row can be rejected.

### Layer 3 — labels and outcomes

Settlement/outcomes live in a logically separate table/dataframe. Training code joins labels only after feature creation to prevent accidental leakage.

## Initial source priority

Essential:
1. Kalshi market metadata/rules/target/open/close/result.
2. Kalshi L2 orderbook snapshots + deltas.
3. Kalshi public trades.
4. Exact BRTI stream/pass-through used by Kalshi.
5. At least two major BTC-USD spot venues with L2/trades.
6. Effective fee schedule/series fee changes.
7. Local high-resolution receive timestamps and connection health.

Second wave:
- CME BTC futures and/or liquid perp feeds
- incentives/rebates metadata
- user queue-position captures during later paper/tiny-live experiments
- network RTT and colocated/remote latency experiments

## Clock and timestamp policy

- system clock must be NTP-synchronized; record offset diagnostics
- use UTC internally
- preserve source timestamp and receive timestamp separately
- use integer nanoseconds internally; never naive datetimes
- decision replay orders events by receive time when testing what was knowable, while exchange/book reconstruction also enforces source sequence

## Book reconstruction invariant

For each subscription:
1. wait for snapshot
2. set `last_seq=snapshot.seq`
3. accept only `delta.seq == last_seq + 1`
4. if a gap occurs, mark the book invalid immediately
5. request a new snapshot / reconnect
6. do not produce model features from invalid state

## Train/validation/test protocol

Use chronological splits only. Recommended development loop:
- expanding or rolling train window
- next block = validation/model selection
- next block = walk-forward test
- preserve a final untouched holdout after architecture is stable

No random row split because adjacent seconds from the same 15-minute market are highly dependent.
Group by market/window to keep one contract from leaking across folds.

## Multiple-testing controls

Every hypothesis must have an experiment id and predeclared metrics. Track how many variants were tried. Favor simple nested comparisons:
- baseline diffusion
- + settlement state
- + external market features
- + Kalshi microstructure

If hundreds of variants are tried, require stronger evidence (bootstrap confidence intervals, false-discovery controls, and a final untouched holdout).

## Replay requirements before strategy claims

A valid backtest must model:
- historical visible depth
- taker price walking through levels
- partial fills
- maker queue uncertainty
- cancel/replace delay
- source and decision latency
- stale-data rejection
- fees effective at time of trade
- exact settlement
- overlapping consecutive markets and capital locking
- disconnects/sequence gaps

Report prediction quality separately from execution P&L.

## P&L attribution

Every simulated fill must be tagged to one source:
- directional probability edge
- spread capture
- complementary/pair arbitrage
- incentives/rebates

And subtract separately:
- taker/maker fees
- fee rounding
- slippage
- adverse selection
- latency impact
- failed/partial leg cost
- operational errors

A strategy whose positive total P&L comes only from incentives while trading P&L is negative is not accepted as alpha.

## Promotion gates

Research -> shadow probability stage only after an untouched out-of-sample sample satisfies, at minimum:
- >= 500 executable decisions
- Brier and log loss each improve at least 1% vs simple diffusion baseline
- ECE <= 3 percentage points
- positive net P&L after realistic fees/slippage
- positive P&L in >= 60% of walk-forward windows
- edge remains positive under deliberately worse latency and transaction-cost assumptions

These defaults are intentionally conservative and should be tightened as sample size increases.
