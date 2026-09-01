# Kalshi BTC15m Research System

A research-only foundation for discovering and rejecting/validating possible edge in Kalshi's `KXBTC15M` Bitcoin 15-minute event contracts.

**This code does not place orders.** It intentionally contains no live trading endpoint or execution credential flow.

## What exists

- typed canonical research events with source and receive timestamps
- current Kalshi REST metadata/trade/fee-change reader
- authenticated Kalshi WebSocket research capture for `KXBTC15M` order books, public trades, and BRTI
- strict 2026 Kalshi fixed-point normalization with schema-drift quarantine
- exact BRTI final-minute 15-minute settlement-average/sample-count capture
- public Coinbase BTC-USD ticker capture with nanosecond timestamp and source-sequence retention
- public Kraken BTC/USD L2 reconstruction with transactional frame application and checksum retention
- subscription-id-aware Kalshi sequence-gap detection
- SQLite/WAL append-only canonical event store with content-hash deduplication
- immutable raw JSONL capture with a chained SHA-256 audit trail
- deterministic receive-time replay with an anti-lookahead ordering guard
- exact 60-sample settlement utilities
- closed-form partial-final-minute Brownian settlement baseline
- normalized distance/volatility/order-book/microprice/lead-lag math
- probability calibration metrics and cost-adjusted edge calculations
- conservative research promotion gates
- unit tests and CI

## Research doctrine

1. Store raw data before modeling.
2. Separate source timestamp from receive timestamp.
3. No lookahead. Features may only use data received by decision time.
4. Exact settlement beats generic candle-close assumptions.
5. Full L2 capture is required for credible microstructure replay.
6. Prediction quality and execution quality are measured separately.
7. Fees, slippage, latency, adverse selection and incentives are separate P&L lines.
8. If robust out-of-sample edge is not positive after costs, reject the idea.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Public Kalshi metadata probe:

```bash
kalshi-research probe --status open
```

Authenticated Kalshi research capture. The private key remains a local file and is never stored in the repository:

```bash
export KALSHI_API_KEY_ID='your-key-id'
export KALSHI_PRIVATE_KEY_PATH='/absolute/path/to/private-key.pem'
kalshi-research capture --max-messages 500
```

Capture an exact open ticker instead of auto-discovery:

```bash
kalshi-research capture --ticker KXBTC15M-... --max-messages 500
```

Public external BTC market-data capture requires no exchange credentials:

```bash
kalshi-research capture-external --venue coinbase --max-messages 500
kalshi-research capture-external --venue kraken --max-messages 500
```

All capture commands are research-only. They contain no order-placement path.

## Data integrity safeguards

- raw frames are written before normalization
- malformed/unknown Kalshi and external frames are hash-chained into quarantine storage
- Kalshi order-book continuity is checked by subscription `sid + seq`
- Kraken book updates are transactional; invalid frames cannot partially mutate the in-memory book
- a malformed Kraken stateful L2 frame stops that capture session instead of building on missing state
- receive-time is the default database/replay ordering and decreasing receive time is rejected
- Coinbase source sequence, Kraken checksum, BRTI upstream timestamp, and Kalshi receive timestamp are retained for later diagnostics
- Kraken checksums are retained for audit but are not falsely labeled verified until the exact CRC canonicalization has its own independent test vectors
- depth-aware taker fill estimation and explicit conservative maker queue assumptions
- expanding walk-forward splits grouped by whole 15-minute markets

See `docs/research_blueprint.md` and `docs/mathematical_spec.md` before adding models.

## Next implementation sequence

1. synchronized multi-source capture session for Kalshi/BRTI/Coinbase/Kraken
2. receive-time feature materializer with freshness masks and stale-source rejection
3. lead/lag experiments: Coinbase/Kraken -> BRTI -> Kalshi probability/book
4. official-settlement reconciler and data-quality audit reports
5. event-driven taker replay across actual visible depth
6. conservative maker queue simulator with latency/cancel stress cases
7. walk-forward experiment runner, model registry, and immutable experiment manifests
8. baseline-vs-feature ablation studies and calibration reports
9. shadow probability outputs only after every promotion gate passes
