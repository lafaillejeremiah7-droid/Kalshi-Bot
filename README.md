# Kalshi BTC15m Research System

A research-only foundation for discovering and rejecting/validating possible edge in Kalshi's `KXBTC15M` Bitcoin 15-minute event contracts.

**This code does not place orders.** It intentionally contains no live trading endpoint or execution credential flow.

## What exists in v0.1

- typed canonical research events with source and receive timestamps
- current Kalshi REST metadata/trade/fee-change reader
- Kalshi WebSocket auth helper and deterministic binary order-book reconstruction
- fatal sequence-gap detection
- SQLite/WAL append-only canonical event store with content-hash deduplication
- deterministic replay engine
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

Public metadata probe (no trading):

```bash
kalshi-research probe --status open
```

See `docs/research_blueprint.md` and `docs/mathematical_spec.md` before adding collectors/models.

## Additional safeguards implemented

- immutable raw JSONL capture with a chained SHA-256 audit trail
- receive-time as-of synchronization with freshness limits
- depth-aware taker fill estimation and explicit conservative maker queue assumptions
- expanding walk-forward splits grouped by whole 15-minute markets

## Next implementation sequence

1. authenticated real-time `KXBTC15M` discovery/subscription capture service
2. exact BRTI pass-through collector after pinning the current official endpoint/schema
3. Coinbase + Kraken L2/trade collectors with clock/sequence diagnostics
4. synchronized receive-time feature materializer with source-freshness masks
5. official-settlement reconciler and data-quality audit reports
6. event-driven taker replay across actual visible depth
7. conservative maker queue simulator with latency/cancel stress cases
8. walk-forward experiment runner, model registry, and immutable experiment manifests
9. baseline-vs-feature ablation studies and calibration reports
10. shadow probability outputs only after every promotion gate passes
