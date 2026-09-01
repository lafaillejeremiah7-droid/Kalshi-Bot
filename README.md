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
- receive-time synchronizer with freshness masks and stale-source rejection
- deterministic model-feature rows with feature-set SHA-256 digests
- structural capture audit, settlement reconciliation, and feature-coverage gates
- exact 60-sample settlement utilities
- closed-form partial-final-minute Brownian settlement baseline
- normalized distance/volatility/order-book/microprice/lead-lag math
- whole-market expanding walk-forward probability experiments
- receive-time lead/lag tests with Bonferroni correction and moving-block bootstrap intervals
- probability calibration metrics and cost-adjusted edge calculations
- deterministic fail-closed research runner with input, plan, feature, and report digests
- conservative research promotion gates
- unit tests and CI

## Research doctrine

1. Store raw data before modeling.
2. Separate source timestamp from receive timestamp.
3. No lookahead. Features may only use data received by decision time.
4. Settlement labels are evaluation-only and never enter feature replay.
5. Exact settlement beats generic candle-close assumptions.
6. Full L2 capture is required for credible microstructure replay.
7. Prediction quality and execution quality are measured separately.
8. Fees, slippage, latency, adverse selection and incentives are separate P&L lines.
9. Experiment parameters are predeclared and versioned rather than tuned after seeing results.
10. If robust out-of-sample edge is not positive after costs, reject the idea.

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

## Deterministic research runner

Run the complete fail-closed research suite against the configured canonical SQLite store:

```bash
kalshi-research research-run
```

Or point it at an explicit research database:

```bash
kalshi-research research-run --db /absolute/path/to/research.sqlite3
```

`research-run` does not place orders and does not expose interactive knobs for decision horizons, lead/lag scans, bootstrap settings, or walk-forward sizes. Those values live in the versioned `ExperimentPlan`. If the research plan changes, the code and plan digest must change with it; parameters should not be tuned repeatedly after inspecting outcomes.

The runner executes in this order:

1. structural event audit
2. settled `KXBTC15M` contract/label derivation
3. per-market receive-time feature replay bounded to the market window
4. explicit settlement-label exclusion from feature replay
5. feature coverage/freshness/non-finite-value gates
6. whole-market walk-forward probability benchmarking against the same-horizon Kalshi implied probability
7. receive-time Coinbase/Kraken -> BRTI lead/lag analysis using the predeclared corrected significance tests
8. deterministic JSON report generation

The run blocks instead of producing research conclusions when critical sequence/clock integrity fails, settlement metadata conflicts, coverage is inadequate, replay violates synchronization rules, or the predeclared probability experiment cannot form a valid walk-forward evaluation.

A successful report contains:

- `events_digest`: hash of the exact ordered canonical event input
- `plan_digest`: hash of the predeclared experiment plan
- one `feature_digest` per market
- deterministic probability and lead/lag results
- `report_digest`: hash of the complete machine-readable report
- `order_placement: false`

A missing `--db` path returns a blocked result and does **not** create an empty database.

## Data integrity safeguards

- raw frames are written before normalization
- malformed/unknown Kalshi and external frames are hash-chained into quarantine storage
- Kalshi order-book continuity is checked by subscription `sid + seq`
- Kraken book updates are transactional; invalid frames cannot partially mutate the in-memory book
- a malformed Kraken stateful L2 frame stops that capture session instead of building on missing state
- receive-time is the default database/replay ordering and decreasing receive time is rejected
- settlement events are excluded from model feature replay even if their receive timestamp equals market close
- pre-open global feed history is excluded from a contract's feature-materialization window
- Coinbase source sequence, Kraken checksum, BRTI upstream timestamp, and Kalshi receive timestamp are retained for later diagnostics
- Kraken checksums are retained for audit but are not falsely labeled verified until the exact CRC canonicalization has its own independent test vectors
- depth-aware taker fill estimation and explicit conservative maker queue assumptions
- expanding walk-forward splits grouped by whole 15-minute markets
- lead/lag inference requires the predeclared minimum pair count, multiple-testing correction, and a block-bootstrap interval excluding zero

See `docs/research_blueprint.md` and `docs/mathematical_spec.md` before adding models.

## Next implementation sequence

1. collect enough real synchronized `KXBTC15M` sessions to satisfy the default research-run coverage and walk-forward gates
2. run the immutable baseline experiment plan and archive report digests
3. perform baseline-vs-feature ablation studies without changing the held-out test markets
4. add event-driven taker replay across actual visible depth
5. add a conservative maker queue simulator with latency/cancel stress cases
6. build an immutable experiment/model registry after repeatable evidence exists
7. generate shadow probability outputs only after every promotion gate passes
8. do not add live strategy/execution authority until probability evidence and realistic replay economics both pass independently
