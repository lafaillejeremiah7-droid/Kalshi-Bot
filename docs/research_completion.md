# Research completion protocol

`kalshi-research research-complete` is the final research-only evaluation path for `KXBTC15M`. It combines structural evidence integrity, out-of-sample probability modeling, fee-aware decision selection, realistic execution replay, stress testing, promotion gates, and immutable reporting. It never places an order.

## Command

```bash
kalshi-research research-complete
```

Optional explicit paths:

```bash
kalshi-research research-complete \
  --db /absolute/path/to/research.sqlite3 \
  --archive /absolute/path/to/experiments
```

A structurally valid evaluation returns one of three verdicts:

- `promoted`: all evidence-volume, prediction-quality, calibration, execution-economics, walk-forward, latency-stress, and cost-stress gates pass.
- `rejected`: enough valid evidence exists for a decision, but at least one promotion gate fails.
- `insufficient_evidence`: the software can evaluate the store, but there is not yet enough trustworthy executable OOS evidence, fee/quote coverage, or eligible settled markets to decide.

Structural corruption is different from insufficient evidence. Sequence/clock integrity failures, conflicting settlement metadata, or other critical audit failures block the run instead of producing a verdict.

## Immutable evidence boundary

The public completion entrypoint audits events in the exact order supplied by the canonical store before downstream deterministic normalization. A receive-time regression cannot be hidden by sorting first.

Every completed report contains:

- exact canonical event count and `events_digest`
- predeclared `plan_digest`
- fixed `model_spec_digest`
- OOS `prediction_digest` when the model evaluation can run
- settled and 60-second-horizon eligible market counts
- model ablations and fold definitions
- fee-aware trade/no-trade selections
- prediction and execution metrics
- evidence deficits and promotion reasons
- final verdict
- `order_placement: false`

The entire JSON report is published through the content-addressed SHA-256 experiment archive. Re-publishing identical evidence and configuration is idempotent; altered content receives a different digest, and corrupted archived content fails verification.

## Predeclared OOS protocol

The default completion plan uses exactly one decision horizon: 60 seconds before contract close. One eligible row is selected per settled market from the safe side of that horizon.

Chronological expanding walk-forward folds use whole 15-minute contracts only:

- minimum training markets: 100
- validation markets: 20
- OOS test markets: 20
- step: 20 markets

The step equals the test-block size by invariant, so OOS test markets never overlap between folds. No random adjacent-second split is used.

For every fold:

1. transformations are fit on training markets only;
2. missing numeric features use training-only zero-after-standardization imputation plus explicit missingness indicators;
3. L2 regularization is chosen using validation log loss only;
4. the final fold model is fit on train + validation markets;
5. test outcomes are used only for scoring, never for fitting or regularization selection.

The deterministic model is an L2-regularized logistic model solved by fixed numerical logic. The intercept is not regularized. The complete model specification is itself hashed.

## Nested ablations

The exact nested feature stages are evaluated on the same OOS test markets:

1. `baseline`
   - normalized BRTI distance to target
   - Kalshi YES midpoint
2. `settlement`
   - baseline features
   - final-minute progress
   - required remaining settlement-average gap
3. `external`
   - settlement features
   - BRTI-vs-external-consensus basis
4. `microstructure`
   - external features
   - Kalshi book imbalance
   - Kalshi spread

Each stage reports OOS Brier score, log loss, calibration error, and same-market Kalshi-implied baseline scores. The final `microstructure` stage drives strategy candidate decisions; the ablations show whether added information actually improves the OOS result rather than merely increasing model complexity.

## Fee-aware trade/no-trade selection

The completion system does not trade every prediction.

At each OOS decision it reconstructs executable binary quotes and looks up the fee schedule that was both effective and knowable at that receive time. It estimates entry fees using the same fixed-point Kalshi fee and rounding primitives used by the execution research layer.

For YES:

```text
net_edge_yes = predicted_yes - executable_yes_ask - estimated_fee_per_contract
```

For NO:

```text
net_edge_no = (1 - predicted_yes) - executable_no_ask - estimated_fee_per_contract
```

The stronger side is submitted to research replay only if its estimated net edge clears the predeclared 1% default threshold. Otherwise the correct action is `no trade`.

Missing as-of fee history or executable quote coverage is recorded as an evidence deficit, not silently filled using future information.

## Execution economics

Submitted OOS intents use the existing v0.8 receive-time execution simulator rather than a second optimistic backtester. It includes:

- sequence-valid historical Kalshi book reconstruction
- visible-depth taker price walking
- partial fills
- stale-book rejection
- effective fee history and exchange-style rounding
- exact settlement payoff
- overlapping capital locking / optional bankroll limit
- separate gross and net P&L
- max drawdown

The broader v0.8 research system also contains a conservative maker queue simulator with zero cancellation credit by default. The v1 completion candidate intentionally uses fee-aware taker intents so the final OOS verdict does not depend on unverifiable historical queue priority.

## Stress tests

The same OOS intents are replayed under:

- base decision latency: 100 ms
- latency stress: 600 ms
- transaction-cost stress: 1.5x modeled fee cost

Promotion requires positive net P&L under both stress cases.

## Default promotion gates

After a valid complete evaluation, the candidate does not advance unless all default gates pass:

- at least 500 executable OOS decisions
- Brier score improves by at least 1% versus the same-horizon market baseline
- log loss improves by at least 1% versus the same-horizon market baseline
- expected calibration error <= 3 percentage points
- positive net OOS P&L
- costs consume no more than 75% of gross edge
- at least 60% of walk-forward test windows are profitable after costs
- positive latency-stress P&L
- positive transaction-cost-stress P&L

If fewer than 500 executable OOS decisions exist, the verdict is `insufficient_evidence`, even if every currently observed trade happened to be profitable.

## What “research complete” means

It means the research software is complete enough to collect, audit, model, replay, stress, reject, or promote evidence under a deterministic and immutable protocol.

It does **not** mean a profitable edge has already been proven. The system is designed to return `insufficient_evidence` or `rejected` when that is what the data supports. No strategy, probability, risk, or execution bot receives live order authority from this command.
