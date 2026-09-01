# Canonical data dictionary

| Field | Meaning | Leakage rule |
|---|---|---|
| `event_ts_ns` | source/exchange event time | informational; not enough alone to prove availability |
| `recv_ts_ns` | local receive time | primary knowability boundary for features |
| `market_ticker` | exact Kalshi contract | immutable key |
| `target_price` | authoritative target/strike | must be captured before decision |
| `close_ts_ns` | scheduled market close | known metadata |
| `brti` | CF Benchmarks Bitcoin RTI | must be actual value received by then |
| `spot_bid/ask` | venue top of book | venue + freshness required |
| `kalshi_yes_bid` | best YES bid | reconstructed from valid book |
| `kalshi_yes_ask` | `1 - best_no_bid` in complementary binary book | only while book state is valid |
| `book_seq` | WS sequence | gaps invalidate book |
| `seconds_remaining` | close time minus decision time | clipped at 0 |
| `rv_*` | backward-looking realized volatility | no centered/future window |
| `settlement_samples_known` | count of final-minute BRTI samples available | future samples forbidden |
| `settlement_known_sum` | sum of known final-minute samples | future samples forbidden |
| `final_outcome` | yes/no label | never in features |
| `final_value` | official settlement average | label/audit only |
| `fee_type/multiplier` | fee rule effective at event time | persist version/effective timestamp |

## Receive-time research contract

`recv_ts_ns` is the information frontier. A model row at decision time `D` may only use source state whose receive timestamp is `<= D`. Exchange/source timestamps may be retained for diagnostics and lead/lag studies, but they never override the receive-time boundary.

Replay is performed in nondecreasing receive-time order. Once later state has been ingested, code must not materialize an earlier decision frame from that state; rebuild from receive-time replay instead. Any backwards receive-time transition fails closed.

BRTI, Coinbase, and Kraken events are global inputs and normally have no Kalshi market ticker. Therefore the canonical event store must **not** be ticker-filtered before synchronized replay. Ticker filtering is applied only after each event has been admitted to the global state machine, otherwise external information would silently disappear from the dataset.

Every synchronized row carries source age/freshness for Kalshi book, BRTI, Coinbase, and Kraken. A stale source is masked out rather than forward-filled as if current. Current research defaults are 750 ms for Kalshi book, 2.5 s for BRTI, and 1.5 s for each external venue; these are research parameters that must be stress-tested rather than treated as permanent constants.

The current probability-readiness gate requires a fresh valid Kalshi book, fresh BRTI, an authoritative target, and at least one fresh external BTC venue. Missing/stale data should reduce the usable sample count; it must not be imputed from the future.

## Model-ready derived fields

| Field | Definition / rule |
|---|---|
| `kalshi_yes_mid` | `(best_yes_bid + implied_yes_ask) / 2` when both sides are fresh |
| `kalshi_book_imbalance` | `(yes_bid_size - implied_yes_ask_size) / (yes_bid_size + implied_yes_ask_size)` |
| `external_consensus_mid` | arithmetic mean of currently fresh Coinbase/Kraken mids; stale venues excluded |
| `brti_vs_external_bps` | `10000 * ln(BRTI / external_consensus_mid)` |
| `brti_log_distance_to_target` | `ln(BRTI / target)` |
| `brti_vol_per_sqrt_second` | square root of `sum(log_return^2) / sum(elapsed_seconds)` over backward BRTI receive-time observations |
| `normalized_distance_to_target` | `ln(BRTI/target) / (sigma * sqrt(seconds_to_close))`; absent when `T <= 0` or volatility is unavailable |
| `final_minute_progress` | `k / 60`, where `k` is the received BRTI settlement-window sample count |
| `required_remaining_brti_average` | `(60*K - k*A_k)/(60-k)`, target `K`, rolling received final-minute average `A_k`; only for `0 <= k < 60` |

The final-minute formula follows from the known sum `k*A_k`. It is settlement-state information available at that point, not a future label.

## Deterministic dataset contract

Feature datasets are regenerated from the canonical event store in receive-time order. Given the same stored events, code version, freshness policy, and feature configuration, serialization must be byte-deterministic. Research JSONL uses sorted keys and rejects NaN/Infinity. A SHA-256 digest is recorded for reproducibility and experiment/model-registry linkage.

Labels such as final settlement value/outcome are attached only after feature generation and must never enter the synchronized feature state for their own or earlier decision timestamps.
