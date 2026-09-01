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
