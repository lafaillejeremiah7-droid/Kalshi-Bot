# Mathematical research specification

## 1. Contract target

For each `KXBTC15M` market define target `K`, current BRTI/price proxy `S_t`, and seconds remaining `tau`.
Never infer `K` from the ticker alone when authoritative market metadata/rules are available.

## 2. First-principles baseline

Use a deliberately simple zero-drift log-diffusion benchmark:

`P_yes = Phi( ln(S_t / K) / (sigma_s * sqrt(tau)) )`

where `sigma_s` is short-horizon log-return volatility per sqrt(second). This is a benchmark, not a claim that BTC is lognormal over every 15-minute window.
Any machine-learned model must beat this baseline out of sample on calibration *and* net executable economics.

## 3. Final-minute settlement state

Kalshi documents BTC crypto settlement as the rounded simple average of 60 one-second CF Benchmarks RTI observations in the final minute. Let `x_1..x_k` be already-known settlement samples and `n = 60-k`.

Required mean of remaining observations:

`R_k = (60*K - sum(x_1..x_k)) / n`

As `k` rises, the unresolved uncertainty shrinks mechanically. A model that ignores the known partial sum is misspecified.

### Brownian baseline for the remaining sum

If future one-second levels follow arithmetic Brownian increments from current index `S_0`,

`X_j = S_0 + mu*j + sigma*W_j`, `j=1..n`, and `Cov(W_i,W_j)=min(i,j)`.

Then

`E[sum X_j] = n*S_0 + mu*n(n+1)/2`

and

`Var[sum X_j] = sigma^2 * n(n+1)(2n+1)/6`.

This gives a closed-form baseline for the probability the final 60-sample average clears `K`. Empirical residual models can then test whether order flow, venue lead/lag, or microstructure add incremental information.

## 4. Features to research

- normalized log distance to target
- realized volatility at 5s/15s/30s/60s/180s/remaining-window horizons
- return momentum and acceleration with strictly backward-looking windows
- Kalshi best bid/derived ask, spread, depth, depth slope and imbalance
- microprice and aggressive trade imbalance
- cancellation/addition intensity and book velocity
- BRTI-vs-spot basis and BRTI-vs-futures basis
- lead/lag cross-correlations and predictive regressions with explicit lags
- time-to-expiry regime and final-minute sample count
- executable pair cost: `yes_ask(q)+no_ask(q)+all_costs(q)`
- adverse selection: markout after maker fills at 100ms/500ms/1s/5s
- latency distributions: source timestamp -> receive timestamp -> decision timestamp

## 5. Probability metrics

Primary probability metrics:
- Brier score
- log loss
- reliability/calibration curves
- expected calibration error (ECE)
- calibration intercept/slope

Never select models only by trading P&L. A model can produce a lucky P&L sample while being badly calibrated.

## 6. Trading-economics research metric

For a YES purchase at executable price `p`:

`net_edge = P_yes - p - fees - slippage - latency_penalty`

All terms must be measured in dollars per $1 payout contract. Research should only label a decision executable if displayed depth can support the tested size under the fill model.

## 7. Leakage controls

At decision timestamp `t`, a feature may use only data whose *receive timestamp* is <= `t`. Source timestamps alone are insufficient because they can create impossible knowledge. Settlement samples after `t` are forbidden. Final market outcome, final BRTI average, future trades, future book changes and post-decision revised metadata are labels only.
