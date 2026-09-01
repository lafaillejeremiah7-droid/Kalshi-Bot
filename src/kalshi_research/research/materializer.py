from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from kalshi_research.math.features import book_imbalance, normalized_log_distance
from kalshi_research.research.synchronizer import SynchronizedFrame


@dataclass(frozen=True, slots=True)
class ModelFeatureRow:
    decision_recv_ts_ns: int
    market_ticker: str
    probability_ready: bool
    baseline_ready: bool

    seconds_to_close: float | None
    target_price: float | None
    brti: float | None
    brti_log_distance_to_target: float | None
    brti_vol_per_sqrt_second: float | None
    normalized_distance_to_target: float | None

    kalshi_yes_bid: float | None
    kalshi_yes_ask: float | None
    kalshi_yes_mid: float | None
    kalshi_spread: float | None
    kalshi_book_imbalance: float | None

    external_consensus_mid: float | None
    brti_vs_external_bps: float | None
    coinbase_mid: float | None
    kraken_mid: float | None

    final_minute_sample_count: int | None
    final_minute_progress: float | None
    final_minute_average: float | None
    required_remaining_brti_average: float | None

    kalshi_book_age_ms: float | None
    brti_age_ms: float | None
    coinbase_age_ms: float | None
    kraken_age_ms: float | None


@dataclass(slots=True)
class FeatureMaterializer:
    volatility_window_ns: int = 30_000_000_000
    _last_decision_ns: int | None = None
    _last_brti_recv_ns: int | None = None
    _brti_history: deque[tuple[int, float]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.volatility_window_ns <= 0:
            raise ValueError("volatility_window_ns must be positive")

    def materialize(self, frame: SynchronizedFrame) -> ModelFeatureRow:
        decision = frame.decision_recv_ts_ns
        if self._last_decision_ns is not None and decision < self._last_decision_ns:
            raise ValueError("feature rows must be materialized in nondecreasing receive time")
        self._last_decision_ns = decision

        self._update_brti_history(frame)
        sigma = self._event_timed_brti_volatility()

        target = _fresh_float(frame.target_price, frame.market_state.fresh)
        brti = _fresh_float(frame.brti, frame.brti_state.fresh)
        seconds_to_close = frame.seconds_to_close if frame.market_state.fresh else None

        log_distance = None
        normalized_distance = None
        if brti is not None and target is not None and brti > 0 and target > 0:
            log_distance = math.log(brti / target)
            if sigma is not None and sigma > 0 and seconds_to_close is not None:
                normalized_distance = normalized_log_distance(
                    brti,
                    target,
                    sigma,
                    max(seconds_to_close, 0.0),
                )

        yes_bid = _fresh_float(frame.yes_bid, frame.kalshi_book_state.fresh)
        yes_ask = _fresh_float(frame.yes_ask, frame.kalshi_book_state.fresh)
        yes_spread = _fresh_float(frame.yes_spread, frame.kalshi_book_state.fresh)
        yes_mid = None
        if yes_bid is not None and yes_ask is not None:
            yes_mid = (yes_bid + yes_ask) / 2

        imbalance = None
        if frame.kalshi_book_state.fresh:
            bid_size = _float_or_none(frame.yes_bid_size)
            ask_size = _float_or_none(frame.yes_ask_size)
            if bid_size is not None and ask_size is not None:
                imbalance = book_imbalance(bid_size, ask_size)

        coinbase_mid = (
            _float_or_none(frame.coinbase_mid) if frame.coinbase_state.fresh else None
        )
        kraken_mid = _float_or_none(frame.kraken_mid) if frame.kraken_state.fresh else None
        external_values = [x for x in (coinbase_mid, kraken_mid) if x is not None]
        external_consensus = None
        if external_values:
            external_consensus = math.fsum(external_values) / len(external_values)

        basis_bps = None
        if brti is not None and external_consensus is not None and external_consensus > 0:
            basis_bps = 10_000 * math.log(brti / external_consensus)

        sample_count = (
            frame.brti_final_minute_sample_count if frame.brti_state.fresh else None
        )
        final_average = (
            _float_or_none(frame.brti_final_minute_average)
            if frame.brti_state.fresh
            else None
        )
        final_progress = None
        required_remaining_average = None
        if sample_count is not None:
            final_progress = sample_count / 60
            if (
                target is not None
                and final_average is not None
                and 0 <= sample_count < 60
            ):
                known_sum = final_average * sample_count
                required_remaining_average = (
                    60 * target - known_sum
                ) / (60 - sample_count)

        baseline_ready = (
            frame.probability_ready
            and sigma is not None
            and sigma > 0
            and normalized_distance is not None
            and yes_mid is not None
        )

        return ModelFeatureRow(
            decision_recv_ts_ns=decision,
            market_ticker=frame.market_ticker,
            probability_ready=frame.probability_ready,
            baseline_ready=baseline_ready,
            seconds_to_close=seconds_to_close,
            target_price=target,
            brti=brti,
            brti_log_distance_to_target=log_distance,
            brti_vol_per_sqrt_second=sigma,
            normalized_distance_to_target=normalized_distance,
            kalshi_yes_bid=yes_bid,
            kalshi_yes_ask=yes_ask,
            kalshi_yes_mid=yes_mid,
            kalshi_spread=yes_spread,
            kalshi_book_imbalance=imbalance,
            external_consensus_mid=external_consensus,
            brti_vs_external_bps=basis_bps,
            coinbase_mid=coinbase_mid,
            kraken_mid=kraken_mid,
            final_minute_sample_count=sample_count,
            final_minute_progress=final_progress,
            final_minute_average=final_average,
            required_remaining_brti_average=required_remaining_average,
            kalshi_book_age_ms=_age_ms(frame.kalshi_book_state.age_ns),
            brti_age_ms=_age_ms(frame.brti_state.age_ns),
            coinbase_age_ms=_age_ms(frame.coinbase_state.age_ns),
            kraken_age_ms=_age_ms(frame.kraken_state.age_ns),
        )

    def _update_brti_history(self, frame: SynchronizedFrame) -> None:
        recv_ns = frame.brti_state.recv_ts_ns
        if not frame.brti_state.fresh or recv_ns is None or frame.brti is None:
            self._prune(frame.decision_recv_ts_ns)
            return

        if self._last_brti_recv_ns is None or recv_ns > self._last_brti_recv_ns:
            self._brti_history.append((recv_ns, float(frame.brti)))
            self._last_brti_recv_ns = recv_ns
        elif recv_ns < self._last_brti_recv_ns:
            raise ValueError("BRTI receive time moved backward")

        self._prune(frame.decision_recv_ts_ns)

    def _prune(self, decision_ns: int) -> None:
        cutoff = decision_ns - self.volatility_window_ns
        while len(self._brti_history) > 1 and self._brti_history[0][0] < cutoff:
            self._brti_history.popleft()

    def _event_timed_brti_volatility(self) -> float | None:
        if len(self._brti_history) < 2:
            return None

        sum_sq = 0.0
        total_seconds = 0.0
        points = list(self._brti_history)
        for (t0, p0), (t1, p1) in zip(points, points[1:]):
            dt_s = (t1 - t0) / 1_000_000_000
            if dt_s <= 0 or p0 <= 0 or p1 <= 0:
                continue
            r = math.log(p1 / p0)
            sum_sq += r * r
            total_seconds += dt_s

        if total_seconds <= 0:
            return None
        return math.sqrt(sum_sq / total_seconds)


def _fresh_float(value, fresh: bool) -> float | None:
    if not fresh:
        return None
    return _float_or_none(value)


def _float_or_none(value) -> float | None:
    return None if value is None else float(value)


def _age_ms(age_ns: int | None) -> float | None:
    return None if age_ns is None else age_ns / 1_000_000
