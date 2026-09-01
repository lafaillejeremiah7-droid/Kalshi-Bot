from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


EPS = 1e-12


def log_returns(prices: Sequence[float]) -> list[float]:
    if len(prices) < 2:
        return []
    if any(p <= 0 for p in prices):
        raise ValueError("prices must be positive")
    return [math.log(b / a) for a, b in zip(prices, prices[1:])]


def realized_volatility_per_sqrt_second(prices: Sequence[float], sample_interval_s: float = 1.0) -> float:
    """Root-mean-square log-return volatility normalized to sqrt(second)."""
    if sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be positive")
    rets = log_returns(prices)
    if not rets:
        return 0.0
    variance = math.fsum(r * r for r in rets) / len(rets)
    return math.sqrt(variance / sample_interval_s)


def normalized_log_distance(spot: float, target: float, sigma_per_sqrt_second: float, seconds: float) -> float:
    if spot <= 0 or target <= 0:
        raise ValueError("spot and target must be positive")
    if sigma_per_sqrt_second < 0 or seconds < 0:
        raise ValueError("sigma and seconds cannot be negative")
    denom = sigma_per_sqrt_second * math.sqrt(seconds)
    raw = math.log(spot / target)
    if denom <= EPS:
        return math.copysign(math.inf, raw) if abs(raw) > EPS else 0.0
    return raw / denom


def book_imbalance(bid_size: float, ask_size: float) -> float:
    total = bid_size + ask_size
    if total <= 0:
        return 0.0
    return (bid_size - ask_size) / total


def microprice(best_bid: float, bid_size: float, best_ask: float, ask_size: float) -> float:
    if not (0 <= best_bid <= best_ask <= 1):
        raise ValueError("binary-market prices must satisfy 0 <= bid <= ask <= 1")
    total = bid_size + ask_size
    if total <= 0:
        return (best_bid + best_ask) / 2
    return (best_ask * bid_size + best_bid * ask_size) / total


def signed_trade_imbalance(buy_volume: float, sell_volume: float) -> float:
    total = buy_volume + sell_volume
    if total <= 0:
        return 0.0
    return (buy_volume - sell_volume) / total


def ewma(values: Sequence[float], alpha: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    out = values[0]
    for value in values[1:]:
        out = alpha * value + (1 - alpha) * out
    return out


@dataclass(frozen=True, slots=True)
class LeadLagScore:
    lag_steps: int
    correlation: float


def lead_lag_correlation(x: Sequence[float], y: Sequence[float], max_lag: int) -> list[LeadLagScore]:
    """Pearson correlations where positive lag means x leads y."""
    if len(x) != len(y):
        raise ValueError("series must be equal length")
    if max_lag < 0:
        raise ValueError("max_lag cannot be negative")

    def corr(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) < 3:
            return 0.0
        ma, mb = math.fsum(a) / len(a), math.fsum(b) / len(b)
        da = [v - ma for v in a]
        db = [v - mb for v in b]
        denom = math.sqrt(math.fsum(v * v for v in da) * math.fsum(v * v for v in db))
        return 0.0 if denom <= EPS else math.fsum(i * j for i, j in zip(da, db)) / denom

    scores: list[LeadLagScore] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            a, b = x[:-lag], y[lag:]
        elif lag < 0:
            a, b = x[-lag:], y[:lag]
        else:
            a, b = x, y
        scores.append(LeadLagScore(lag, corr(a, b)))
    return scores
