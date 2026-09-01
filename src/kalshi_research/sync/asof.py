from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TimedValue(Generic[T]):
    recv_ts_ns: int
    value: T


class ReceiveTimeSeries(Generic[T]):
    """Strict as-of lookup using receive timestamps, never future source time."""

    def __init__(self, values: Iterable[TimedValue[T]] = ()):
        self._values = sorted(values, key=lambda x: x.recv_ts_ns)
        self._times = [v.recv_ts_ns for v in self._values]

    def add(self, value: TimedValue[T]) -> None:
        if self._times and value.recv_ts_ns < self._times[-1]:
            raise ValueError("online receive-time series must be appended monotonically")
        self._values.append(value)
        self._times.append(value.recv_ts_ns)

    def asof(self, decision_ts_ns: int, max_age_ns: int | None = None) -> TimedValue[T] | None:
        idx = bisect_right(self._times, decision_ts_ns) - 1
        if idx < 0:
            return None
        candidate = self._values[idx]
        if max_age_ns is not None and decision_ts_ns - candidate.recv_ts_ns > max_age_ns:
            return None
        return candidate
