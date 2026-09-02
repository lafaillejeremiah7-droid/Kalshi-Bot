from __future__ import annotations

import asyncio
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from kalshi_research.capture.market_metadata import (
    MarketMetadataSchemaError,
    capture_market_metadata,
    pending_settlement_tickers,
)
from kalshi_research.capture.supervisor import (
    ComponentState,
    EvidenceSupervisor,
    EvidenceSupervisorError,
    SupervisorPolicy,
)
from kalshi_research.config import ResearchConfig
from kalshi_research.storage.sqlite_store import SqliteEventStore


@dataclass(frozen=True, slots=True)
class SettlementBackfillResult:
    candidates: tuple[str, ...]
    settled_count: int


CaptureMarketMetadata = Callable[[ResearchConfig, str], bool]


def settlement_backfill_once(
    config: ResearchConfig,
    *,
    capture_market: CaptureMarketMetadata = capture_market_metadata,
) -> SettlementBackfillResult:
    """Backfill official labels for captured closed markets after any outage."""
    if not config.research_db_path.exists():
        return SettlementBackfillResult(candidates=(), settled_count=0)

    with SqliteEventStore(config.research_db_path) as store:
        candidates = pending_settlement_tickers(store)

    settled_count = 0
    for ticker in candidates:
        if capture_market(config, ticker):
            settled_count += 1
    return SettlementBackfillResult(candidates=candidates, settled_count=settled_count)


async def _wait_or_stop(stop_event: asyncio.Event, seconds: float) -> bool:
    if stop_event.is_set():
        return True
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except TimeoutError:
        return False
    return True


async def _run_settlement_loop(
    supervisor: EvidenceSupervisor,
    stop_event: asyncio.Event,
    *,
    interval_s: float,
) -> None:
    state = supervisor.components["settlement"]
    while not stop_event.is_set():
        state.status = "running"
        state.starts += 1
        state.last_started_ts_ns = time.time_ns()
        try:
            result = await asyncio.to_thread(settlement_backfill_once, supervisor.config)
        except httpx.HTTPError as exc:
            # Network/server failures are retriable. They are surfaced in the
            # heartbeat, but do not discard otherwise valid live market data.
            state.status = "backoff"
            state.restarts += 1
            supervisor._record_error("settlement", exc)
        except MarketMetadataSchemaError as exc:
            # Schema/semantic drift can corrupt labels, so stop the entire
            # research capture rather than silently skipping a malformed result.
            state.status = "failed"
            supervisor._record_error("settlement", exc)
            raise EvidenceSupervisorError(
                "settlement metadata schema changed; capture stopped fail-closed"
            ) from exc
        except Exception as exc:
            state.status = "failed"
            supervisor._record_error("settlement", exc)
            raise EvidenceSupervisorError(
                "settlement backfill failed unexpectedly; capture stopped fail-closed"
            ) from exc
        else:
            state.completed_messages += len(result.candidates)
            state.completed_sessions += result.settled_count
            state.status = "waiting"
            state.last_error = None
            state.last_error_ts_ns = None

        if await _wait_or_stop(stop_event, interval_s):
            break

    if state.status != "failed":
        state.status = "stopped"
        state.last_stopped_ts_ns = time.time_ns()


def _install_signal_handlers(stop_event: asyncio.Event) -> tuple[signal.Signals, ...]:
    """Translate SIGINT/SIGTERM into cooperative supervisor shutdown where supported."""
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError, ValueError):
            continue
        installed.append(sig)
    return tuple(installed)


def _remove_signal_handlers(signals: tuple[signal.Signals, ...]) -> None:
    loop = asyncio.get_running_loop()
    for sig in signals:
        loop.remove_signal_handler(sig)


async def run_evidence_runtime(
    config: ResearchConfig,
    *,
    policy: SupervisorPolicy | None = None,
    stop_event: asyncio.Event | None = None,
    settlement_poll_interval_s: float = 15.0,
) -> None:
    """Run unattended capture, settlement recovery, evaluation and graceful shutdown."""
    if settlement_poll_interval_s <= 0:
        raise ValueError("settlement_poll_interval_s must be positive")

    supervisor = EvidenceSupervisor(config, policy=policy)
    supervisor.components["settlement"] = ComponentState(status="waiting")
    stop = stop_event or asyncio.Event()
    installed_signals: tuple[signal.Signals, ...] = ()
    if stop_event is None:
        installed_signals = _install_signal_handlers(stop)

    supervisor_task = asyncio.create_task(
        supervisor.run(stop_event=stop),
        name="evidence-supervisor",
    )
    settlement_task = asyncio.create_task(
        _run_settlement_loop(
            supervisor,
            stop,
            interval_s=settlement_poll_interval_s,
        ),
        name="settlement-backfill",
    )
    tasks = (supervisor_task, settlement_task)

    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                raise exc

        if not stop.is_set():
            finished = next(iter(done))
            raise EvidenceSupervisorError(
                f"evidence runtime component exited unexpectedly: {finished.get_name()}"
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                result,
                asyncio.CancelledError,
            ):
                raise result
    finally:
        stop.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if installed_signals:
            _remove_signal_handlers(installed_signals)
