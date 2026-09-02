from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from kalshi_research.capture.external_runner import Venue, run_external_capture
from kalshi_research.capture.runner import discover_open_btc15m_market, run_kalshi_capture
from kalshi_research.config import ResearchConfig
from kalshi_research.research.completion_entrypoint import run_research_completion_events
from kalshi_research.research.registry import ExperimentReportArchive
from kalshi_research.research.runner import research_report_digest
from kalshi_research.storage.sqlite_store import SqliteEventStore


class EvidenceSupervisorError(RuntimeError):
    """Raised when unattended evidence capture cannot continue safely."""


@dataclass(frozen=True, slots=True)
class SupervisorPolicy:
    market_poll_interval_s: float = 5.0
    restart_delay_s: float = 2.0
    evaluation_interval_s: float = 900.0
    heartbeat_interval_s: float = 5.0
    evaluate: bool = True

    def __post_init__(self) -> None:
        if self.market_poll_interval_s <= 0:
            raise ValueError("market_poll_interval_s must be positive")
        if self.restart_delay_s < 0:
            raise ValueError("restart_delay_s cannot be negative")
        if self.evaluation_interval_s <= 0:
            raise ValueError("evaluation_interval_s must be positive")
        if self.heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be positive")


@dataclass(slots=True)
class ComponentState:
    status: Literal["starting", "running", "backoff", "waiting", "failed", "stopped"] = (
        "starting"
    )
    starts: int = 0
    restarts: int = 0
    rollovers: int = 0
    completed_sessions: int = 0
    completed_messages: int = 0
    last_started_ts_ns: int | None = None
    last_stopped_ts_ns: int | None = None
    last_error: str | None = None
    last_error_ts_ns: int | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    status: Literal["waiting", "evaluated"]
    verdict: Literal["promoted", "rejected", "insufficient_evidence"] | None
    report_digest: str | None
    archived_report: str | None
    settled_market_count: int
    horizon_eligible_market_count: int
    executable_decisions: int
    evidence_deficits: tuple[str, ...]
    reason: str | None = None


DiscoverMarket = Callable[[ResearchConfig], str]
KalshiCapture = Callable[..., Awaitable[int]]
ExternalCapture = Callable[..., Awaitable[int]]
EvaluateOnce = Callable[[ResearchConfig], EvaluationResult]


class EvidenceSupervisor:
    """Run the research feeds continuously without any order-placement authority.

    Kalshi capture is rolled to the newly-open 15-minute contract when discovery
    changes ticker. Public external feeds restart after disconnects. The evaluator
    periodically archives the deterministic completion report. A structural audit
    or archive failure is fatal instead of being silently ignored.
    """

    def __init__(
        self,
        config: ResearchConfig,
        *,
        policy: SupervisorPolicy | None = None,
        discover_market: DiscoverMarket = discover_open_btc15m_market,
        kalshi_capture: KalshiCapture = run_kalshi_capture,
        external_capture: ExternalCapture = run_external_capture,
        evaluate_once: EvaluateOnce | None = None,
    ) -> None:
        self.config = config
        self.policy = policy or SupervisorPolicy()
        self._discover_market = discover_market
        self._kalshi_capture = kalshi_capture
        self._external_capture = external_capture
        self._evaluate_once = evaluate_once or self._default_evaluate_once
        self.started_ts_ns = time.time_ns()
        self.active_market: str | None = None
        self.last_evaluation: EvaluationResult | None = None
        self.components: dict[str, ComponentState] = {
            "kalshi": ComponentState(),
            "coinbase": ComponentState(),
            "kraken": ComponentState(),
            "evaluator": ComponentState(status="waiting" if self.policy.evaluate else "stopped"),
        }

    @property
    def status_path(self) -> Path:
        return self.config.research_db_path.parent / "ops" / "status.json"

    def _preflight(self) -> None:
        if not self.config.kalshi_api_key_id:
            raise EvidenceSupervisorError(
                "KALSHI_API_KEY_ID is required for continuous Kalshi market-data capture"
            )
        key_path = self.config.kalshi_private_key_path
        if key_path is None:
            raise EvidenceSupervisorError(
                "KALSHI_PRIVATE_KEY_PATH is required for continuous Kalshi market-data capture"
            )
        if not key_path.is_file():
            raise EvidenceSupervisorError("configured Kalshi private key file does not exist")

    def _sanitize_error(self, exc: BaseException) -> str:
        text = str(exc)
        secrets = [self.config.kalshi_api_key_id]
        if self.config.kalshi_private_key_path is not None:
            secrets.append(str(self.config.kalshi_private_key_path))
        for secret in secrets:
            if secret:
                text = text.replace(secret, "<redacted>")
        if len(text) > 500:
            text = text[:497] + "..."
        return f"{type(exc).__name__}: {text}"

    def _record_error(self, component: str, exc: BaseException) -> None:
        state = self.components[component]
        state.last_error = self._sanitize_error(exc)
        state.last_error_ts_ns = time.time_ns()

    async def _wait_or_stop(self, stop_event: asyncio.Event, seconds: float) -> bool:
        if stop_event.is_set():
            return True
        if seconds <= 0:
            await asyncio.sleep(0)
            return stop_event.is_set()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=seconds)
        except TimeoutError:
            return False
        return True

    async def _run_external_loop(self, venue: Venue, stop_event: asyncio.Event) -> None:
        state = self.components[venue]
        while not stop_event.is_set():
            state.status = "running"
            state.starts += 1
            state.last_started_ts_ns = time.time_ns()
            try:
                processed = await self._external_capture(self.config, venue=venue)
                state.completed_sessions += 1
                state.completed_messages += processed
                if stop_event.is_set():
                    break
                state.restarts += 1
                state.status = "backoff"
                state.last_error = "capture_session_ended"
                state.last_error_ts_ns = time.time_ns()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state.restarts += 1
                state.status = "backoff"
                self._record_error(venue, exc)
            if await self._wait_or_stop(stop_event, self.policy.restart_delay_s):
                break
        state.status = "stopped"
        state.last_stopped_ts_ns = time.time_ns()

    async def _discover(self) -> str:
        return await asyncio.to_thread(self._discover_market, self.config)

    async def _cancel_capture(self, task: asyncio.Task[int] | None) -> None:
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _run_kalshi_loop(self, stop_event: asyncio.Event) -> None:
        state = self.components["kalshi"]
        capture_task: asyncio.Task[int] | None = None
        ticker: str | None = None
        try:
            while not stop_event.is_set():
                if capture_task is None:
                    try:
                        ticker = await self._discover()
                    except Exception as exc:
                        state.status = "backoff"
                        state.restarts += 1
                        self._record_error("kalshi", exc)
                        if await self._wait_or_stop(stop_event, self.policy.restart_delay_s):
                            break
                        continue
                    self.active_market = ticker
                    state.status = "running"
                    state.starts += 1
                    state.last_started_ts_ns = time.time_ns()
                    capture_task = asyncio.create_task(
                        self._kalshi_capture(self.config, market_ticker=ticker)
                    )

                if await self._wait_or_stop(stop_event, self.policy.market_poll_interval_s):
                    break

                if capture_task.done():
                    try:
                        processed = capture_task.result()
                        state.completed_sessions += 1
                        state.completed_messages += processed
                        state.last_error = "capture_session_ended"
                        state.last_error_ts_ns = time.time_ns()
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        self._record_error("kalshi", exc)
                    state.restarts += 1
                    state.status = "backoff"
                    capture_task = None
                    ticker = None
                    self.active_market = None
                    if await self._wait_or_stop(stop_event, self.policy.restart_delay_s):
                        break
                    continue

                try:
                    discovered = await self._discover()
                except Exception as exc:
                    # A transient REST discovery failure must not discard a healthy
                    # websocket session. Keep collecting and retry on the next poll.
                    self._record_error("kalshi", exc)
                    continue

                if discovered != ticker:
                    state.rollovers += 1
                    await self._cancel_capture(capture_task)
                    capture_task = None
                    ticker = None
                    self.active_market = None
                    # Start the newly discovered contract immediately on the next loop.
                    continue
        finally:
            await self._cancel_capture(capture_task)
            self.active_market = None
            state.status = "stopped"
            state.last_stopped_ts_ns = time.time_ns()

    def _default_evaluate_once(self, config: ResearchConfig) -> EvaluationResult:
        if not config.research_db_path.exists():
            return EvaluationResult(
                status="waiting",
                verdict=None,
                report_digest=None,
                archived_report=None,
                settled_market_count=0,
                horizon_eligible_market_count=0,
                executable_decisions=0,
                evidence_deficits=("research_db_not_found",),
                reason="research_db_not_found",
            )

        with SqliteEventStore(config.research_db_path) as store:
            events = tuple(store.iter_events(order_by="receive"))
        if not events:
            return EvaluationResult(
                status="waiting",
                verdict=None,
                report_digest=None,
                archived_report=None,
                settled_market_count=0,
                horizon_eligible_market_count=0,
                executable_decisions=0,
                evidence_deficits=("research_store_empty",),
                reason="research_store_empty",
            )

        # This audited entrypoint raises on structural corruption. Do not swallow it.
        report = run_research_completion_events(events)
        archived = ExperimentReportArchive(config.report_archive_dir).publish(report)
        executable = 0 if report.economics is None else report.economics.executable_decisions
        return EvaluationResult(
            status="evaluated",
            verdict=report.verdict,
            report_digest=research_report_digest(report),
            archived_report=str(archived.path),
            settled_market_count=report.settled_market_count,
            horizon_eligible_market_count=report.horizon_eligible_market_count,
            executable_decisions=executable,
            evidence_deficits=report.evidence_deficits,
        )

    async def _run_evaluator_loop(self, stop_event: asyncio.Event) -> None:
        state = self.components["evaluator"]
        if not self.policy.evaluate:
            state.status = "stopped"
            return

        while not stop_event.is_set():
            state.status = "running"
            state.starts += 1
            state.last_started_ts_ns = time.time_ns()
            try:
                result = await asyncio.to_thread(self._evaluate_once, self.config)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state.status = "failed"
                self._record_error("evaluator", exc)
                await self.write_status()
                raise EvidenceSupervisorError(
                    "periodic research evaluation failed; capture stopped fail-closed"
                ) from exc

            self.last_evaluation = result
            state.completed_sessions += 1
            state.status = "waiting"
            state.last_error = result.reason
            state.last_error_ts_ns = time.time_ns() if result.reason else None
            if await self._wait_or_stop(stop_event, self.policy.evaluation_interval_s):
                break

        state.status = "stopped"
        state.last_stopped_ts_ns = time.time_ns()

    def snapshot(self) -> dict[str, object]:
        return {
            "mode": "research_capture_only",
            "order_placement": False,
            "started_ts_ns": self.started_ts_ns,
            "updated_ts_ns": time.time_ns(),
            "active_market": self.active_market,
            "components": {name: asdict(state) for name, state in sorted(self.components.items())},
            "evaluation": None if self.last_evaluation is None else asdict(self.last_evaluation),
            "paths": {
                "research_db": str(self.config.research_db_path),
                "raw_capture_dir": str(self.config.raw_capture_dir),
                "report_archive": str(self.config.report_archive_dir),
                "status_file": str(self.status_path),
            },
        }

    async def write_status(self) -> None:
        path = self.status_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        payload = json.dumps(
            self.snapshot(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        temporary.write_text(payload + "\n", encoding="utf-8")
        os.replace(temporary, path)

    async def _run_heartbeat_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            await self.write_status()
            if await self._wait_or_stop(stop_event, self.policy.heartbeat_interval_s):
                break
        await self.write_status()

    async def run(self, *, stop_event: asyncio.Event | None = None) -> None:
        self._preflight()
        self.config.ensure_research_dirs()
        stop = stop_event or asyncio.Event()
        component_tasks = [
            asyncio.create_task(self._run_kalshi_loop(stop), name="kalshi-capture"),
            asyncio.create_task(self._run_external_loop("coinbase", stop), name="coinbase-capture"),
            asyncio.create_task(self._run_external_loop("kraken", stop), name="kraken-capture"),
            asyncio.create_task(self._run_evaluator_loop(stop), name="research-evaluator"),
            asyncio.create_task(self._run_heartbeat_loop(stop), name="ops-heartbeat"),
        ]
        stop_waiter = asyncio.create_task(stop.wait(), name="stop-waiter")
        try:
            done, _ = await asyncio.wait(
                [stop_waiter, *component_tasks],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_waiter not in done:
                finished = next(task for task in done if task is not stop_waiter)
                exc = finished.exception()
                if exc is not None:
                    raise exc
                raise EvidenceSupervisorError(
                    f"supervisor component exited unexpectedly: {finished.get_name()}"
                )
        finally:
            stop.set()
            stop_waiter.cancel()
            for task in component_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(stop_waiter, *component_tasks, return_exceptions=True)
            for state in self.components.values():
                if state.status != "failed":
                    state.status = "stopped"
                    state.last_stopped_ts_ns = time.time_ns()
            await self.write_status()


def read_supervisor_status(path: str | Path) -> dict[str, object]:
    status_path = Path(path)
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceSupervisorError("supervisor status payload must be a JSON object")
    if payload.get("order_placement") is not False:
        raise EvidenceSupervisorError("invalid supervisor status: research-only invariant missing")
    return payload


async def run_evidence_supervisor(
    config: ResearchConfig,
    *,
    policy: SupervisorPolicy | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    supervisor = EvidenceSupervisor(config, policy=policy)
    await supervisor.run(stop_event=stop_event)
