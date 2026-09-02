from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kalshi_research.capture.supervisor import (
    EvidenceSupervisor,
    EvidenceSupervisorError,
    EvaluationResult,
    SupervisorPolicy,
    read_supervisor_status,
)
from kalshi_research.config import ResearchConfig


def _config(tmp_path: Path) -> ResearchConfig:
    key = tmp_path / "kalshi.pem"
    key.write_text("fake-private-key", encoding="utf-8")
    return ResearchConfig(
        kalshi_api_key_id="SECRET-KEY-ID",
        kalshi_private_key_path=key,
        research_db_path=tmp_path / "data" / "research.sqlite3",
        raw_capture_dir=tmp_path / "data" / "raw",
        report_archive_dir=tmp_path / "data" / "experiments",
    )


def _policy(*, evaluate: bool = False) -> SupervisorPolicy:
    return SupervisorPolicy(
        market_poll_interval_s=0.01,
        restart_delay_s=0.0,
        evaluation_interval_s=1.0,
        heartbeat_interval_s=0.01,
        evaluate=evaluate,
    )


def test_supervisor_rolls_to_new_market_and_cancels_old_capture(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = _config(tmp_path)
        discovery_calls = 0
        started: list[str] = []
        cancelled: list[str] = []
        second_started = asyncio.Event()

        def discover(_config: ResearchConfig) -> str:
            nonlocal discovery_calls
            discovery_calls += 1
            return "KXBTC15M-A" if discovery_calls < 3 else "KXBTC15M-B"

        async def kalshi_capture(
            _config: ResearchConfig,
            *,
            market_ticker: str,
            max_messages: int | None = None,
        ) -> int:
            assert max_messages is None
            started.append(market_ticker)
            if market_ticker == "KXBTC15M-B":
                second_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(market_ticker)
                raise
            return 0

        async def external_capture(
            _config: ResearchConfig,
            *,
            venue: str,
            max_messages: int | None = None,
        ) -> int:
            assert venue in {"coinbase", "kraken"}
            assert max_messages is None
            await asyncio.Event().wait()
            return 0

        supervisor = EvidenceSupervisor(
            config,
            policy=_policy(),
            discover_market=discover,
            kalshi_capture=kalshi_capture,
            external_capture=external_capture,
        )
        stop = asyncio.Event()
        task = asyncio.create_task(supervisor.run(stop_event=stop))
        await asyncio.wait_for(second_started.wait(), timeout=1.0)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)

        assert started[:2] == ["KXBTC15M-A", "KXBTC15M-B"]
        assert "KXBTC15M-A" in cancelled
        assert supervisor.components["kalshi"].rollovers >= 1
        status = read_supervisor_status(supervisor.status_path)
        assert status["order_placement"] is False
        assert status["mode"] == "research_capture_only"

    asyncio.run(scenario())


def test_supervisor_restarts_external_feed_and_redacts_credentials(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = _config(tmp_path)
        coinbase_attempts = 0
        coinbase_restarted = asyncio.Event()

        def discover(_config: ResearchConfig) -> str:
            return "KXBTC15M-A"

        async def kalshi_capture(
            _config: ResearchConfig,
            *,
            market_ticker: str,
            max_messages: int | None = None,
        ) -> int:
            assert market_ticker == "KXBTC15M-A"
            assert max_messages is None
            await asyncio.Event().wait()
            return 0

        async def external_capture(
            _config: ResearchConfig,
            *,
            venue: str,
            max_messages: int | None = None,
        ) -> int:
            nonlocal coinbase_attempts
            assert max_messages is None
            if venue == "coinbase":
                coinbase_attempts += 1
                if coinbase_attempts == 1:
                    raise RuntimeError(
                        f"bad auth SECRET-KEY-ID {config.kalshi_private_key_path}"
                    )
                coinbase_restarted.set()
            await asyncio.Event().wait()
            return 0

        supervisor = EvidenceSupervisor(
            config,
            policy=_policy(),
            discover_market=discover,
            kalshi_capture=kalshi_capture,
            external_capture=external_capture,
        )
        stop = asyncio.Event()
        task = asyncio.create_task(supervisor.run(stop_event=stop))
        await asyncio.wait_for(coinbase_restarted.wait(), timeout=1.0)
        await supervisor.write_status()
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)

        assert coinbase_attempts >= 2
        assert supervisor.components["coinbase"].restarts >= 1
        raw_status = supervisor.status_path.read_text(encoding="utf-8")
        assert "SECRET-KEY-ID" not in raw_status
        assert str(config.kalshi_private_key_path) not in raw_status
        status = read_supervisor_status(supervisor.status_path)
        assert status["order_placement"] is False

    asyncio.run(scenario())


def test_evaluator_failure_stops_supervisor_fail_closed(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = _config(tmp_path)

        def discover(_config: ResearchConfig) -> str:
            return "KXBTC15M-A"

        async def kalshi_capture(
            _config: ResearchConfig,
            *,
            market_ticker: str,
            max_messages: int | None = None,
        ) -> int:
            await asyncio.Event().wait()
            return 0

        async def external_capture(
            _config: ResearchConfig,
            *,
            venue: str,
            max_messages: int | None = None,
        ) -> int:
            await asyncio.Event().wait()
            return 0

        def evaluate_once(_config: ResearchConfig) -> EvaluationResult:
            raise RuntimeError("structural audit broken")

        supervisor = EvidenceSupervisor(
            config,
            policy=_policy(evaluate=True),
            discover_market=discover,
            kalshi_capture=kalshi_capture,
            external_capture=external_capture,
            evaluate_once=evaluate_once,
        )

        with pytest.raises(EvidenceSupervisorError, match="evaluation failed"):
            await asyncio.wait_for(supervisor.run(), timeout=1.0)

        assert supervisor.components["evaluator"].status == "failed"
        status = read_supervisor_status(supervisor.status_path)
        assert status["order_placement"] is False
        assert status["components"]["evaluator"]["status"] == "failed"

    asyncio.run(scenario())


def test_preflight_refuses_missing_credentials(tmp_path: Path) -> None:
    config = ResearchConfig(
        research_db_path=tmp_path / "data" / "research.sqlite3",
        raw_capture_dir=tmp_path / "data" / "raw",
        report_archive_dir=tmp_path / "data" / "experiments",
    )
    supervisor = EvidenceSupervisor(config, policy=_policy())

    with pytest.raises(EvidenceSupervisorError, match="KALSHI_API_KEY_ID"):
        asyncio.run(supervisor.run())
