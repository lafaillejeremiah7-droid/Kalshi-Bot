from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from kalshi_research.capture.external_runner import run_external_capture
from kalshi_research.capture.runner import discover_open_btc15m_market, run_kalshi_capture
from kalshi_research.capture.supervisor import (
    EvidenceSupervisorError,
    SupervisorPolicy,
    read_supervisor_status,
    run_evidence_supervisor,
)
from kalshi_research.config import ResearchConfig
from kalshi_research.feeds.kalshi_rest import KalshiRestClient
from kalshi_research.research.complete import ResearchCompletionError
from kalshi_research.research.completion_entrypoint import run_research_completion_store
from kalshi_research.research.evidence_status import (
    evidence_readiness_from_events,
    evidence_readiness_store,
)
from kalshi_research.research.registry import ExperimentReportArchive, ReportArchiveError
from kalshi_research.research.runner import (
    ResearchRunError,
    research_report_digest,
    research_report_json,
    run_research_store,
)
from kalshi_research.storage.sqlite_store import SqliteEventStore


def cmd_probe(args: argparse.Namespace) -> int:
    config = ResearchConfig.from_env()
    client = KalshiRestClient(config.kalshi_rest_base)
    series = client.get_series(config.kalshi_series_ticker)
    markets = client.get_markets(series_ticker=config.kalshi_series_ticker, status=args.status)
    result = {
        "series": {
            "ticker": series.get("ticker"),
            "title": series.get("title"),
            "fee_type": series.get("fee_type"),
            "fee_multiplier": series.get("fee_multiplier"),
            "settlement_sources": series.get("settlement_sources"),
        },
        "market_count": len(markets),
        "sample_market_tickers": [m.get("ticker") for m in markets[:5]],
    }
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    config = ResearchConfig.from_env()
    config.ensure_research_dirs()
    ticker = args.ticker or discover_open_btc15m_market(config)
    print(
        json.dumps(
            {
                "mode": "research_capture_only",
                "market_ticker": ticker,
                "raw_capture_dir": str(config.raw_capture_dir),
                "research_db": str(config.research_db_path),
                "order_placement": False,
            },
            indent=2,
        )
    )
    processed = asyncio.run(
        run_kalshi_capture(config, market_ticker=ticker, max_messages=args.max_messages)
    )
    print(json.dumps({"captured_messages": processed, "market_ticker": ticker}, indent=2))
    return 0


def cmd_capture_external(args: argparse.Namespace) -> int:
    config = ResearchConfig.from_env()
    config.ensure_research_dirs()
    print(
        json.dumps(
            {
                "mode": "research_capture_only",
                "venue": args.venue,
                "raw_capture_dir": str(config.raw_capture_dir / "external"),
                "research_db": str(config.research_db_path),
                "authentication_required": False,
                "order_placement": False,
            },
            indent=2,
        )
    )
    processed = asyncio.run(
        run_external_capture(config, venue=args.venue, max_messages=args.max_messages)
    )
    print(json.dumps({"captured_messages": processed, "venue": args.venue}, indent=2))
    return 0


def cmd_capture_all(args: argparse.Namespace) -> int:
    config = ResearchConfig.from_env()
    try:
        policy = SupervisorPolicy(
            market_poll_interval_s=args.market_poll_seconds,
            restart_delay_s=args.restart_delay_seconds,
            evaluation_interval_s=args.evaluation_interval_seconds,
            heartbeat_interval_s=args.heartbeat_seconds,
            evaluate=not args.no_evaluate,
        )
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "mode": "research_capture_only",
                    "order_placement": False,
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    status_path = config.research_db_path.parent / "ops" / "status.json"
    print(
        json.dumps(
            {
                "status": "starting",
                "mode": "research_capture_only",
                "order_placement": False,
                "series_ticker": config.kalshi_series_ticker,
                "research_db": str(config.research_db_path),
                "raw_capture_dir": str(config.raw_capture_dir),
                "report_archive": str(config.report_archive_dir),
                "status_file": str(status_path),
                "feeds": ["kalshi", "brti", "coinbase", "kraken"],
                "periodic_evaluation": policy.evaluate,
                "evaluation_interval_seconds": policy.evaluation_interval_s,
            },
            indent=2,
            sort_keys=True,
        )
    )

    try:
        asyncio.run(run_evidence_supervisor(config, policy=policy))
    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "status": "stopped_by_user",
                    "mode": "research_capture_only",
                    "order_placement": False,
                    "status_file": str(status_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except EvidenceSupervisorError as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "mode": "research_capture_only",
                    "order_placement": False,
                    "research_db": str(config.research_db_path),
                    "status_file": str(status_path),
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    return 0


def _research_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    config = ResearchConfig.from_env()
    if getattr(args, "db", None):
        db_path = Path(args.db).expanduser()
    else:
        db_path = config.research_db_path
    if getattr(args, "archive", None):
        archive_path = Path(args.archive).expanduser()
    else:
        archive_path = config.report_archive_dir
    return db_path, archive_path


def _missing_db_payload(db_path: Path) -> dict[str, object]:
    return {
        "status": "blocked",
        "mode": "research_only",
        "order_placement": False,
        "research_db": str(db_path),
        "reason": "research_db_not_found",
    }


def cmd_evidence_status(args: argparse.Namespace) -> int:
    db_path, _ = _research_paths(args)
    if not db_path.exists():
        readiness = evidence_readiness_from_events(())
        payload = readiness.to_dict()
        payload["research_db"] = str(db_path)
        payload["reason"] = "research_db_not_found"
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0

    try:
        with SqliteEventStore(db_path) as store:
            readiness = evidence_readiness_store(store)
    except ResearchCompletionError as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "mode": "research_only",
                    "order_placement": False,
                    "research_db": str(db_path),
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    payload = readiness.to_dict()
    payload["research_db"] = str(db_path)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


def cmd_ops_status(args: argparse.Namespace) -> int:
    db_path, _ = _research_paths(args)
    status_path = db_path.parent / "ops" / "status.json"
    if not status_path.exists():
        print(
            json.dumps(
                {
                    "status": "not_running",
                    "mode": "research_capture_only",
                    "order_placement": False,
                    "status_file": str(status_path),
                    "reason": "status_file_not_found",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    try:
        payload = read_supervisor_status(status_path)
    except (EvidenceSupervisorError, OSError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "mode": "research_capture_only",
                    "order_placement": False,
                    "status_file": str(status_path),
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    updated_ts_ns = payload.get("updated_ts_ns")
    heartbeat_age_seconds: float | None = None
    heartbeat_stale = True
    if isinstance(updated_ts_ns, int):
        heartbeat_age_seconds = max(0.0, (time.time_ns() - updated_ts_ns) / 1_000_000_000)
        heartbeat_stale = heartbeat_age_seconds > 30.0

    components = payload.get("components")
    component_states: list[object] = []
    if isinstance(components, dict):
        component_states = [
            state.get("status")
            for state in components.values()
            if isinstance(state, dict)
        ]

    if any(state == "failed" for state in component_states):
        ops_status = "failed"
    elif component_states and all(state == "stopped" for state in component_states):
        ops_status = "stopped"
    elif heartbeat_stale:
        ops_status = "stale"
    else:
        ops_status = "running"

    result = dict(payload)
    result["status"] = ops_status
    result["heartbeat_age_seconds"] = heartbeat_age_seconds
    result["heartbeat_stale"] = heartbeat_stale
    result["status_file"] = str(status_path)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def cmd_research_run(args: argparse.Namespace) -> int:
    db_path, archive_path = _research_paths(args)
    if not db_path.exists():
        print(json.dumps(_missing_db_payload(db_path), indent=2, sort_keys=True))
        return 2

    try:
        with SqliteEventStore(db_path) as store:
            report = run_research_store(store)
        archived = ExperimentReportArchive(archive_path).publish(report)
    except (ResearchRunError, ReportArchiveError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "mode": "research_only",
                    "order_placement": False,
                    "research_db": str(db_path),
                    "report_archive": str(archive_path),
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    payload = {
        "status": "passed",
        "report_digest": research_report_digest(report),
        "archived_report": str(archived.path),
        "report": json.loads(research_report_json(report)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


def cmd_research_complete(args: argparse.Namespace) -> int:
    db_path, archive_path = _research_paths(args)
    if not db_path.exists():
        print(json.dumps(_missing_db_payload(db_path), indent=2, sort_keys=True))
        return 2

    try:
        with SqliteEventStore(db_path) as store:
            report = run_research_completion_store(store)
        archived = ExperimentReportArchive(archive_path).publish(report)
    except (ResearchCompletionError, ReportArchiveError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "mode": "research_only",
                    "order_placement": False,
                    "research_db": str(db_path),
                    "report_archive": str(archive_path),
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    payload = {
        "status": "evaluated",
        "verdict": report.verdict,
        "report_digest": research_report_digest(report),
        "archived_report": str(archived.path),
        "order_placement": False,
        "report": json.loads(research_report_json(report)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


def cmd_research_list(args: argparse.Namespace) -> int:
    _, archive_path = _research_paths(args)
    try:
        entries = ExperimentReportArchive(archive_path).list()
    except ReportArchiveError as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "mode": "research_only",
                    "order_placement": False,
                    "report_archive": str(archive_path),
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    payload = {
        "status": "passed",
        "mode": "research_only",
        "order_placement": False,
        "report_archive": str(archive_path),
        "count": len(entries),
        "reports": [
            {
                "digest": entry.digest,
                "series_ticker": entry.series_ticker,
                "plan_digest": entry.plan_digest,
                "events_digest": entry.events_digest,
                "market_count": entry.market_count,
                "event_count": entry.event_count,
                "path": str(entry.path),
            }
            for entry in entries
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


def cmd_research_show(args: argparse.Namespace) -> int:
    _, archive_path = _research_paths(args)
    try:
        archive = ExperimentReportArchive(archive_path)
        entry = archive.get(args.digest)
        report = archive.read_payload(args.digest)
    except ReportArchiveError as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "mode": "research_only",
                    "order_placement": False,
                    "report_archive": str(archive_path),
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    print(
        json.dumps(
            {
                "status": "passed",
                "report_digest": entry.digest,
                "archived_report": str(entry.path),
                "report": report,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Kalshi BTC15m research utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="Read current KXBTC15M public metadata only")
    probe.add_argument(
        "--status",
        choices=["unopened", "open", "closed", "settled"],
        default="open",
    )
    probe.set_defaults(func=cmd_probe)

    capture = sub.add_parser(
        "capture",
        help="Capture authenticated KXBTC15M market data; never places orders",
    )
    capture.add_argument(
        "--ticker",
        help="Exact KXBTC15M market ticker; omit to discover the nearest open market",
    )
    capture.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Optional message limit for controlled research samples",
    )
    capture.set_defaults(func=cmd_capture)

    external = sub.add_parser(
        "capture-external",
        help="Capture public BTC market data from Coinbase or Kraken; never places orders",
    )
    external.add_argument("--venue", choices=["coinbase", "kraken"], required=True)
    external.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Optional message limit for controlled research samples",
    )
    external.set_defaults(func=cmd_capture_external)

    capture_all = sub.add_parser(
        "capture-all",
        help=(
            "Continuously capture Kalshi/BRTI/Coinbase/Kraken, roll 15-minute markets, "
            "and archive periodic research verdicts; never places orders"
        ),
    )
    capture_all.add_argument(
        "--market-poll-seconds",
        type=float,
        default=5.0,
        help="Operational interval for detecting KXBTC15M contract rollover",
    )
    capture_all.add_argument(
        "--restart-delay-seconds",
        type=float,
        default=2.0,
        help="Operational delay before restarting a disconnected feed",
    )
    capture_all.add_argument(
        "--evaluation-interval-seconds",
        type=float,
        default=900.0,
        help="How often to run and immutably archive research-complete",
    )
    capture_all.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=5.0,
        help="Operational status heartbeat interval",
    )
    capture_all.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Capture continuously without periodic evaluation; does not alter model settings",
    )
    capture_all.set_defaults(func=cmd_capture_all)

    evidence_status = sub.add_parser(
        "evidence-status",
        help="Show progress toward first OOS evaluation and the 500-decision promotion gate",
    )
    evidence_status.add_argument(
        "--db",
        help="Optional research SQLite path; defaults to the configured research DB",
    )
    evidence_status.set_defaults(func=cmd_evidence_status)

    ops_status = sub.add_parser(
        "ops-status",
        help="Read the unattended capture heartbeat and component health",
    )
    ops_status.add_argument(
        "--db",
        help="Optional research SQLite path used to locate the adjacent ops status file",
    )
    ops_status.set_defaults(func=cmd_ops_status)

    research_run = sub.add_parser(
        "research-run",
        help=(
            "Run and immutably archive the predeclared fail-closed research suite; "
            "never places orders"
        ),
    )
    research_run.add_argument(
        "--db",
        help="Optional research SQLite path; defaults to the configured research DB",
    )
    research_run.add_argument(
        "--archive",
        help="Optional immutable report archive path; defaults to configured data/experiments",
    )
    research_run.set_defaults(func=cmd_research_run)

    research_complete = sub.add_parser(
        "research-complete",
        help=(
            "Run the full immutable OOS model, execution-economics, stress, and promotion "
            "verdict; never places orders"
        ),
    )
    research_complete.add_argument(
        "--db",
        help="Optional research SQLite path; defaults to the configured research DB",
    )
    research_complete.add_argument(
        "--archive",
        help="Optional immutable report archive path; defaults to configured data/experiments",
    )
    research_complete.set_defaults(func=cmd_research_complete)

    research_list = sub.add_parser(
        "research-list",
        help="Verify and list immutable archived research reports",
    )
    research_list.add_argument(
        "--archive",
        help="Optional immutable report archive path",
    )
    research_list.set_defaults(func=cmd_research_list)

    research_show = sub.add_parser(
        "research-show",
        help="Verify and display one immutable archived research report",
    )
    research_show.add_argument("--digest", required=True, help="SHA-256 research report digest")
    research_show.add_argument(
        "--archive",
        help="Optional immutable report archive path",
    )
    research_show.set_defaults(func=cmd_research_show)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
