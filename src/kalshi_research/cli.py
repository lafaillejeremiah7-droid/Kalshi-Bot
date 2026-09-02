from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from kalshi_research.capture.external_runner import run_external_capture
from kalshi_research.capture.runner import discover_open_btc15m_market, run_kalshi_capture
from kalshi_research.config import ResearchConfig
from kalshi_research.feeds.kalshi_rest import KalshiRestClient
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


def cmd_research_run(args: argparse.Namespace) -> int:
    db_path, archive_path = _research_paths(args)
    if not db_path.exists():
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "mode": "research_only",
                    "order_placement": False,
                    "research_db": str(db_path),
                    "reason": "research_db_not_found",
                },
                indent=2,
                sort_keys=True,
            )
        )
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
