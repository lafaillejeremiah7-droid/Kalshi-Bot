from __future__ import annotations

import argparse
import asyncio
import json

from kalshi_research.capture.runner import discover_open_btc15m_market, run_kalshi_capture
from kalshi_research.config import ResearchConfig
from kalshi_research.feeds.kalshi_rest import KalshiRestClient


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Kalshi BTC15m research utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="Read current KXBTC15M public metadata only")
    probe.add_argument("--status", choices=["unopened", "open", "closed", "settled"], default="open")
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

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
