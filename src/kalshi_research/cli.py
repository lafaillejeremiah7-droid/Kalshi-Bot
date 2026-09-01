from __future__ import annotations

import argparse
import json

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Kalshi BTC15m research utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe", help="Read current KXBTC15M public metadata only")
    probe.add_argument("--status", choices=["unopened", "open", "closed", "settled"], default="open")
    probe.set_defaults(func=cmd_probe)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
