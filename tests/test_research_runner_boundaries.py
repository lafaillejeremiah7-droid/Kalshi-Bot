import argparse
import json
from decimal import Decimal

from kalshi_research.cli import cmd_research_run
from kalshi_research.domain.events import MarketEvent, SettlementEvent
from kalshi_research.research.runner import _feature_events_for_market


NS = 1_000_000_000
T0 = 1_900_000_000 * NS
TICKER = "KXBTC15M-BOUNDARY"


def test_feature_replay_explicitly_excludes_settlement_received_at_close():
    open_ts = T0
    close_ts = T0 + 900 * NS
    market = MarketEvent(
        event_ts_ns=open_ts - NS,
        recv_ts_ns=open_ts - NS,
        market_ticker=TICKER,
        event_ticker="KXBTC15M-BOUNDARY",
        series_ticker="KXBTC15M",
        target_price=Decimal("100"),
        open_ts_ns=open_ts,
        close_ts_ns=close_ts,
        status="open",
    )
    settlement = SettlementEvent(
        event_ts_ns=close_ts,
        recv_ts_ns=close_ts,
        market_ticker=TICKER,
        target_price=Decimal("100"),
        final_value=Decimal("101"),
        result="yes",
    )

    safe = tuple(
        _feature_events_for_market(
            (market, settlement),
            market_ticker=TICKER,
            open_ts_ns=open_ts,
            close_ts_ns=close_ts,
        )
    )

    assert safe == (market,)
    assert settlement not in safe


def test_research_run_missing_db_is_blocked_without_creating_file(tmp_path, capsys):
    db_path = tmp_path / "missing.sqlite3"

    exit_code = cmd_research_run(argparse.Namespace(db=str(db_path)))
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["mode"] == "research_only"
    assert payload["order_placement"] is False
    assert payload["reason"] == "research_db_not_found"
    assert payload["research_db"] == str(db_path)
    assert not db_path.exists()
