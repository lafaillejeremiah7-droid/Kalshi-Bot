from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from dashboard_server import _read_performance
from xau_company.dashboard import DashboardLogHandler, DashboardPublisher, default_state


def _employee(state, employee_id):
    return next(row for row in state["employees"] if row["id"] == employee_id)


def test_default_dashboard_has_26_unique_employees():
    state = default_state()
    ids = [row["id"] for row in state["employees"]]
    assert len(ids) == 26
    assert len(set(ids)) == 26
    assert "strategy_evolution" not in ids
    assert "strategy_invention" not in ids
    assert {room["id"] for room in state["rooms"]} == {
        "market", "research", "analysis", "risk", "decision", "performance"
    }


def test_publisher_writes_atomic_handoff_state(tmp_path):
    path = tmp_path / "dashboard.json"
    publisher = DashboardPublisher(path)
    publisher.employee("market_data", "working", "Fetching candles")
    publisher.handoff("market_data", "data_quality", "OHLC Packet", "5 timeframes")

    raw = json.loads(path.read_text())
    assert raw["handoffs"][-1]["from"] == "market_data"
    assert raw["handoffs"][-1]["to"] == "data_quality"
    assert _employee(raw, "market_data")["state"] == "handoff"
    assert _employee(raw, "data_quality")["state"] == "receiving"
    assert not path.with_suffix(".json.tmp").exists()


def test_log_handler_turns_real_signal_log_into_visible_office_flow(tmp_path):
    publisher = DashboardPublisher(tmp_path / "state.json")
    handler = DashboardLogHandler(publisher)
    record = logging.LogRecord(
        "xau-company",
        logging.INFO,
        __file__,
        1,
        "Signal BUY using Trend Pullback #184 raw_confidence 78.0% calibrated_confidence 74.0% samples=22 daily_slot=1/2",
        (),
        None,
    )
    handler.emit(record)
    state = publisher.snapshot()

    assert state["boss"]["decision"] == "BUY"
    assert state["boss"]["strategy"] == "Trend Pullback #184"
    assert state["boss"]["confidence"] == 0.74
    assert _employee(state, "selector")["state"] in {"handoff", "done"}
    assert any(row["to"] == "boss" and row["document"] == "Strategy Recommendation" for row in state["handoffs"])


def test_session_veto_is_visible_as_blocked_guard(tmp_path):
    publisher = DashboardPublisher(tmp_path / "state.json")
    handler = DashboardLogHandler(publisher)
    record = logging.LogRecord(
        "xau-company", logging.INFO, __file__, 1,
        "Market-quality veto: XAU/USD session is closed", (), None,
    )
    handler.emit(record)
    state = publisher.snapshot()
    assert state["system"]["market_open"] is False
    assert _employee(state, "session")["state"] == "handoff"
    assert any(row["document"] == "Market Closed Veto" for row in state["handoffs"])


def test_dashboard_server_reads_real_signal_outcome_ledger(tmp_path):
    db = tmp_path / "outcomes.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE signal_outcomes (
          observed_at TEXT, direction TEXT, entry REAL, stop_loss REAL, take_profit REAL,
          calibrated_confidence REAL, strategy TEXT, regime TEXT, status TEXT,
          delivery_state TEXT, telegram_message_id TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO signal_outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("2026-08-31T01:00:00Z", "BUY", 2400.0, 2395.0, 2408.5, .75, "trend", "trend_up", "WIN", "SENT", "1"),
            ("2026-08-31T02:00:00Z", "SELL", 2410.0, 2415.0, 2401.5, .60, "breakout", "trend_down", "LOSS", "SENT", "2"),
        ],
    )
    conn.commit()
    conn.close()

    result = _read_performance(db)
    assert result is not None
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["win_rate"] == 0.5
    assert result["recent_signals"][0]["direction"] == "SELL"


def test_dashboard_frontend_assets_exist():
    root = Path(__file__).resolve().parents[1] / "dashboard"
    for name in ("index.html", "styles.css", "app.js"):
        path = root / name
        assert path.exists()
        assert path.stat().st_size > 500
