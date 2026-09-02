from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kalshi_research.cli import cmd_capture_all, cmd_evidence_status, cmd_ops_status
from kalshi_research.config import ResearchConfig


def test_evidence_status_missing_db_does_not_create_file(
    tmp_path: Path,
    capsys,
) -> None:
    db_path = tmp_path / "missing.sqlite3"
    args = argparse.Namespace(db=str(db_path))

    assert cmd_evidence_status(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["phase"] == "collecting"
    assert payload["verdict"] == "insufficient_evidence"
    assert payload["reason"] == "research_db_not_found"
    assert payload["executable_decisions_remaining"] == 500
    assert payload["order_placement"] is False
    assert not db_path.exists()


def test_ops_status_missing_heartbeat_is_not_running(
    tmp_path: Path,
    capsys,
) -> None:
    db_path = tmp_path / "data" / "research.sqlite3"
    args = argparse.Namespace(db=str(db_path))

    assert cmd_ops_status(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_running"
    assert payload["reason"] == "status_file_not_found"
    assert payload["order_placement"] is False
    assert not db_path.exists()


def test_ops_status_reports_fresh_running_supervisor(
    tmp_path: Path,
    capsys,
) -> None:
    db_path = tmp_path / "data" / "research.sqlite3"
    status_path = db_path.parent / "ops" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "mode": "research_capture_only",
                "order_placement": False,
                "updated_ts_ns": time.time_ns(),
                "components": {
                    "kalshi": {"status": "running"},
                    "coinbase": {"status": "running"},
                    "kraken": {"status": "running"},
                    "evaluator": {"status": "waiting"},
                },
            }
        ),
        encoding="utf-8",
    )

    assert cmd_ops_status(argparse.Namespace(db=str(db_path))) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "running"
    assert payload["heartbeat_stale"] is False
    assert payload["heartbeat_age_seconds"] >= 0.0
    assert payload["order_placement"] is False


def test_ops_status_rejects_tampered_trading_authority(
    tmp_path: Path,
    capsys,
) -> None:
    db_path = tmp_path / "data" / "research.sqlite3"
    status_path = db_path.parent / "ops" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "mode": "research_capture_only",
                "order_placement": True,
                "updated_ts_ns": time.time_ns(),
                "components": {},
            }
        ),
        encoding="utf-8",
    )

    assert cmd_ops_status(argparse.Namespace(db=str(db_path))) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["order_placement"] is False
    assert "research-only invariant" in payload["reason"]


def test_capture_all_missing_credentials_blocks_before_network(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    config = ResearchConfig(
        research_db_path=tmp_path / "data" / "research.sqlite3",
        raw_capture_dir=tmp_path / "data" / "raw",
        report_archive_dir=tmp_path / "data" / "experiments",
    )
    monkeypatch.setattr(
        ResearchConfig,
        "from_env",
        classmethod(lambda cls: config),
    )
    args = argparse.Namespace(
        market_poll_seconds=5.0,
        restart_delay_seconds=2.0,
        evaluation_interval_seconds=900.0,
        heartbeat_seconds=5.0,
        no_evaluate=False,
    )

    assert cmd_capture_all(args) == 2

    output = capsys.readouterr().out
    assert "KALSHI_API_KEY_ID" in output
    assert '"order_placement": false' in output
