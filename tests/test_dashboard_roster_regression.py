from __future__ import annotations

import json
from pathlib import Path

from dashboard_server import _read_state
from xau_company.dashboard import DashboardPublisher, default_state


def _legacy_27_employee_state() -> dict:
    state = default_state()
    state["version"] = 1
    state["employees"] = [
        employee
        for employee in state["employees"]
        if employee["id"] != "strategy_invention"
    ]
    market_data = next(employee for employee in state["employees"] if employee["id"] == "market_data")
    market_data["state"] = "done"
    market_data["task"] = "Preserve this live task"
    return state


def _assert_canonical_28(state: dict) -> None:
    employees = state["employees"]
    ids = [employee["id"] for employee in employees]
    assert len(ids) == 28
    assert len(set(ids)) == 28
    assert "strategy_invention" in ids
    assert ids.index("strategy_evolution") < ids.index("strategy_invention") < ids.index("backtest_auditor")


def test_publisher_upgrades_restored_27_employee_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "dashboard_state.json"
    path.write_text(json.dumps(_legacy_27_employee_state()), encoding="utf-8")

    state = DashboardPublisher(path).snapshot()

    _assert_canonical_28(state)
    market_data = next(employee for employee in state["employees"] if employee["id"] == "market_data")
    assert market_data["state"] == "done"
    assert market_data["task"] == "Preserve this live task"


def test_dashboard_server_uses_same_roster_upgrade(tmp_path: Path) -> None:
    path = tmp_path / "dashboard_state.json"
    path.write_text(json.dumps(_legacy_27_employee_state()), encoding="utf-8")

    state = _read_state(path)

    _assert_canonical_28(state)


def test_frontend_counts_online_roster_not_only_non_idle_workers() -> None:
    app = Path("dashboard/app.js").read_text(encoding="utf-8")

    assert 'els.activeCount.textContent = String(employees.length);' in app
    assert 'if (normalized !== "idle") active += 1;' not in app
    assert 'nextRosterSignature !== rosterSignature' in app
