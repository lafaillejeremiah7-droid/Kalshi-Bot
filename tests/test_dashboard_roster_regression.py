from __future__ import annotations

import json
from pathlib import Path

from dashboard_server import _read_state
from xau_company.dashboard import DashboardPublisher, default_state


def _legacy_state_with_retired_employees() -> dict:
    state = default_state()
    state["version"] = 2
    employees = list(state["employees"])
    insert_at = next(i for i, row in enumerate(employees) if row["id"] == "backtest_auditor")
    employees[insert_at:insert_at] = [
        {
            "id": "strategy_evolution",
            "name": "Strategy Evolution",
            "room": "research",
            "state": "idle",
            "task": "Retired",
            "detail": "",
            "direction": "HOLD",
            "confidence": None,
            "updated_at": None,
        },
        {
            "id": "strategy_invention",
            "name": "Strategy Invention",
            "room": "research",
            "state": "idle",
            "task": "Retired",
            "detail": "",
            "direction": "HOLD",
            "confidence": None,
            "updated_at": None,
        },
    ]
    state["employees"] = employees
    market_data = next(employee for employee in state["employees"] if employee["id"] == "market_data")
    market_data["state"] = "done"
    market_data["task"] = "Preserve this live task"
    return state


def _assert_canonical_26(state: dict) -> None:
    employees = state["employees"]
    ids = [employee["id"] for employee in employees]
    assert len(ids) == 26
    assert len(set(ids)) == 26
    assert "strategy_evolution" not in ids
    assert "strategy_invention" not in ids


def test_publisher_removes_retired_employees_from_restored_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "dashboard_state.json"
    path.write_text(json.dumps(_legacy_state_with_retired_employees()), encoding="utf-8")

    state = DashboardPublisher(path).snapshot()

    _assert_canonical_26(state)
    market_data = next(employee for employee in state["employees"] if employee["id"] == "market_data")
    assert market_data["state"] == "done"
    assert market_data["task"] == "Preserve this live task"


def test_dashboard_server_uses_same_roster_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "dashboard_state.json"
    path.write_text(json.dumps(_legacy_state_with_retired_employees()), encoding="utf-8")

    state = _read_state(path)

    _assert_canonical_26(state)


def test_frontend_counts_online_roster_not_only_non_idle_workers() -> None:
    app = Path("dashboard/app.js").read_text(encoding="utf-8")

    assert 'els.activeCount.textContent = String(employees.length);' in app
    assert 'if (normalized !== "idle") active += 1;' not in app
    assert 'nextRosterSignature !== rosterSignature' in app
