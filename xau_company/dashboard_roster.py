from __future__ import annotations

from copy import deepcopy

from . import dashboard

_REMOVED_EMPLOYEE_IDS = {"strategy_evolution", "strategy_invention"}


def _reconcile_saved_roster() -> None:
    """Make persisted dashboard state match the current canonical roster."""
    original = dashboard.DashboardPublisher._ensure_loaded
    if getattr(original, "_canonical_roster_wrapped", False):
        return

    def wrapped(self) -> None:
        original(self)
        state = self._state
        if not isinstance(state, dict):
            return

        defaults = dashboard.default_state()
        existing_rows = state.get("employees")
        if not isinstance(existing_rows, list):
            existing_rows = []

        by_id = {
            row.get("id"): row
            for row in existing_rows
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        }
        canonical: list[dict[str, object]] = []
        for template in defaults["employees"]:
            employee_id = template["id"]
            row = deepcopy(template)
            saved = by_id.get(employee_id)
            if isinstance(saved, dict):
                row.update(saved)
            row["id"] = template["id"]
            row["name"] = template["name"]
            row["room"] = template["room"]
            canonical.append(row)

        state["employees"] = canonical
        state["rooms"] = [dict(room) for room in dashboard.ROOMS]
        state["version"] = max(3, int(state.get("version") or 1))

    wrapped._canonical_roster_wrapped = True  # type: ignore[attr-defined]
    dashboard.DashboardPublisher._ensure_loaded = wrapped


def install_canonical_dashboard_roster() -> None:
    """Permanently remove retired strategy-generation employees from the floor."""
    dashboard.EMPLOYEES = tuple(
        employee
        for employee in dashboard.EMPLOYEES
        if employee.get("id") not in _REMOVED_EMPLOYEE_IDS
    )
    _reconcile_saved_roster()
