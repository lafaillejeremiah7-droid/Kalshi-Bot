from __future__ import annotations

import re
from copy import deepcopy

from . import dashboard

_INVENTION_RE = re.compile(
    r"invented_new_families=(\d+).*invented_new_variants=(\d+).*"
    r"invention_promoted=(\d+).*invention_quarantined=(\d+).*"
    r"invented_family_total=(\d+).*invented_family_promoted=(\d+)"
)


def _install_roster_reconciliation() -> None:
    """Upgrade restored dashboard state to the current canonical employee roster.

    Runtime state is persisted between on-demand sessions. Older snapshots may
    predate newly-added dashboard employees, so loading them verbatim can leave
    the live floor permanently short a bot. Reconciliation preserves every
    known employee's live task/state while inserting missing canonical rows and
    dropping stale duplicates/unknown rows.
    """

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
            # Identity and room assignment are controlled by the current code,
            # not by a potentially stale persisted snapshot.
            row["id"] = template["id"]
            row["name"] = template["name"]
            row["room"] = template["room"]
            canonical.append(row)

        state["employees"] = canonical
        state["rooms"] = [dict(room) for room in dashboard.ROOMS]
        state["version"] = max(2, int(state.get("version") or 1))

    wrapped._canonical_roster_wrapped = True  # type: ignore[attr-defined]
    dashboard.DashboardPublisher._ensure_loaded = wrapped


def install_invention_dashboard() -> None:
    """Extend dashboard telemetry with the Strategy Invention employee.

    Kept separate from the trading runtime so dashboard failures remain fail-open.
    """
    if not any(e.get("id") == "strategy_invention" for e in dashboard.EMPLOYEES):
        employees = list(dashboard.EMPLOYEES)
        insert_at = next(
            (i + 1 for i, e in enumerate(employees) if e.get("id") == "strategy_evolution"),
            len(employees),
        )
        employees.insert(
            insert_at,
            {"id": "strategy_invention", "name": "Strategy Invention", "room": "research"},
        )
        dashboard.EMPLOYEES = tuple(employees)

    _install_roster_reconciliation()

    original = dashboard.DashboardLogHandler._handle
    if getattr(original, "_strategy_invention_wrapped", False):
        return

    def wrapped(self, message: str, level: int) -> None:
        original(self, message, level)
        if not message.startswith("Strategy lab universe="):
            return
        match = _INVENTION_RE.search(message)
        if not match:
            return
        new_families, new_variants, promoted, quarantined, total, promoted_families = map(int, match.groups())
        state = "done" if new_families or promoted or quarantined else "idle"
        task = (
            f"Invented {new_families} families / {new_variants} variants"
            if new_families
            else "Invention catalog audited"
        )
        detail = (
            f"Total families={total}; promoted families={promoted_families}; "
            f"new promoted variants={promoted}; quarantined variants={quarantined}"
        )
        self.publisher.employee("strategy_invention", state, task, detail=detail)
        if new_variants:
            self.publisher.handoff("research_lab", "strategy_invention", "Invention Brief")
            self.publisher.handoff(
                "strategy_invention",
                "backtest_auditor",
                "Invented Family Pack",
                detail,
            )

    wrapped._strategy_invention_wrapped = True  # type: ignore[attr-defined]
    dashboard.DashboardLogHandler._handle = wrapped
