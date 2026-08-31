from __future__ import annotations

import re

from . import dashboard

_INVENTION_RE = re.compile(
    r"invented_new_families=(\d+).*invented_new_variants=(\d+).*"
    r"invention_promoted=(\d+).*invention_quarantined=(\d+).*"
    r"invented_family_total=(\d+).*invented_family_promoted=(\d+)"
)


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
