from __future__ import annotations

from collections.abc import Iterable

from kalshi_research.domain.events import ResearchEvent
from kalshi_research.research.audit import AuditPolicy, audit_events
from kalshi_research.research.complete import (
    CompletionPlan,
    ResearchCompletionError,
    ResearchCompletionReport,
    run_research_completion_events as _run_completion_core,
)
from kalshi_research.storage.sqlite_store import SqliteEventStore


def run_research_completion_events(
    events: Iterable[ResearchEvent],
    *,
    plan: CompletionPlan | None = None,
    audit_policy: AuditPolicy | None = None,
    series_ticker: str = "KXBTC15M",
) -> ResearchCompletionReport:
    """Audit evidence in supplied order before any downstream normalization.

    The completion core may normalize event ordering after this boundary for
    deterministic replay, but it is never allowed to hide a receive-time
    regression that existed in the evidence supplied by the caller.
    """
    materialized = tuple(events)
    if not materialized:
        raise ResearchCompletionError("research store is empty")

    policy = audit_policy or AuditPolicy()
    audit = audit_events(materialized, policy=policy)
    if not audit.passed:
        codes = ",".join(
            issue.code for issue in audit.issues if issue.severity == "critical"
        )
        raise ResearchCompletionError(
            f"structural audit failed:{codes or 'unknown_critical_issue'}"
        )

    return _run_completion_core(
        materialized,
        plan=plan,
        audit_policy=policy,
        series_ticker=series_ticker,
    )


def run_research_completion_store(
    store: SqliteEventStore,
    *,
    plan: CompletionPlan | None = None,
    audit_policy: AuditPolicy | None = None,
    series_ticker: str = "KXBTC15M",
) -> ResearchCompletionReport:
    """Run completion from canonical receive-order storage via the audited boundary."""
    return run_research_completion_events(
        tuple(store.iter_events(order_by="receive")),
        plan=plan,
        audit_policy=audit_policy,
        series_ticker=series_ticker,
    )
