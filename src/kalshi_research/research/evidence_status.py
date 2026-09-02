from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal

from kalshi_research.domain.events import ResearchEvent
from kalshi_research.research.complete import CompletionPlan
from kalshi_research.research.completion_entrypoint import run_research_completion_events
from kalshi_research.research.runner import research_report_digest
from kalshi_research.storage.sqlite_store import SqliteEventStore


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    source: str
    event_count: int
    latest_recv_ts_ns: int
    age_seconds: float


@dataclass(frozen=True, slots=True)
class EvidenceReadiness:
    mode: str
    order_placement: bool
    phase: Literal["collecting", "evaluated", "promoted"]
    verdict: Literal["promoted", "rejected", "insufficient_evidence"]
    report_digest: str | None
    event_count: int
    market_count: int
    settled_market_count: int
    horizon_eligible_market_count: int
    first_oos_market_requirement: int
    markets_until_first_oos: int
    selected_trade_intents: int
    executable_decisions: int
    executable_decision_target: int
    executable_decisions_remaining: int
    executable_progress_fraction: float
    evidence_deficits: tuple[str, ...]
    promotion_reasons: tuple[str, ...]
    source_evidence: tuple[SourceEvidence, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _source_evidence(
    events: tuple[ResearchEvent, ...],
    *,
    now_ns: int,
) -> tuple[SourceEvidence, ...]:
    counts = Counter(str(event.source) for event in events)
    latest: dict[str, int] = {}
    for event in events:
        source = str(event.source)
        latest[source] = max(latest.get(source, 0), event.recv_ts_ns)

    return tuple(
        SourceEvidence(
            source=source,
            event_count=counts[source],
            latest_recv_ts_ns=latest[source],
            age_seconds=max(0.0, (now_ns - latest[source]) / 1_000_000_000),
        )
        for source in sorted(counts)
    )


def evidence_readiness_from_events(
    events: tuple[ResearchEvent, ...],
    *,
    plan: CompletionPlan | None = None,
    now_ns: int | None = None,
) -> EvidenceReadiness:
    """Summarize progress toward the immutable OOS promotion gate.

    Structural corruption is intentionally not converted into a friendly status:
    the completion entrypoint still raises in that case. Sparse or empty evidence
    is reported as collection progress instead of being treated as a software error.
    """
    selected_plan = plan or CompletionPlan()
    timestamp_ns = time.time_ns() if now_ns is None else now_ns
    first_oos_requirement = (
        selected_plan.min_train_markets
        + selected_plan.validation_markets
        + selected_plan.test_markets
    )
    market_count = len(
        {event.market_ticker for event in events if event.market_ticker is not None}
    )
    sources = _source_evidence(events, now_ns=timestamp_ns)

    if not events:
        return EvidenceReadiness(
            mode="research_only",
            order_placement=False,
            phase="collecting",
            verdict="insufficient_evidence",
            report_digest=None,
            event_count=0,
            market_count=0,
            settled_market_count=0,
            horizon_eligible_market_count=0,
            first_oos_market_requirement=first_oos_requirement,
            markets_until_first_oos=first_oos_requirement,
            selected_trade_intents=0,
            executable_decisions=0,
            executable_decision_target=selected_plan.min_executable_decisions,
            executable_decisions_remaining=selected_plan.min_executable_decisions,
            executable_progress_fraction=0.0,
            evidence_deficits=("research_store_empty",),
            promotion_reasons=(),
            source_evidence=(),
        )

    report = run_research_completion_events(events, plan=selected_plan)
    executable_decisions = 0 if report.economics is None else report.economics.executable_decisions
    selected_intents = sum(selection.intent_id is not None for selection in report.selections)
    target = selected_plan.min_executable_decisions
    phase: Literal["collecting", "evaluated", "promoted"]
    if report.verdict == "promoted":
        phase = "promoted"
    elif report.verdict == "insufficient_evidence":
        phase = "collecting"
    else:
        phase = "evaluated"

    return EvidenceReadiness(
        mode="research_only",
        order_placement=False,
        phase=phase,
        verdict=report.verdict,
        report_digest=research_report_digest(report),
        event_count=report.event_count,
        market_count=market_count,
        settled_market_count=report.settled_market_count,
        horizon_eligible_market_count=report.horizon_eligible_market_count,
        first_oos_market_requirement=first_oos_requirement,
        markets_until_first_oos=max(0, first_oos_requirement - report.horizon_eligible_market_count),
        selected_trade_intents=selected_intents,
        executable_decisions=executable_decisions,
        executable_decision_target=target,
        executable_decisions_remaining=max(0, target - executable_decisions),
        executable_progress_fraction=min(1.0, executable_decisions / target),
        evidence_deficits=report.evidence_deficits,
        promotion_reasons=report.promotion_reasons,
        source_evidence=sources,
    )


def evidence_readiness_store(
    store: SqliteEventStore,
    *,
    plan: CompletionPlan | None = None,
    now_ns: int | None = None,
) -> EvidenceReadiness:
    return evidence_readiness_from_events(
        tuple(store.iter_events(order_by="receive")),
        plan=plan,
        now_ns=now_ns,
    )
