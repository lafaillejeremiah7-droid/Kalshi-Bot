from __future__ import annotations

from kalshi_research.domain.events import HealthEvent, Source
from kalshi_research.research.evidence_status import _source_evidence, evidence_readiness_from_events


def test_empty_evidence_reports_collection_targets() -> None:
    readiness = evidence_readiness_from_events((), now_ns=10_000_000_000)

    assert readiness.phase == "collecting"
    assert readiness.verdict == "insufficient_evidence"
    assert readiness.first_oos_market_requirement == 140
    assert readiness.markets_until_first_oos == 140
    assert readiness.executable_decision_target == 500
    assert readiness.executable_decisions_remaining == 500
    assert readiness.executable_progress_fraction == 0.0
    assert readiness.evidence_deficits == ("research_store_empty",)
    assert readiness.order_placement is False


def test_source_evidence_reports_counts_and_receive_age() -> None:
    event = HealthEvent(
        source=Source.SYSTEM,
        event_ts_ns=1_000_000_000,
        recv_ts_ns=2_000_000_000,
        component="clock",
        status="ok",
    )

    sources = _source_evidence((event,), now_ns=5_000_000_000)

    assert len(sources) == 1
    assert sources[0].source == "system"
    assert sources[0].event_count == 1
    assert sources[0].latest_recv_ts_ns == 2_000_000_000
    assert sources[0].age_seconds == 3.0
