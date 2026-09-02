from __future__ import annotations

from dataclasses import replace

import pytest

from kalshi_research.research.audit import DataQualityReport
from kalshi_research.research.registry import ExperimentReportArchive, ReportArchiveError
from kalshi_research.research.runner import ResearchRunReport, research_report_digest


def _audit() -> DataQualityReport:
    return DataQualityReport(
        total_events=1,
        counts_by_source={},
        counts_by_kind={},
        receive_time_regressions=0,
        orderbook_sequence_gaps=0,
        orderbook_sequence_regressions=0,
        orderbook_deltas_without_snapshot=0,
        index_sequence_gaps=0,
        index_sequence_regressions=0,
        brti_sample_count_regressions=0,
        negative_latency_events=0,
        settlement_reconciliations=(),
        issues=(),
    )


def _report(events_digest: str = "a" * 64) -> ResearchRunReport:
    return ResearchRunReport(
        mode="research_only",
        series_ticker="KXBTC15M",
        order_placement=False,
        plan_digest="b" * 64,
        event_count=1,
        events_digest=events_digest,
        audit=_audit(),
        markets=(),
        probability_benchmarks=(),
        lead_lag=(),
    )


def test_publish_is_content_addressed_and_idempotent(tmp_path):
    archive = ExperimentReportArchive(tmp_path / "experiments")
    report = _report()

    first = archive.publish(report)
    second = archive.publish(report)

    assert first == second
    assert first.digest == research_report_digest(report)
    assert first.path.name == f"{first.digest}.json"
    assert first.path.exists()
    assert archive.list() == (first,)
    assert archive.read_payload(first.digest)["order_placement"] is False


def test_distinct_report_creates_distinct_immutable_entry(tmp_path):
    archive = ExperimentReportArchive(tmp_path / "experiments")
    first = archive.publish(_report("a" * 64))
    second = archive.publish(_report("c" * 64))

    assert first.digest != second.digest
    assert len(archive.list()) == 2
    assert first.path.read_text(encoding="utf-8") != second.path.read_text(encoding="utf-8")


def test_archive_detects_tampering(tmp_path):
    archive = ExperimentReportArchive(tmp_path / "experiments")
    entry = archive.publish(_report())
    entry.path.write_text(entry.path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ReportArchiveError, match="archive digest mismatch"):
        archive.get(entry.digest)


def test_archive_rejects_non_research_payload_even_with_matching_filename(tmp_path):
    archive = ExperimentReportArchive(tmp_path / "experiments")
    report = replace(_report(), order_placement=True)
    digest = research_report_digest(report)
    archive.root.mkdir(parents=True)
    path = archive.root / f"{digest}.json"
    from kalshi_research.research.runner import research_report_json

    path.write_text(research_report_json(report), encoding="utf-8")

    with pytest.raises(ReportArchiveError, match="non-research report"):
        archive.get(digest)


def test_invalid_digest_is_rejected_before_path_access(tmp_path):
    archive = ExperimentReportArchive(tmp_path / "experiments")

    with pytest.raises(ReportArchiveError, match="lowercase SHA-256"):
        archive.get("../not-a-digest")
