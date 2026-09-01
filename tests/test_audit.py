from decimal import Decimal

from kalshi_research.domain.events import (
    IndexTickEvent,
    MarketEvent,
    OrderbookDeltaEvent,
    OrderbookSnapshotEvent,
    PriceLevel,
    SettlementEvent,
)
from kalshi_research.research.audit import AuditPolicy, audit_events


T0 = 1_800_000_000_000_000_000
TICKER = "KXBTC15M-TEST"
CLOSE = T0 + 60_000_000_000


def market(*, result: str | None = "yes", settlement_value: str | None = "100.01"):
    return MarketEvent(
        event_ts_ns=T0,
        recv_ts_ns=T0,
        market_ticker=TICKER,
        event_ticker="KXBTC15M",
        series_ticker="KXBTC15M",
        target_price=Decimal("100"),
        open_ts_ns=T0 - 900_000_000_000,
        close_ts_ns=CLOSE,
        status="settled",
        result=result,
        settlement_value=Decimal(settlement_value) if settlement_value is not None else None,
    )


def snapshot(seq: int = 10, recv: int = T0 + 1_000_000):
    return OrderbookSnapshotEvent(
        event_ts_ns=recv,
        recv_ts_ns=recv,
        market_ticker=TICKER,
        sid=1,
        seq=seq,
        yes_bids=(PriceLevel(price=Decimal("0.50"), size=Decimal("2")),),
        no_bids=(PriceLevel(price=Decimal("0.49"), size=Decimal("3")),),
    )


def delta(seq: int, recv: int):
    return OrderbookDeltaEvent(
        event_ts_ns=recv,
        recv_ts_ns=recv,
        market_ticker=TICKER,
        sid=1,
        seq=seq,
        side="yes",
        price=Decimal("0.51"),
        delta=Decimal("1"),
    )


def brti(
    *,
    seq: int = 5,
    recv: int = T0 + 3_000_000,
    sample_count: int = 60,
    average: str = "100.01",
    window_start: int = CLOSE - 60_000_000_000,
):
    return IndexTickEvent(
        event_ts_ns=recv,
        recv_ts_ns=recv,
        sid=2,
        seq=seq,
        value=Decimal("100.01"),
        final_minute_average=Decimal(average),
        final_minute_sample_count=sample_count,
        final_minute_window_start_ts_ns=window_start,
        final_minute_window_end_ts_ns=CLOSE,
    )


def settlement(*, result: str = "yes", final_value: str = "100.01", recv: int = T0 + 4_000_000):
    return SettlementEvent(
        event_ts_ns=recv,
        recv_ts_ns=recv,
        market_ticker=TICKER,
        target_price=Decimal("100"),
        final_value=Decimal(final_value),
        result=result,
    )


def test_clean_capture_and_settlement_pass():
    report = audit_events(
        [
            market(),
            snapshot(),
            delta(11, T0 + 2_000_000),
            brti(),
            settlement(),
        ]
    )

    assert report.passed
    assert report.critical_issue_count == 0
    assert report.orderbook_sequence_gaps == 0
    assert len(report.settlement_reconciliations) == 1
    rec = report.settlement_reconciliations[0]
    assert rec.result_matches_math
    assert rec.market_result_matches
    assert rec.brti_within_tolerance
    assert rec.brti_value_difference == Decimal("0.00")


def test_orderbook_gap_is_critical():
    report = audit_events([snapshot(), delta(12, T0 + 2_000_000)])

    assert not report.passed
    assert report.orderbook_sequence_gaps == 1
    assert any(issue.code == "orderbook_sequence_gap" for issue in report.issues)


def test_delta_without_snapshot_is_critical():
    report = audit_events([delta(11, T0 + 2_000_000)])

    assert not report.passed
    assert report.orderbook_deltas_without_snapshot == 1


def test_receive_time_regression_is_critical():
    report = audit_events([snapshot(recv=T0 + 2_000_000), market()])

    assert not report.passed
    assert report.receive_time_regressions == 1


def test_settlement_result_must_match_binary_math():
    report = audit_events([market(result="no", settlement_value="100.01"), settlement(result="no")])

    assert not report.passed
    rec = report.settlement_reconciliations[0]
    assert rec.computed_result == "yes"
    assert not rec.result_matches_math
    assert any(issue.code == "settlement_result_math_mismatch" for issue in report.issues)


def test_complete_brti_average_must_match_official_settlement_within_tolerance():
    report = audit_events(
        [market(), brti(average="100.04"), settlement()],
        policy=AuditPolicy(settlement_value_tolerance=Decimal("0.01")),
    )

    assert not report.passed
    rec = report.settlement_reconciliations[0]
    assert rec.brti_value_difference == Decimal("0.03")
    assert rec.brti_within_tolerance is False
    assert any(issue.code == "brti_settlement_value_mismatch" for issue in report.issues)


def test_missing_complete_brti_window_is_warning_not_critical():
    report = audit_events([market(), settlement()])

    assert report.passed
    assert report.warning_count == 1
    assert any(issue.code == "complete_brti_window_missing" for issue in report.issues)


def test_brti_sample_count_regression_in_same_window_is_critical():
    first = brti(seq=5, recv=T0 + 3_000_000, sample_count=30, average="99.9")
    second = brti(seq=6, recv=T0 + 4_000_000, sample_count=29, average="99.9")
    report = audit_events([first, second])

    assert not report.passed
    assert report.brti_sample_count_regressions == 1


def test_index_sequence_gap_is_critical():
    first = brti(seq=5, recv=T0 + 3_000_000, sample_count=20)
    second = brti(seq=7, recv=T0 + 4_000_000, sample_count=21)
    report = audit_events([first, second])

    assert not report.passed
    assert report.index_sequence_gaps == 1
