from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from kalshi_research.domain.events import (
    IndexTickEvent,
    MarketEvent,
    OrderbookDeltaEvent,
    OrderbookSnapshotEvent,
    ResearchEvent,
    SettlementEvent,
)


@dataclass(frozen=True, slots=True)
class AuditIssue:
    severity: str
    code: str
    detail: str
    market_ticker: str | None = None


@dataclass(frozen=True, slots=True)
class SettlementReconciliation:
    market_ticker: str
    target_price: Decimal
    official_final_value: Decimal
    official_result: str
    computed_result: str
    result_matches_math: bool
    market_result_matches: bool | None
    market_value_difference: Decimal | None
    brti_final_minute_average: Decimal | None
    brti_value_difference: Decimal | None
    brti_within_tolerance: bool | None


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    total_events: int
    counts_by_source: dict[str, int]
    counts_by_kind: dict[str, int]
    receive_time_regressions: int
    orderbook_sequence_gaps: int
    orderbook_sequence_regressions: int
    orderbook_deltas_without_snapshot: int
    index_sequence_gaps: int
    index_sequence_regressions: int
    brti_sample_count_regressions: int
    negative_latency_events: int
    settlement_reconciliations: tuple[SettlementReconciliation, ...]
    issues: tuple[AuditIssue, ...]

    @property
    def critical_issue_count(self) -> int:
        return sum(issue.severity == "critical" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def passed(self) -> bool:
        return self.critical_issue_count == 0


@dataclass(frozen=True, slots=True)
class AuditPolicy:
    settlement_value_tolerance: Decimal = Decimal("0.01")
    settlement_window_tolerance_ns: int = 5_000_000_000

    def __post_init__(self) -> None:
        if self.settlement_value_tolerance < 0:
            raise ValueError("settlement_value_tolerance cannot be negative")
        if self.settlement_window_tolerance_ns < 0:
            raise ValueError("settlement_window_tolerance_ns cannot be negative")


def audit_events(
    events: Iterable[ResearchEvent],
    *,
    policy: AuditPolicy | None = None,
) -> DataQualityReport:
    """Audit canonical events without repairing or silently reordering them."""
    policy = policy or AuditPolicy()
    source_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    issues: list[AuditIssue] = []

    total = 0
    recv_regressions = 0
    negative_latency = 0
    last_recv_ns: int | None = None

    book_last_seq: dict[tuple[str, str], int] = {}
    book_has_snapshot: set[tuple[str, str]] = set()
    book_gaps = 0
    book_regressions = 0
    deltas_without_snapshot = 0

    index_last_seq: dict[tuple[str, str], int] = {}
    index_gaps = 0
    index_regressions = 0

    brti_sample_last: dict[int, int] = {}
    brti_sample_regressions = 0
    complete_brti_windows: list[IndexTickEvent] = []

    markets: dict[str, MarketEvent] = {}
    settlements: list[SettlementEvent] = []

    for event in events:
        total += 1
        source_counts[str(event.source)] += 1
        kind_counts[str(event.kind)] += 1

        if last_recv_ns is not None and event.recv_ts_ns < last_recv_ns:
            recv_regressions += 1
            issues.append(
                AuditIssue(
                    severity="critical",
                    code="receive_time_regression",
                    detail=(
                        f"recv_ts_ns moved backward from {last_recv_ns} "
                        f"to {event.recv_ts_ns}"
                    ),
                    market_ticker=event.market_ticker,
                )
            )
        last_recv_ns = event.recv_ts_ns

        if event.latency_ms < 0:
            negative_latency += 1

        if isinstance(event, MarketEvent):
            markets[event.market_ticker] = event

        if isinstance(event, OrderbookSnapshotEvent):
            key = _book_stream_key(event)
            book_has_snapshot.add(key)
            book_last_seq[key] = event.seq

        elif isinstance(event, OrderbookDeltaEvent):
            key = _book_stream_key(event)
            previous = book_last_seq.get(key)
            if key not in book_has_snapshot or previous is None:
                deltas_without_snapshot += 1
                issues.append(
                    AuditIssue(
                        severity="critical",
                        code="orderbook_delta_without_snapshot",
                        detail=f"delta seq={event.seq} has no prior snapshot for {key}",
                        market_ticker=event.market_ticker,
                    )
                )
            else:
                if event.seq <= previous:
                    book_regressions += 1
                    issues.append(
                        AuditIssue(
                            severity="critical",
                            code="orderbook_sequence_regression",
                            detail=f"book seq moved from {previous} to {event.seq} for {key}",
                            market_ticker=event.market_ticker,
                        )
                    )
                elif event.seq != previous + 1:
                    book_gaps += 1
                    issues.append(
                        AuditIssue(
                            severity="critical",
                            code="orderbook_sequence_gap",
                            detail=f"book seq jumped from {previous} to {event.seq} for {key}",
                            market_ticker=event.market_ticker,
                        )
                    )
            book_last_seq[key] = event.seq

        if isinstance(event, IndexTickEvent):
            if event.sid is not None and event.seq is not None:
                key = (event.index_id, str(event.sid))
                previous = index_last_seq.get(key)
                if previous is not None:
                    if event.seq <= previous:
                        index_regressions += 1
                        issues.append(
                            AuditIssue(
                                severity="critical",
                                code="index_sequence_regression",
                                detail=(
                                    f"{event.index_id} seq moved from {previous} "
                                    f"to {event.seq} for sid={event.sid}"
                                ),
                            )
                        )
                    elif event.seq != previous + 1:
                        index_gaps += 1
                        issues.append(
                            AuditIssue(
                                severity="critical",
                                code="index_sequence_gap",
                                detail=(
                                    f"{event.index_id} seq jumped from {previous} "
                                    f"to {event.seq} for sid={event.sid}"
                                ),
                            )
                        )
                index_last_seq[key] = event.seq

            if (
                event.final_minute_window_start_ts_ns is not None
                and event.final_minute_sample_count is not None
            ):
                window = event.final_minute_window_start_ts_ns
                previous_count = brti_sample_last.get(window)
                if previous_count is not None and event.final_minute_sample_count < previous_count:
                    brti_sample_regressions += 1
                    issues.append(
                        AuditIssue(
                            severity="critical",
                            code="brti_sample_count_regression",
                            detail=(
                                f"final-minute sample count moved from {previous_count} "
                                f"to {event.final_minute_sample_count} for window={window}"
                            ),
                        )
                    )
                brti_sample_last[window] = event.final_minute_sample_count

            if (
                event.final_minute_sample_count == 60
                and event.final_minute_average is not None
                and event.final_minute_window_end_ts_ns is not None
            ):
                complete_brti_windows.append(event)

        if isinstance(event, SettlementEvent):
            settlements.append(event)

    reconciliations = tuple(
        _reconcile_settlement(
            settlement,
            markets.get(settlement.market_ticker),
            complete_brti_windows,
            policy,
            issues,
        )
        for settlement in settlements
    )

    return DataQualityReport(
        total_events=total,
        counts_by_source=dict(source_counts),
        counts_by_kind=dict(kind_counts),
        receive_time_regressions=recv_regressions,
        orderbook_sequence_gaps=book_gaps,
        orderbook_sequence_regressions=book_regressions,
        orderbook_deltas_without_snapshot=deltas_without_snapshot,
        index_sequence_gaps=index_gaps,
        index_sequence_regressions=index_regressions,
        brti_sample_count_regressions=brti_sample_regressions,
        negative_latency_events=negative_latency,
        settlement_reconciliations=reconciliations,
        issues=tuple(issues),
    )


def _book_stream_key(
    event: OrderbookSnapshotEvent | OrderbookDeltaEvent,
) -> tuple[str, str]:
    if event.sid is not None:
        return ("sid", str(event.sid))
    return ("market", event.market_ticker)


def _reconcile_settlement(
    settlement: SettlementEvent,
    market: MarketEvent | None,
    complete_brti_windows: list[IndexTickEvent],
    policy: AuditPolicy,
    issues: list[AuditIssue],
) -> SettlementReconciliation:
    computed_result = "yes" if settlement.final_value >= settlement.target_price else "no"
    result_matches_math = settlement.result == computed_result
    if not result_matches_math:
        issues.append(
            AuditIssue(
                severity="critical",
                code="settlement_result_math_mismatch",
                detail=(
                    f"official result={settlement.result}, but final_value={settlement.final_value} "
                    f"and target={settlement.target_price} imply {computed_result}"
                ),
                market_ticker=settlement.market_ticker,
            )
        )

    market_result_matches: bool | None = None
    market_value_difference: Decimal | None = None
    if market is not None:
        if market.result is not None:
            market_result_matches = market.result.lower() == settlement.result
            if not market_result_matches:
                issues.append(
                    AuditIssue(
                        severity="critical",
                        code="market_settlement_result_mismatch",
                        detail=(
                            f"market metadata result={market.result}, "
                            f"settlement result={settlement.result}"
                        ),
                        market_ticker=settlement.market_ticker,
                    )
                )
        if market.settlement_value is not None:
            market_value_difference = abs(market.settlement_value - settlement.final_value)
            if market_value_difference > policy.settlement_value_tolerance:
                issues.append(
                    AuditIssue(
                        severity="critical",
                        code="market_settlement_value_mismatch",
                        detail=(
                            f"market settlement value differs by {market_value_difference}"
                        ),
                        market_ticker=settlement.market_ticker,
                    )
                )

    brti_average: Decimal | None = None
    brti_difference: Decimal | None = None
    brti_within: bool | None = None
    if market is not None:
        candidate = _nearest_complete_brti_window(
            complete_brti_windows,
            market.close_ts_ns,
            policy.settlement_window_tolerance_ns,
        )
        if candidate is not None:
            brti_average = candidate.final_minute_average
            if brti_average is not None:
                brti_difference = abs(brti_average - settlement.final_value)
                brti_within = brti_difference <= policy.settlement_value_tolerance
                if not brti_within:
                    issues.append(
                        AuditIssue(
                            severity="critical",
                            code="brti_settlement_value_mismatch",
                            detail=(
                                f"captured 60-sample BRTI average differs from official "
                                f"settlement by {brti_difference}"
                            ),
                            market_ticker=settlement.market_ticker,
                        )
                    )
        else:
            issues.append(
                AuditIssue(
                    severity="warning",
                    code="complete_brti_window_missing",
                    detail="no captured 60-sample BRTI window could be matched to market close",
                    market_ticker=settlement.market_ticker,
                )
            )

    return SettlementReconciliation(
        market_ticker=settlement.market_ticker,
        target_price=settlement.target_price,
        official_final_value=settlement.final_value,
        official_result=settlement.result,
        computed_result=computed_result,
        result_matches_math=result_matches_math,
        market_result_matches=market_result_matches,
        market_value_difference=market_value_difference,
        brti_final_minute_average=brti_average,
        brti_value_difference=brti_difference,
        brti_within_tolerance=brti_within,
    )


def _nearest_complete_brti_window(
    windows: list[IndexTickEvent],
    close_ts_ns: int,
    tolerance_ns: int,
) -> IndexTickEvent | None:
    if not windows:
        return None
    candidate = min(
        windows,
        key=lambda event: abs((event.final_minute_window_end_ts_ns or 0) - close_ts_ns),
    )
    distance = abs((candidate.final_minute_window_end_ts_ns or 0) - close_ts_ns)
    return candidate if distance <= tolerance_ns else None
