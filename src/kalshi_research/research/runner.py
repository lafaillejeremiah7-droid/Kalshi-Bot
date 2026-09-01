from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from kalshi_research.domain.events import (
    IndexTickEvent,
    MarketEvent,
    ResearchEvent,
    SettlementEvent,
    Source,
    SpotTickEvent,
)
from kalshi_research.research.audit import (
    AuditPolicy,
    DataQualityReport,
    audit_events,
)
from kalshi_research.research.coverage import (
    CoveragePolicy,
    FeatureCoverageReport,
    audit_feature_coverage,
)
from kalshi_research.research.dataset import (
    FeatureReplayPipeline,
    feature_rows_digest,
)
from kalshi_research.research.experiments import (
    ExperimentError,
    ExperimentPlan,
    FoldProbabilityBenchmark,
    LeadLagReport,
    TimedPrice,
    benchmark_probability_walkforward,
    scan_receive_time_lead_lag,
)
from kalshi_research.research.materializer import ModelFeatureRow
from kalshi_research.research.synchronizer import SynchronizationError
from kalshi_research.storage.sqlite_store import SqliteEventStore


class ResearchRunError(RuntimeError):
    """Raised when stored research data fails a fail-closed experiment contract."""


@dataclass(frozen=True, slots=True)
class MarketResearchSummary:
    market_ticker: str
    open_ts_ns: int
    close_ts_ns: int
    settlement_recv_ts_ns: int
    outcome: int
    feature_rows: int
    feature_digest: str
    coverage: FeatureCoverageReport


@dataclass(frozen=True, slots=True)
class VenueLeadLagSummary:
    venue: str
    available: bool
    reason: str | None
    report: LeadLagReport | None


@dataclass(frozen=True, slots=True)
class ResearchRunReport:
    mode: str
    series_ticker: str
    order_placement: bool
    plan_digest: str
    event_count: int
    events_digest: str
    audit: DataQualityReport
    markets: tuple[MarketResearchSummary, ...]
    probability_benchmarks: tuple[FoldProbabilityBenchmark, ...]
    lead_lag: tuple[VenueLeadLagSummary, ...]

    @property
    def market_count(self) -> int:
        return len(self.markets)


@dataclass(frozen=True, slots=True)
class _Contract:
    market: MarketEvent
    settlement: SettlementEvent

    @property
    def outcome(self) -> int:
        return 1 if self.settlement.result == "yes" else 0


def run_research_store(
    store: SqliteEventStore,
    *,
    plan: ExperimentPlan | None = None,
    audit_policy: AuditPolicy | None = None,
    coverage_policy: CoveragePolicy | None = None,
    series_ticker: str = "KXBTC15M",
) -> ResearchRunReport:
    """Run the predeclared research suite from canonical receive-time storage."""
    return run_research_events(
        tuple(store.iter_events(order_by="receive")),
        plan=plan,
        audit_policy=audit_policy,
        coverage_policy=coverage_policy,
        series_ticker=series_ticker,
    )


def run_research_events(
    events: Iterable[ResearchEvent],
    *,
    plan: ExperimentPlan | None = None,
    audit_policy: AuditPolicy | None = None,
    coverage_policy: CoveragePolicy | None = None,
    series_ticker: str = "KXBTC15M",
) -> ResearchRunReport:
    """Audit, replay, and evaluate stored research data without trading authority.

    Settlement events are used only to construct evaluation labels after feature
    replay. Per-market feature streams stop at the market close receive-time
    frontier, so post-close labels and observations can never enter predictors.
    """
    plan = plan or ExperimentPlan()
    audit_policy = audit_policy or AuditPolicy()
    coverage_policy = coverage_policy or CoveragePolicy()
    materialized_events = tuple(events)
    if not materialized_events:
        raise ResearchRunError("research store is empty")

    audit = audit_events(materialized_events, policy=audit_policy)
    if not audit.passed:
        codes = ",".join(issue.code for issue in audit.issues if issue.severity == "critical")
        raise ResearchRunError(f"structural audit failed:{codes or 'unknown_critical_issue'}")

    contracts = _derive_contracts(materialized_events, series_ticker)
    ordered_market_ids = [contract.market.market_ticker for contract in contracts]
    outcomes = {contract.market.market_ticker: contract.outcome for contract in contracts}

    all_feature_rows: list[ModelFeatureRow] = []
    market_summaries: list[MarketResearchSummary] = []
    for contract in contracts:
        ticker = contract.market.market_ticker
        safe_events = tuple(
            _feature_events_for_market(
                materialized_events,
                market_ticker=ticker,
                open_ts_ns=contract.market.open_ts_ns,
                close_ts_ns=contract.market.close_ts_ns,
            )
        )
        pipeline = FeatureReplayPipeline(market_ticker=ticker)
        try:
            rows = list(pipeline.run(safe_events))
        except SynchronizationError as exc:
            raise ResearchRunError(f"feature replay failed for {ticker}:{exc}") from exc

        coverage = audit_feature_coverage(rows, policy=coverage_policy)
        if not coverage.passed:
            failures = ",".join(coverage.failures)
            raise ResearchRunError(f"feature coverage failed for {ticker}:{failures}")

        digest = feature_rows_digest(rows)
        all_feature_rows.extend(rows)
        market_summaries.append(
            MarketResearchSummary(
                market_ticker=ticker,
                open_ts_ns=contract.market.open_ts_ns,
                close_ts_ns=contract.market.close_ts_ns,
                settlement_recv_ts_ns=contract.settlement.recv_ts_ns,
                outcome=contract.outcome,
                feature_rows=len(rows),
                feature_digest=digest,
                coverage=coverage,
            )
        )

    try:
        probability_benchmarks = benchmark_probability_walkforward(
            all_feature_rows,
            outcomes,
            ordered_market_ids,
            plan,
        )
    except ExperimentError as exc:
        raise ResearchRunError(f"probability experiment failed:{exc}") from exc
    if not probability_benchmarks:
        raise ResearchRunError("probability experiment produced no evaluable walk-forward results")

    window_start = min(contract.market.open_ts_ns for contract in contracts)
    window_end = max(contract.market.close_ts_ns for contract in contracts)
    lead_lag = _lead_lag_reports(
        materialized_events,
        plan,
        window_start_ns=window_start,
        window_end_ns=window_end,
    )

    return ResearchRunReport(
        mode="research_only",
        series_ticker=series_ticker,
        order_placement=False,
        plan_digest=plan.digest,
        event_count=len(materialized_events),
        events_digest=events_digest(materialized_events),
        audit=audit,
        markets=tuple(market_summaries),
        probability_benchmarks=probability_benchmarks,
        lead_lag=lead_lag,
    )


def events_digest(events: Iterable[ResearchEvent]) -> str:
    """Hash canonical events in the exact order supplied to the experiment."""
    hasher = hashlib.sha256()
    for event in events:
        payload = json.dumps(
            event.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        hasher.update(payload.encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def research_report_json(report: ResearchRunReport, *, indent: int | None = None) -> str:
    return json.dumps(
        _jsonable(report),
        sort_keys=True,
        separators=None if indent is not None else (",", ":"),
        indent=indent,
        allow_nan=False,
    )


def research_report_digest(report: ResearchRunReport) -> str:
    return hashlib.sha256(research_report_json(report).encode()).hexdigest()


def _derive_contracts(
    events: Sequence[ResearchEvent],
    series_ticker: str,
) -> tuple[_Contract, ...]:
    markets: dict[str, MarketEvent] = {}
    market_core: dict[str, tuple[object, ...]] = {}
    settlements: dict[str, SettlementEvent] = {}

    for event in events:
        if isinstance(event, MarketEvent) and event.series_ticker == series_ticker:
            core = (
                event.event_ticker,
                event.series_ticker,
                event.target_price,
                event.open_ts_ns,
                event.close_ts_ns,
            )
            previous_core = market_core.get(event.market_ticker)
            if previous_core is not None and previous_core != core:
                raise ResearchRunError(
                    f"conflicting market metadata for {event.market_ticker}"
                )
            market_core[event.market_ticker] = core
            markets[event.market_ticker] = event

    if not markets:
        raise ResearchRunError(f"no {series_ticker} market metadata found")

    for event in events:
        if not isinstance(event, SettlementEvent):
            continue
        is_target = event.market_ticker in markets or event.market_ticker.startswith(series_ticker)
        if not is_target:
            continue
        market = markets.get(event.market_ticker)
        if market is None:
            raise ResearchRunError(
                f"settlement lacks {series_ticker} market metadata:{event.market_ticker}"
            )
        if event.target_price != market.target_price:
            raise ResearchRunError(
                f"settlement target mismatch for {event.market_ticker}"
            )
        if event.recv_ts_ns < market.close_ts_ns:
            raise ResearchRunError(
                f"settlement received before market close:{event.market_ticker}"
            )
        previous = settlements.get(event.market_ticker)
        if previous is not None and (
            previous.target_price != event.target_price
            or previous.final_value != event.final_value
            or previous.result != event.result
        ):
            raise ResearchRunError(
                f"conflicting settlement labels for {event.market_ticker}"
            )
        settlements[event.market_ticker] = event

    contracts = [
        _Contract(market=market, settlement=settlements[ticker])
        for ticker, market in markets.items()
        if ticker in settlements
    ]
    if not contracts:
        raise ResearchRunError(f"no settled {series_ticker} contracts with explicit labels")

    contracts.sort(key=lambda contract: (contract.market.close_ts_ns, contract.market.market_ticker))
    return tuple(contracts)


def _feature_events_for_market(
    events: Sequence[ResearchEvent],
    *,
    market_ticker: str,
    open_ts_ns: int,
    close_ts_ns: int,
) -> Iterable[ResearchEvent]:
    for event in events:
        if event.recv_ts_ns > close_ts_ns:
            continue
        if isinstance(event, MarketEvent) and event.market_ticker == market_ticker:
            yield event
            continue
        if event.recv_ts_ns < open_ts_ns:
            continue
        if event.market_ticker is None or event.market_ticker == market_ticker:
            yield event
            continue
        if isinstance(event, (IndexTickEvent, SpotTickEvent)):
            yield event


def _lead_lag_reports(
    events: Sequence[ResearchEvent],
    plan: ExperimentPlan,
    *,
    window_start_ns: int,
    window_end_ns: int,
) -> tuple[VenueLeadLagSummary, ...]:
    window_events = tuple(
        event
        for event in events
        if window_start_ns <= event.recv_ts_ns <= window_end_ns
    )
    follower = _brti_prices(window_events)
    summaries: list[VenueLeadLagSummary] = []
    for source in (Source.COINBASE, Source.KRAKEN):
        leader = _spot_prices(window_events, source)
        if len(follower) < 2:
            summaries.append(
                VenueLeadLagSummary(
                    venue=str(source),
                    available=False,
                    reason="fewer_than_two_brti_observations",
                    report=None,
                )
            )
            continue
        if len(leader) < 2:
            summaries.append(
                VenueLeadLagSummary(
                    venue=str(source),
                    available=False,
                    reason="fewer_than_two_venue_observations",
                    report=None,
                )
            )
            continue
        summaries.append(
            VenueLeadLagSummary(
                venue=str(source),
                available=True,
                reason=None,
                report=scan_receive_time_lead_lag(leader, follower, plan),
            )
        )
    return tuple(summaries)


def _brti_prices(events: Sequence[ResearchEvent]) -> tuple[TimedPrice, ...]:
    by_receive: dict[int, float] = {}
    for event in events:
        if isinstance(event, IndexTickEvent):
            by_receive[event.recv_ts_ns] = float(event.value)
    return tuple(TimedPrice(timestamp, by_receive[timestamp]) for timestamp in sorted(by_receive))


def _spot_prices(
    events: Sequence[ResearchEvent],
    source: Source,
) -> tuple[TimedPrice, ...]:
    by_receive: dict[int, float] = {}
    for event in events:
        if not isinstance(event, SpotTickEvent) or event.source != source:
            continue
        price = _spot_reference_price(event)
        if price is not None:
            by_receive[event.recv_ts_ns] = price
    return tuple(TimedPrice(timestamp, by_receive[timestamp]) for timestamp in sorted(by_receive))


def _spot_reference_price(event: SpotTickEvent) -> float | None:
    if event.bid is not None and event.ask is not None:
        return float((event.bid + event.ask) / Decimal("2"))
    if event.last is not None:
        return float(event.last)
    return None


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("research report contains a non-finite float")
        return value
    return value
