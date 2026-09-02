from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Literal

import numpy as np

from kalshi_research.domain.events import (
    FeeScheduleEvent,
    IndexTickEvent,
    MarketEvent,
    ResearchEvent,
    SettlementEvent,
    SpotTickEvent,
)
from kalshi_research.math.execution import FeeAccumulator, quadratic_trade_fee
from kalshi_research.math.probability import (
    brier_score,
    expected_calibration_error,
    log_loss,
)
from kalshi_research.research.acceptance import (
    PromotionDecision,
    ResearchMetrics,
    evaluate_for_probability_stage,
)
from kalshi_research.research.audit import AuditPolicy, DataQualityReport, audit_events
from kalshi_research.research.dataset import FeatureReplayPipeline
from kalshi_research.research.execution_replay import (
    ExecutionPolicy,
    ExecutionReplayError,
    ResearchOrderIntent,
    replay_orders,
)
from kalshi_research.research.experiments import select_horizon_rows
from kalshi_research.research.fees import FeeScheduleError, FeeScheduleTimeline
from kalshi_research.research.materializer import ModelFeatureRow
from kalshi_research.research.runner import events_digest
from kalshi_research.research.synchronizer import SynchronizationError
from kalshi_research.research.walkforward import expanding_walkforward
from kalshi_research.storage.sqlite_store import SqliteEventStore


class ResearchCompletionError(RuntimeError):
    """Raised only when the evidence itself is structurally untrustworthy."""


ABLATION_FEATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "baseline",
        (
            "normalized_distance_to_target",
            "kalshi_yes_mid",
        ),
    ),
    (
        "settlement",
        (
            "normalized_distance_to_target",
            "kalshi_yes_mid",
            "final_minute_progress",
            "settlement_gap_bps",
        ),
    ),
    (
        "external",
        (
            "normalized_distance_to_target",
            "kalshi_yes_mid",
            "final_minute_progress",
            "settlement_gap_bps",
            "brti_vs_external_bps",
        ),
    ),
    (
        "microstructure",
        (
            "normalized_distance_to_target",
            "kalshi_yes_mid",
            "final_minute_progress",
            "settlement_gap_bps",
            "brti_vs_external_bps",
            "kalshi_book_imbalance",
            "kalshi_spread",
        ),
    ),
)
FULL_STAGE = ABLATION_FEATURES[-1][0]


@dataclass(frozen=True, slots=True)
class CompletionPlan:
    decision_horizon_s: int = 60
    min_train_markets: int = 100
    validation_markets: int = 20
    test_markets: int = 20
    step_markets: int = 20
    l2_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    minimum_net_edge: float = 0.01
    order_quantity: Decimal = Decimal("1")
    base_latency_ms: int = 100
    latency_stress_ms: int = 600
    cost_stress_multiplier: Decimal = Decimal("1.5")
    max_book_age_ms: int = 2_000
    bankroll: Decimal | None = None
    min_executable_decisions: int = 500

    def __post_init__(self) -> None:
        if self.decision_horizon_s <= 0:
            raise ValueError("decision_horizon_s must be positive")
        if min(
            self.min_train_markets,
            self.validation_markets,
            self.test_markets,
            self.step_markets,
        ) <= 0:
            raise ValueError("walk-forward market counts must be positive")
        if self.step_markets != self.test_markets:
            raise ValueError("step_markets must equal test_markets so OOS test markets never overlap")
        if not self.l2_grid or any(value <= 0 or not math.isfinite(value) for value in self.l2_grid):
            raise ValueError("l2_grid must contain finite positive values")
        if tuple(sorted(set(self.l2_grid))) != self.l2_grid:
            raise ValueError("l2_grid must be unique and strictly increasing")
        if self.minimum_net_edge < 0 or not math.isfinite(self.minimum_net_edge):
            raise ValueError("minimum_net_edge must be finite and nonnegative")
        if self.order_quantity <= 0:
            raise ValueError("order_quantity must be positive")
        if min(self.base_latency_ms, self.latency_stress_ms, self.max_book_age_ms) < 0:
            raise ValueError("latency and book-age settings cannot be negative")
        if self.latency_stress_ms < self.base_latency_ms:
            raise ValueError("latency stress must be at least the base latency")
        if self.cost_stress_multiplier < 1:
            raise ValueError("cost stress multiplier must be at least one")
        if self.bankroll is not None and self.bankroll <= 0:
            raise ValueError("bankroll must be positive")
        if self.min_executable_decisions <= 0:
            raise ValueError("min_executable_decisions must be positive")

    @property
    def digest(self) -> str:
        payload = asdict(self)
        payload["order_quantity"] = str(self.order_quantity)
        payload["cost_stress_multiplier"] = str(self.cost_stress_multiplier)
        payload["bankroll"] = None if self.bankroll is None else str(self.bankroll)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class AblationScore:
    stage: str
    features: tuple[str, ...]
    count: int
    brier: float
    log_loss: float
    calibration_error: float
    market_brier: float
    market_log_loss: float


@dataclass(frozen=True, slots=True)
class FoldModelSummary:
    fold_index: int
    train_market_ids: tuple[str, ...]
    validation_market_ids: tuple[str, ...]
    test_market_ids: tuple[str, ...]
    l2_by_stage: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class OOSPrediction:
    fold_index: int
    market_ticker: str
    decision_recv_ts_ns: int
    predicted_yes: float
    market_yes_mid: float
    outcome: int
    selected_l2: float


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    ablations: tuple[AblationScore, ...]
    folds: tuple[FoldModelSummary, ...]
    full_predictions: tuple[OOSPrediction, ...]
    prediction_digest: str


@dataclass(frozen=True, slots=True)
class TradeSelection:
    market_ticker: str
    fold_index: int
    decision_recv_ts_ns: int
    predicted_yes: float
    outcome: int
    yes_ask: float | None
    no_ask: float | None
    yes_net_edge: float | None
    no_net_edge: float | None
    selected_side: Literal["yes", "no"] | None
    intent_id: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class CompletionEconomics:
    gross_pnl: float
    net_pnl: float
    max_drawdown: float
    executable_decisions: int
    rejected_intents: int
    profitable_walkforward_windows: int
    total_walkforward_windows: int
    latency_stress_net_pnl: float
    cost_stress_net_pnl: float


@dataclass(frozen=True, slots=True)
class ResearchCompletionReport:
    mode: str
    report_kind: str
    series_ticker: str
    order_placement: bool
    plan_digest: str
    event_count: int
    events_digest: str
    markets: tuple[str, ...]
    settled_market_count: int
    horizon_eligible_market_count: int
    audit: DataQualityReport
    model_spec_digest: str
    verdict: Literal["promoted", "rejected", "insufficient_evidence"]
    evidence_deficits: tuple[str, ...]
    promotion_reasons: tuple[str, ...]
    ablations: tuple[AblationScore, ...]
    folds: tuple[FoldModelSummary, ...]
    prediction_digest: str | None
    selections: tuple[TradeSelection, ...]
    metrics: ResearchMetrics | None
    economics: CompletionEconomics | None


@dataclass(frozen=True, slots=True)
class _Contract:
    market: MarketEvent
    settlement: SettlementEvent

    @property
    def outcome(self) -> int:
        return 1 if self.settlement.result == "yes" else 0


@dataclass(frozen=True, slots=True)
class _Transformer:
    features: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def design(self, rows: Sequence[ModelFeatureRow]) -> np.ndarray:
        matrix = np.zeros((len(rows), len(self.features) * 2), dtype=float)
        for row_index, row in enumerate(rows):
            for feature_index, feature in enumerate(self.features):
                value = _feature_value(row, feature)
                if value is None:
                    matrix[row_index, feature_index] = 0.0
                    matrix[row_index, len(self.features) + feature_index] = 1.0
                    continue
                matrix[row_index, feature_index] = (
                    value - self.means[feature_index]
                ) / self.scales[feature_index]
        return matrix


@dataclass(frozen=True, slots=True)
class _LogisticModel:
    transformer: _Transformer
    coefficients: tuple[float, ...]

    def predict(self, rows: Sequence[ModelFeatureRow]) -> list[float]:
        x = self.transformer.design(rows)
        x = np.column_stack((np.ones(len(rows), dtype=float), x))
        logits = np.clip(x @ np.asarray(self.coefficients, dtype=float), -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return [float(value) for value in probabilities]


def evaluate_oos_models(
    selected_rows: Mapping[str, ModelFeatureRow],
    outcomes_by_market: Mapping[str, int],
    ordered_market_ids: Sequence[str],
    plan: CompletionPlan,
) -> ModelEvaluation:
    eligible_ids = [
        market_id
        for market_id in ordered_market_ids
        if market_id in selected_rows and market_id in outcomes_by_market
    ]
    if len(set(eligible_ids)) != len(eligible_ids):
        raise ResearchCompletionError("ordered market ids contain duplicates")
    for market_id in eligible_ids:
        if outcomes_by_market[market_id] not in (0, 1):
            raise ResearchCompletionError(f"invalid binary outcome:{market_id}")

    folds = expanding_walkforward(
        eligible_ids,
        min_train=plan.min_train_markets,
        validation_size=plan.validation_markets,
        test_size=plan.test_markets,
        step=plan.step_markets,
    )
    if not folds:
        raise ResearchCompletionError("insufficient markets for completion model evaluation")

    predictions_by_stage: dict[str, list[tuple[str, float, float, int]]] = {
        stage: [] for stage, _ in ABLATION_FEATURES
    }
    full_predictions: list[OOSPrediction] = []
    fold_summaries: list[FoldModelSummary] = []
    seen_test: set[str] = set()

    for fold_index, fold in enumerate(folds):
        overlap = seen_test.intersection(fold.test)
        if overlap:
            raise ResearchCompletionError(
                "OOS test markets overlap across folds:" + ",".join(sorted(overlap))
            )
        seen_test.update(fold.test)

        train_rows, train_outcomes = _rows_and_outcomes(
            fold.train, selected_rows, outcomes_by_market
        )
        validation_rows, validation_outcomes = _rows_and_outcomes(
            fold.validation, selected_rows, outcomes_by_market
        )
        test_rows, test_outcomes = _rows_and_outcomes(
            fold.test, selected_rows, outcomes_by_market
        )

        l2_choices: list[tuple[str, float]] = []
        for stage, features in ABLATION_FEATURES:
            best_l2 = _choose_l2(
                train_rows,
                train_outcomes,
                validation_rows,
                validation_outcomes,
                features,
                plan.l2_grid,
            )
            final_model = _fit_logistic(
                train_rows + validation_rows,
                train_outcomes + validation_outcomes,
                features,
                best_l2,
            )
            probabilities = final_model.predict(test_rows)
            l2_choices.append((stage, best_l2))

            for market_id, row, probability, outcome in zip(
                fold.test, test_rows, probabilities, test_outcomes
            ):
                market_mid = row.kalshi_yes_mid
                if market_mid is None or not 0 <= market_mid <= 1:
                    raise ResearchCompletionError(
                        f"OOS row has no valid Kalshi midpoint:{market_id}"
                    )
                predictions_by_stage[stage].append(
                    (market_id, probability, float(market_mid), outcome)
                )
                if stage == FULL_STAGE:
                    full_predictions.append(
                        OOSPrediction(
                            fold_index=fold_index,
                            market_ticker=market_id,
                            decision_recv_ts_ns=row.decision_recv_ts_ns,
                            predicted_yes=probability,
                            market_yes_mid=float(market_mid),
                            outcome=outcome,
                            selected_l2=best_l2,
                        )
                    )

        fold_summaries.append(
            FoldModelSummary(
                fold_index=fold_index,
                train_market_ids=fold.train,
                validation_market_ids=fold.validation,
                test_market_ids=fold.test,
                l2_by_stage=tuple(l2_choices),
            )
        )

    ablations: list[AblationScore] = []
    for stage, features in ABLATION_FEATURES:
        stage_predictions = predictions_by_stage[stage]
        probabilities = [item[1] for item in stage_predictions]
        market_probabilities = [item[2] for item in stage_predictions]
        outcomes = [item[3] for item in stage_predictions]
        ablations.append(
            AblationScore(
                stage=stage,
                features=features,
                count=len(outcomes),
                brier=brier_score(probabilities, outcomes),
                log_loss=log_loss(probabilities, outcomes),
                calibration_error=expected_calibration_error(probabilities, outcomes),
                market_brier=brier_score(market_probabilities, outcomes),
                market_log_loss=log_loss(market_probabilities, outcomes),
            )
        )

    digest_payload = [
        {
            "fold_index": prediction.fold_index,
            "market_ticker": prediction.market_ticker,
            "decision_recv_ts_ns": prediction.decision_recv_ts_ns,
            "predicted_yes": prediction.predicted_yes,
            "market_yes_mid": prediction.market_yes_mid,
            "outcome": prediction.outcome,
            "selected_l2": prediction.selected_l2,
        }
        for prediction in full_predictions
    ]
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ModelEvaluation(
        ablations=tuple(ablations),
        folds=tuple(fold_summaries),
        full_predictions=tuple(full_predictions),
        prediction_digest=digest,
    )


def classify_research_verdict(
    metrics: ResearchMetrics | None,
    promotion: PromotionDecision | None,
    evidence_deficits: Sequence[str],
    *,
    minimum_executable_decisions: int,
) -> Literal["promoted", "rejected", "insufficient_evidence"]:
    if metrics is None or promotion is None:
        return "insufficient_evidence"
    if evidence_deficits or metrics.trade_count < minimum_executable_decisions:
        return "insufficient_evidence"
    return "promoted" if promotion.accepted else "rejected"


def run_research_completion_store(
    store: SqliteEventStore,
    *,
    plan: CompletionPlan | None = None,
    audit_policy: AuditPolicy | None = None,
    series_ticker: str = "KXBTC15M",
) -> ResearchCompletionReport:
    return run_research_completion_events(
        tuple(store.iter_events(order_by="receive")),
        plan=plan,
        audit_policy=audit_policy,
        series_ticker=series_ticker,
    )


def run_research_completion_events(
    events: Iterable[ResearchEvent],
    *,
    plan: CompletionPlan | None = None,
    audit_policy: AuditPolicy | None = None,
    series_ticker: str = "KXBTC15M",
) -> ResearchCompletionReport:
    plan = plan or CompletionPlan()
    audit_policy = audit_policy or AuditPolicy()
    materialized = tuple(sorted(events, key=lambda event: (event.recv_ts_ns, event.event_ts_ns)))
    if not materialized:
        raise ResearchCompletionError("research store is empty")

    audit = audit_events(materialized, policy=audit_policy)
    if not audit.passed:
        codes = ",".join(
            issue.code for issue in audit.issues if issue.severity == "critical"
        )
        raise ResearchCompletionError(
            f"structural audit failed:{codes or 'unknown_critical_issue'}"
        )

    contracts = _derive_contracts(materialized, series_ticker)
    settled_market_ids = tuple(contract.market.market_ticker for contract in contracts)
    model_spec_digest = _model_spec_digest()
    if not contracts:
        return _insufficient_report(
            materialized,
            audit,
            plan,
            series_ticker,
            settled_market_ids,
            model_spec_digest,
            deficits=("no_settled_contracts",),
        )

    outcomes = {contract.market.market_ticker: contract.outcome for contract in contracts}
    all_rows: list[ModelFeatureRow] = []
    for contract in contracts:
        safe_events = tuple(
            _feature_events_for_market(
                materialized,
                market_ticker=contract.market.market_ticker,
                open_ts_ns=contract.market.open_ts_ns,
                close_ts_ns=contract.market.close_ts_ns,
            )
        )
        try:
            rows = FeatureReplayPipeline(contract.market.market_ticker).run(safe_events)
            all_rows.extend(rows)
        except SynchronizationError as exc:
            raise ResearchCompletionError(
                f"feature replay failed for {contract.market.market_ticker}:{exc}"
            ) from exc

    selected_rows = select_horizon_rows(all_rows, plan.decision_horizon_s)
    eligible_ids = tuple(
        market_id for market_id in settled_market_ids if market_id in selected_rows
    )
    minimum_for_fold = (
        plan.min_train_markets + plan.validation_markets + plan.test_markets
    )
    if len(eligible_ids) < minimum_for_fold:
        return _insufficient_report(
            materialized,
            audit,
            plan,
            series_ticker,
            settled_market_ids,
            model_spec_digest,
            horizon_eligible=len(eligible_ids),
            deficits=(
                f"horizon_eligible_markets={len(eligible_ids)} required_at_least={minimum_for_fold}",
            ),
        )

    try:
        model_evaluation = evaluate_oos_models(
            selected_rows,
            outcomes,
            eligible_ids,
            plan,
        )
    except ResearchCompletionError:
        raise
    except Exception as exc:
        raise ResearchCompletionError(f"model evaluation failed:{exc}") from exc

    fee_events = tuple(
        event for event in materialized if isinstance(event, FeeScheduleEvent)
    )
    intents, selections, selection_deficits = _build_fee_aware_intents(
        model_evaluation.full_predictions,
        selected_rows,
        fee_events,
        plan,
        series_ticker,
    )

    base_policy = ExecutionPolicy(
        latency_ns=plan.base_latency_ms * 1_000_000,
        max_book_age_ns=plan.max_book_age_ms * 1_000_000,
        fee_stress_multiplier=Decimal("1"),
        cancellation_credit_fraction=Decimal("0"),
        bankroll=plan.bankroll,
    )
    latency_policy = ExecutionPolicy(
        latency_ns=plan.latency_stress_ms * 1_000_000,
        max_book_age_ns=plan.max_book_age_ms * 1_000_000,
        fee_stress_multiplier=Decimal("1"),
        cancellation_credit_fraction=Decimal("0"),
        bankroll=plan.bankroll,
    )
    cost_policy = ExecutionPolicy(
        latency_ns=plan.base_latency_ms * 1_000_000,
        max_book_age_ns=plan.max_book_age_ms * 1_000_000,
        fee_stress_multiplier=plan.cost_stress_multiplier,
        cancellation_credit_fraction=Decimal("0"),
        bankroll=plan.bankroll,
    )

    try:
        base_report = replay_orders(
            materialized,
            intents,
            policy=base_policy,
            series_ticker=series_ticker,
        )
        latency_report = replay_orders(
            materialized,
            intents,
            policy=latency_policy,
            series_ticker=series_ticker,
        )
        cost_report = replay_orders(
            materialized,
            intents,
            policy=cost_policy,
            series_ticker=series_ticker,
        )
    except ExecutionReplayError as exc:
        raise ResearchCompletionError(f"execution replay failed:{exc}") from exc

    full_score = next(
        score for score in model_evaluation.ablations if score.stage == FULL_STAGE
    )
    executable_decisions = base_report.filled_orders + base_report.partial_orders
    fold_pnl = _fold_net_pnl(model_evaluation, base_report.orders)
    profitable_windows = sum(value > 0 for value in fold_pnl)

    metrics = ResearchMetrics(
        test_brier=full_score.brier,
        baseline_brier=full_score.market_brier,
        test_log_loss=full_score.log_loss,
        baseline_log_loss=full_score.market_log_loss,
        calibration_error=full_score.calibration_error,
        net_pnl=float(base_report.net_pnl),
        gross_pnl=float(base_report.gross_pnl),
        max_drawdown=float(base_report.max_drawdown),
        trade_count=executable_decisions,
        profitable_walkforward_windows=profitable_windows,
        total_walkforward_windows=len(fold_pnl),
        latency_stress_net_pnl=float(latency_report.net_pnl),
        cost_stress_net_pnl=float(cost_report.net_pnl),
    )
    promotion = evaluate_for_probability_stage(metrics)

    deficits = list(selection_deficits)
    if executable_decisions < plan.min_executable_decisions:
        deficits.append(
            f"executable_oos_decisions={executable_decisions} required={plan.min_executable_decisions}"
        )
    verdict = classify_research_verdict(
        metrics,
        promotion,
        deficits,
        minimum_executable_decisions=plan.min_executable_decisions,
    )
    economics = CompletionEconomics(
        gross_pnl=float(base_report.gross_pnl),
        net_pnl=float(base_report.net_pnl),
        max_drawdown=float(base_report.max_drawdown),
        executable_decisions=executable_decisions,
        rejected_intents=base_report.rejected_orders,
        profitable_walkforward_windows=profitable_windows,
        total_walkforward_windows=len(fold_pnl),
        latency_stress_net_pnl=float(latency_report.net_pnl),
        cost_stress_net_pnl=float(cost_report.net_pnl),
    )

    return ResearchCompletionReport(
        mode="research_only",
        report_kind="research_complete_v1",
        series_ticker=series_ticker,
        order_placement=False,
        plan_digest=plan.digest,
        event_count=len(materialized),
        events_digest=events_digest(materialized),
        markets=settled_market_ids,
        settled_market_count=len(settled_market_ids),
        horizon_eligible_market_count=len(eligible_ids),
        audit=audit,
        model_spec_digest=model_spec_digest,
        verdict=verdict,
        evidence_deficits=tuple(dict.fromkeys(deficits)),
        promotion_reasons=promotion.reasons,
        ablations=model_evaluation.ablations,
        folds=model_evaluation.folds,
        prediction_digest=model_evaluation.prediction_digest,
        selections=selections,
        metrics=metrics,
        economics=economics,
    )


def _insufficient_report(
    events: Sequence[ResearchEvent],
    audit: DataQualityReport,
    plan: CompletionPlan,
    series_ticker: str,
    markets: tuple[str, ...],
    model_spec_digest: str,
    *,
    horizon_eligible: int = 0,
    deficits: tuple[str, ...],
) -> ResearchCompletionReport:
    return ResearchCompletionReport(
        mode="research_only",
        report_kind="research_complete_v1",
        series_ticker=series_ticker,
        order_placement=False,
        plan_digest=plan.digest,
        event_count=len(events),
        events_digest=events_digest(events),
        markets=markets,
        settled_market_count=len(markets),
        horizon_eligible_market_count=horizon_eligible,
        audit=audit,
        model_spec_digest=model_spec_digest,
        verdict="insufficient_evidence",
        evidence_deficits=deficits,
        promotion_reasons=(),
        ablations=(),
        folds=(),
        prediction_digest=None,
        selections=(),
        metrics=None,
        economics=None,
    )


def _derive_contracts(
    events: Sequence[ResearchEvent], series_ticker: str
) -> tuple[_Contract, ...]:
    markets: dict[str, MarketEvent] = {}
    cores: dict[str, tuple[object, ...]] = {}
    settlements: dict[str, SettlementEvent] = {}

    for event in events:
        if not isinstance(event, MarketEvent) or event.series_ticker != series_ticker:
            continue
        core = (
            event.event_ticker,
            event.series_ticker,
            event.target_price,
            event.open_ts_ns,
            event.close_ts_ns,
        )
        previous = cores.get(event.market_ticker)
        if previous is not None and previous != core:
            raise ResearchCompletionError(
                f"conflicting market metadata:{event.market_ticker}"
            )
        cores[event.market_ticker] = core
        markets[event.market_ticker] = event

    for event in events:
        if not isinstance(event, SettlementEvent):
            continue
        market = markets.get(event.market_ticker)
        if market is None:
            if event.market_ticker.startswith(series_ticker):
                raise ResearchCompletionError(
                    f"settlement lacks market metadata:{event.market_ticker}"
                )
            continue
        if event.target_price != market.target_price:
            raise ResearchCompletionError(
                f"settlement target mismatch:{event.market_ticker}"
            )
        if event.recv_ts_ns < market.close_ts_ns:
            raise ResearchCompletionError(
                f"settlement received before close:{event.market_ticker}"
            )
        previous = settlements.get(event.market_ticker)
        if previous is not None and (
            previous.final_value != event.final_value or previous.result != event.result
        ):
            raise ResearchCompletionError(
                f"conflicting settlement labels:{event.market_ticker}"
            )
        settlements[event.market_ticker] = event

    contracts = [
        _Contract(market=market, settlement=settlements[ticker])
        for ticker, market in markets.items()
        if ticker in settlements
    ]
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
        if isinstance(event, SettlementEvent):
            continue
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


def _rows_and_outcomes(
    market_ids: Sequence[str],
    rows: Mapping[str, ModelFeatureRow],
    outcomes: Mapping[str, int],
) -> tuple[list[ModelFeatureRow], list[int]]:
    row_list: list[ModelFeatureRow] = []
    outcome_list: list[int] = []
    for market_id in market_ids:
        if market_id not in rows or market_id not in outcomes:
            raise ResearchCompletionError(f"fold references unavailable market:{market_id}")
        row_list.append(rows[market_id])
        outcome_list.append(outcomes[market_id])
    return row_list, outcome_list


def _feature_value(row: ModelFeatureRow, feature: str) -> float | None:
    if feature == "settlement_gap_bps":
        required = row.required_remaining_brti_average
        target = row.target_price
        if required is None or target is None or required <= 0 or target <= 0:
            return None
        value = 10_000.0 * math.log(required / target)
    else:
        raw = getattr(row, feature)
        if raw is None:
            return None
        value = float(raw)
    return value if math.isfinite(value) else None


def _fit_transformer(
    rows: Sequence[ModelFeatureRow], features: tuple[str, ...]
) -> _Transformer:
    means: list[float] = []
    scales: list[float] = []
    for feature in features:
        values = [
            value
            for row in rows
            if (value := _feature_value(row, feature)) is not None
        ]
        if not values:
            means.append(0.0)
            scales.append(1.0)
            continue
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        scale = math.sqrt(variance)
        means.append(mean)
        scales.append(scale if scale > 1e-12 else 1.0)
    return _Transformer(features, tuple(means), tuple(scales))


def _fit_logistic(
    rows: Sequence[ModelFeatureRow],
    outcomes: Sequence[int],
    features: tuple[str, ...],
    l2: float,
) -> _LogisticModel:
    if not rows or len(rows) != len(outcomes):
        raise ResearchCompletionError("logistic fit requires equal nonempty rows/outcomes")
    transformer = _fit_transformer(rows, features)
    design = transformer.design(rows)
    x = np.column_stack((np.ones(len(rows), dtype=float), design))
    y = np.asarray(outcomes, dtype=float)
    coefficients = np.zeros(x.shape[1], dtype=float)
    penalty = np.eye(x.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    jitter = np.eye(x.shape[1], dtype=float) * 1e-8

    for _ in range(80):
        logits = np.clip(x @ coefficients, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = x.T @ (probabilities - y) + l2 * (penalty @ coefficients)
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-8)
        hessian = (x.T * weights) @ x + l2 * penalty + jitter
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        coefficients -= step
        if float(np.linalg.norm(step)) < 1e-8:
            break

    if not np.all(np.isfinite(coefficients)):
        raise ResearchCompletionError("logistic fit produced non-finite coefficients")
    return _LogisticModel(
        transformer=transformer,
        coefficients=tuple(float(value) for value in coefficients),
    )


def _choose_l2(
    train_rows: Sequence[ModelFeatureRow],
    train_outcomes: Sequence[int],
    validation_rows: Sequence[ModelFeatureRow],
    validation_outcomes: Sequence[int],
    features: tuple[str, ...],
    l2_grid: Sequence[float],
) -> float:
    best_l2: float | None = None
    best_loss = math.inf
    for l2 in l2_grid:
        model = _fit_logistic(train_rows, train_outcomes, features, l2)
        probabilities = model.predict(validation_rows)
        candidate_loss = log_loss(probabilities, validation_outcomes)
        if candidate_loss < best_loss - 1e-15:
            best_loss = candidate_loss
            best_l2 = l2
    if best_l2 is None:
        raise ResearchCompletionError("failed to choose logistic regularization")
    return best_l2


def _estimate_entry_fee(
    *,
    price: Decimal,
    quantity: Decimal,
    schedule: FeeScheduleEvent,
) -> Decimal:
    trade_fee = quadratic_trade_fee(
        price=price,
        quantity=quantity,
        fee_multiplier=schedule.fee_multiplier,
        liquidity_role="taker",
        fee_type=schedule.fee_type,
    )
    breakdown = FeeAccumulator().apply_fill(
        revenue=-(price * quantity),
        trade_fee=trade_fee,
    )
    return breakdown.net_fee


def _build_fee_aware_intents(
    predictions: Sequence[OOSPrediction],
    rows: Mapping[str, ModelFeatureRow],
    fee_events: Sequence[FeeScheduleEvent],
    plan: CompletionPlan,
    series_ticker: str,
) -> tuple[tuple[ResearchOrderIntent, ...], tuple[TradeSelection, ...], tuple[str, ...]]:
    timeline: FeeScheduleTimeline | None
    try:
        timeline = FeeScheduleTimeline.from_events(
            fee_events, series_ticker=series_ticker
        )
    except FeeScheduleError:
        timeline = None

    intents: list[ResearchOrderIntent] = []
    selections: list[TradeSelection] = []
    fee_unavailable = 0
    quote_unavailable = 0

    for prediction in predictions:
        row = rows[prediction.market_ticker]
        if row.kalshi_yes_ask is None or row.kalshi_yes_bid is None:
            quote_unavailable += 1
            selections.append(
                _no_selection(prediction, reason="missing_executable_quote")
            )
            continue

        yes_ask = float(row.kalshi_yes_ask)
        no_ask = 1.0 - float(row.kalshi_yes_bid)
        if not (0 <= yes_ask <= 1 and 0 <= no_ask <= 1):
            raise ResearchCompletionError(
                f"invalid executable binary quote:{prediction.market_ticker}"
            )
        if timeline is None:
            fee_unavailable += 1
            selections.append(
                _no_selection(
                    prediction,
                    reason="fee_schedule_unavailable",
                    yes_ask=yes_ask,
                    no_ask=no_ask,
                )
            )
            continue

        try:
            schedule = timeline.at(
                prediction.decision_recv_ts_ns,
                knowledge_cutoff_ns=prediction.decision_recv_ts_ns,
                allow_posthoc_history=False,
            )
            yes_fee = _estimate_entry_fee(
                price=Decimal(str(yes_ask)),
                quantity=plan.order_quantity,
                schedule=schedule,
            )
            no_fee = _estimate_entry_fee(
                price=Decimal(str(no_ask)),
                quantity=plan.order_quantity,
                schedule=schedule,
            )
        except (FeeScheduleError, ValueError):
            fee_unavailable += 1
            selections.append(
                _no_selection(
                    prediction,
                    reason="fee_schedule_unavailable",
                    yes_ask=yes_ask,
                    no_ask=no_ask,
                )
            )
            continue

        yes_net_edge = (
            prediction.predicted_yes
            - yes_ask
            - float(yes_fee / plan.order_quantity)
        )
        no_net_edge = (
            (1.0 - prediction.predicted_yes)
            - no_ask
            - float(no_fee / plan.order_quantity)
        )
        best_side: Literal["yes", "no"] = (
            "yes" if yes_net_edge >= no_net_edge else "no"
        )
        best_edge = max(yes_net_edge, no_net_edge)
        if best_edge < plan.minimum_net_edge:
            selections.append(
                TradeSelection(
                    market_ticker=prediction.market_ticker,
                    fold_index=prediction.fold_index,
                    decision_recv_ts_ns=prediction.decision_recv_ts_ns,
                    predicted_yes=prediction.predicted_yes,
                    outcome=prediction.outcome,
                    yes_ask=yes_ask,
                    no_ask=no_ask,
                    yes_net_edge=yes_net_edge,
                    no_net_edge=no_net_edge,
                    selected_side=None,
                    intent_id=None,
                    reason="edge_below_threshold",
                )
            )
            continue

        intent_id = (
            f"oos:{prediction.fold_index}:{prediction.market_ticker}:{best_side}"
        )
        intents.append(
            ResearchOrderIntent(
                intent_id=intent_id,
                market_ticker=prediction.market_ticker,
                decision_recv_ts_ns=prediction.decision_recv_ts_ns,
                outcome_side=best_side,
                quantity=plan.order_quantity,
                liquidity_role="taker",
                attribution="full_model_oos",
            )
        )
        selections.append(
            TradeSelection(
                market_ticker=prediction.market_ticker,
                fold_index=prediction.fold_index,
                decision_recv_ts_ns=prediction.decision_recv_ts_ns,
                predicted_yes=prediction.predicted_yes,
                outcome=prediction.outcome,
                yes_ask=yes_ask,
                no_ask=no_ask,
                yes_net_edge=yes_net_edge,
                no_net_edge=no_net_edge,
                selected_side=best_side,
                intent_id=intent_id,
                reason="submitted_to_execution_replay",
            )
        )

    deficits: list[str] = []
    if fee_unavailable:
        deficits.append(f"fee_schedule_coverage_incomplete={fee_unavailable}")
    if quote_unavailable:
        deficits.append(f"executable_quote_coverage_incomplete={quote_unavailable}")
    return tuple(intents), tuple(selections), tuple(deficits)


def _no_selection(
    prediction: OOSPrediction,
    *,
    reason: str,
    yes_ask: float | None = None,
    no_ask: float | None = None,
) -> TradeSelection:
    return TradeSelection(
        market_ticker=prediction.market_ticker,
        fold_index=prediction.fold_index,
        decision_recv_ts_ns=prediction.decision_recv_ts_ns,
        predicted_yes=prediction.predicted_yes,
        outcome=prediction.outcome,
        yes_ask=yes_ask,
        no_ask=no_ask,
        yes_net_edge=None,
        no_net_edge=None,
        selected_side=None,
        intent_id=None,
        reason=reason,
    )


def _fold_net_pnl(
    evaluation: ModelEvaluation,
    orders: Sequence,
) -> tuple[float, ...]:
    fold_by_market = {
        prediction.market_ticker: prediction.fold_index
        for prediction in evaluation.full_predictions
    }
    totals = [0.0 for _ in evaluation.folds]
    for order in orders:
        fold_index = fold_by_market.get(order.market_ticker)
        if fold_index is not None:
            totals[fold_index] += float(order.net_pnl)
    return tuple(totals)


def _model_spec_digest() -> str:
    payload = {
        "algorithm": "deterministic_l2_logistic_irls_v1",
        "standardization": "training_only_mean_std",
        "missingness": "training_only_zero_imputation_plus_missing_indicator",
        "intercept_regularized": False,
        "selection_metric": "validation_log_loss",
        "stages": [
            {"stage": stage, "features": list(features)}
            for stage, features in ABLATION_FEATURES
        ],
        "execution": "existing_receive_time_depth_replay_v0.8",
        "label_access": "test_labels_scoring_only",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
