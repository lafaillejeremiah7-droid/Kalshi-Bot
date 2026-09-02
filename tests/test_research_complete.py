from __future__ import annotations

from decimal import Decimal

from kalshi_research.research.acceptance import PromotionDecision, ResearchMetrics
from kalshi_research.research.audit import DataQualityReport
from kalshi_research.research.complete import (
    ABLATION_FEATURES,
    CompletionPlan,
    ResearchCompletionReport,
    classify_research_verdict,
    evaluate_oos_models,
)
from kalshi_research.research.materializer import ModelFeatureRow
from kalshi_research.research.registry import ExperimentReportArchive
from kalshi_research.research.runner import research_report_digest


def _row(index: int, outcome: int, *, missing_extras: bool = False) -> ModelFeatureRow:
    signal = 2.0 if outcome else -2.0
    return ModelFeatureRow(
        decision_recv_ts_ns=1_900_000_000_000_000_000 + index * 1_000_000_000,
        market_ticker=f"KXBTC15M-M{index}",
        probability_ready=True,
        baseline_ready=True,
        seconds_to_close=60.0,
        target_price=100.0,
        brti=101.0 if outcome else 99.0,
        brti_log_distance_to_target=0.01 if outcome else -0.01,
        brti_vol_per_sqrt_second=0.001,
        normalized_distance_to_target=signal,
        kalshi_yes_bid=0.49,
        kalshi_yes_ask=0.51,
        kalshi_yes_mid=0.50,
        kalshi_spread=0.02,
        kalshi_book_imbalance=(None if missing_extras else (0.5 if outcome else -0.5)),
        external_consensus_mid=(None if missing_extras else (101.0 if outcome else 99.0)),
        brti_vs_external_bps=(None if missing_extras else (5.0 if outcome else -5.0)),
        coinbase_mid=None if missing_extras else 100.0,
        kraken_mid=None if missing_extras else 100.0,
        final_minute_sample_count=None if missing_extras else 1,
        final_minute_progress=None if missing_extras else 1 / 60,
        final_minute_average=None if missing_extras else (101.0 if outcome else 99.0),
        required_remaining_brti_average=(
            None if missing_extras else (99.98 if outcome else 100.02)
        ),
        kalshi_book_age_ms=100.0,
        brti_age_ms=100.0,
        coinbase_age_ms=None if missing_extras else 100.0,
        kraken_age_ms=None if missing_extras else 100.0,
    )


def _small_plan() -> CompletionPlan:
    return CompletionPlan(
        min_train_markets=2,
        validation_markets=2,
        test_markets=2,
        step_markets=2,
        l2_grid=(0.1, 1.0),
        min_executable_decisions=2,
    )


def _dataset(*, missing_extras: bool = False):
    outcomes = {f"KXBTC15M-M{i}": i % 2 for i in range(8)}
    rows = {
        market: _row(i, outcome, missing_extras=missing_extras)
        for i, (market, outcome) in enumerate(outcomes.items())
    }
    ordered = tuple(outcomes)
    return rows, outcomes, ordered


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


def test_oos_models_are_deterministic_and_test_markets_do_not_overlap():
    rows, outcomes, ordered = _dataset()
    plan = _small_plan()

    first = evaluate_oos_models(rows, outcomes, ordered, plan)
    second = evaluate_oos_models(rows, outcomes, ordered, plan)

    assert first == second
    assert first.prediction_digest == second.prediction_digest
    assert len(first.folds) == 2
    assert len(first.full_predictions) == 4
    markets = [prediction.market_ticker for prediction in first.full_predictions]
    assert len(markets) == len(set(markets))
    assert set(first.folds[0].test_market_ids).isdisjoint(first.folds[1].test_market_ids)


def test_first_fold_predictions_do_not_use_first_fold_test_labels():
    rows, outcomes, ordered = _dataset()
    plan = _small_plan()
    original = evaluate_oos_models(rows, outcomes, ordered, plan)

    changed = dict(outcomes)
    for market in original.folds[0].test_market_ids:
        changed[market] = 1 - changed[market]
    modified = evaluate_oos_models(rows, changed, ordered, plan)

    original_first = [
        prediction.predicted_yes
        for prediction in original.full_predictions
        if prediction.fold_index == 0
    ]
    modified_first = [
        prediction.predicted_yes
        for prediction in modified.full_predictions
        if prediction.fold_index == 0
    ]
    assert original_first == modified_first


def test_missing_optional_features_are_training_only_imputed_not_row_dropped():
    rows, outcomes, ordered = _dataset(missing_extras=True)
    evaluation = evaluate_oos_models(rows, outcomes, ordered, _small_plan())

    assert len(evaluation.full_predictions) == 4
    assert [score.stage for score in evaluation.ablations] == [
        stage for stage, _ in ABLATION_FEATURES
    ]
    assert all(score.count == 4 for score in evaluation.ablations)


def _metrics(*, trade_count: int, net_pnl: float = 10.0) -> ResearchMetrics:
    return ResearchMetrics(
        test_brier=0.18,
        baseline_brier=0.20,
        test_log_loss=0.50,
        baseline_log_loss=0.55,
        calibration_error=0.02,
        net_pnl=net_pnl,
        gross_pnl=20.0,
        max_drawdown=3.0,
        trade_count=trade_count,
        profitable_walkforward_windows=7,
        total_walkforward_windows=10,
        latency_stress_net_pnl=5.0,
        cost_stress_net_pnl=4.0,
    )


def test_verdict_distinguishes_insufficient_evidence_promotion_and_rejection():
    accepted = PromotionDecision(True, ())
    rejected = PromotionDecision(False, ("edge fails latency stress",))

    assert classify_research_verdict(
        _metrics(trade_count=499),
        accepted,
        (),
        minimum_executable_decisions=500,
    ) == "insufficient_evidence"
    assert classify_research_verdict(
        _metrics(trade_count=500),
        accepted,
        (),
        minimum_executable_decisions=500,
    ) == "promoted"
    assert classify_research_verdict(
        _metrics(trade_count=500, net_pnl=-1.0),
        rejected,
        (),
        minimum_executable_decisions=500,
    ) == "rejected"
    assert classify_research_verdict(
        _metrics(trade_count=500),
        accepted,
        ("fee_schedule_coverage_incomplete=1",),
        minimum_executable_decisions=500,
    ) == "insufficient_evidence"


def test_completion_plan_digest_is_stable_and_configuration_sensitive():
    first = CompletionPlan()
    second = CompletionPlan()
    changed = CompletionPlan(minimum_net_edge=0.02)

    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert first.digest != changed.digest
    assert first.order_quantity == Decimal("1")


def test_completion_report_uses_existing_content_addressed_archive(tmp_path):
    report = ResearchCompletionReport(
        mode="research_only",
        report_kind="research_complete_v1",
        series_ticker="KXBTC15M",
        order_placement=False,
        plan_digest="b" * 64,
        event_count=1,
        events_digest="a" * 64,
        markets=("KXBTC15M-M1",),
        settled_market_count=1,
        horizon_eligible_market_count=0,
        audit=_audit(),
        model_spec_digest="c" * 64,
        verdict="insufficient_evidence",
        evidence_deficits=("horizon_eligible_markets=0 required_at_least=140",),
        promotion_reasons=(),
        ablations=(),
        folds=(),
        prediction_digest=None,
        selections=(),
        metrics=None,
        economics=None,
    )
    archive = ExperimentReportArchive(tmp_path / "experiments")

    first = archive.publish(report)
    second = archive.publish(report)

    assert first == second
    assert first.digest == research_report_digest(report)
    payload = archive.read_payload(first.digest)
    assert payload["report_kind"] == "research_complete_v1"
    assert payload["verdict"] == "insufficient_evidence"
    assert payload["order_placement"] is False
