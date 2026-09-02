from __future__ import annotations

from decimal import Decimal

from kalshi_research.domain.events import (
    FeeScheduleEvent,
    MarketEvent,
    SettlementEvent,
    Source,
)
from kalshi_research.research.complete import (
    CompletionPlan,
    OOSPrediction,
    _build_fee_aware_intents,
)
from kalshi_research.research.completion_entrypoint import run_research_completion_events
from kalshi_research.research.materializer import ModelFeatureRow


def _decision_row() -> ModelFeatureRow:
    return ModelFeatureRow(
        decision_recv_ts_ns=2_000_000_000,
        market_ticker="KXBTC15M-M1",
        probability_ready=True,
        baseline_ready=True,
        seconds_to_close=60.0,
        target_price=100.0,
        brti=100.0,
        brti_log_distance_to_target=0.0,
        brti_vol_per_sqrt_second=0.001,
        normalized_distance_to_target=0.0,
        kalshi_yes_bid=0.49,
        kalshi_yes_ask=0.51,
        kalshi_yes_mid=0.50,
        kalshi_spread=0.02,
        kalshi_book_imbalance=0.0,
        external_consensus_mid=100.0,
        brti_vs_external_bps=0.0,
        coinbase_mid=100.0,
        kraken_mid=100.0,
        final_minute_sample_count=1,
        final_minute_progress=1 / 60,
        final_minute_average=100.0,
        required_remaining_brti_average=100.0,
        kalshi_book_age_ms=100.0,
        brti_age_ms=100.0,
        coinbase_age_ms=100.0,
        kraken_age_ms=100.0,
    )


def _fee_event() -> FeeScheduleEvent:
    return FeeScheduleEvent(
        source=Source.KALSHI,
        event_ts_ns=1_000_000_000,
        recv_ts_ns=1_000_000_000,
        series_ticker="KXBTC15M",
        fee_type="quadratic",
        fee_multiplier=Decimal("1"),
        effective_ts_ns=1_000_000_000,
        historical=False,
    )


def test_sparse_valid_settled_evidence_returns_insufficient_evidence_not_error():
    market = MarketEvent(
        source=Source.KALSHI,
        event_ts_ns=100,
        recv_ts_ns=100,
        market_ticker="KXBTC15M-M1",
        event_ticker="KXBTC15M-E1",
        series_ticker="KXBTC15M",
        target_price=Decimal("100"),
        open_ts_ns=100,
        close_ts_ns=200,
        status="settled",
    )
    settlement = SettlementEvent(
        source=Source.KALSHI,
        event_ts_ns=300,
        recv_ts_ns=300,
        market_ticker="KXBTC15M-M1",
        target_price=Decimal("100"),
        final_value=Decimal("101"),
        result="yes",
    )

    report = run_research_completion_events((market, settlement))

    assert report.verdict == "insufficient_evidence"
    assert report.order_placement is False
    assert report.settled_market_count == 1
    assert report.horizon_eligible_market_count == 0
    assert report.evidence_deficits


def test_fee_aware_selector_submits_strong_yes_edge_but_no_trades_marginal_edge():
    row = _decision_row()
    rows = {row.market_ticker: row}
    plan = CompletionPlan(
        min_train_markets=2,
        validation_markets=2,
        test_markets=2,
        step_markets=2,
        min_executable_decisions=2,
    )

    strong = OOSPrediction(
        fold_index=0,
        market_ticker=row.market_ticker,
        decision_recv_ts_ns=row.decision_recv_ts_ns,
        predicted_yes=0.80,
        market_yes_mid=0.50,
        outcome=1,
        selected_l2=1.0,
    )
    intents, selections, deficits = _build_fee_aware_intents(
        (strong,), rows, (_fee_event(),), plan, "KXBTC15M"
    )
    assert deficits == ()
    assert len(intents) == 1
    assert intents[0].outcome_side == "yes"
    assert selections[0].selected_side == "yes"
    assert selections[0].yes_net_edge is not None
    assert selections[0].yes_net_edge > plan.minimum_net_edge

    marginal = OOSPrediction(
        fold_index=0,
        market_ticker=row.market_ticker,
        decision_recv_ts_ns=row.decision_recv_ts_ns,
        predicted_yes=0.52,
        market_yes_mid=0.50,
        outcome=1,
        selected_l2=1.0,
    )
    intents, selections, deficits = _build_fee_aware_intents(
        (marginal,), rows, (_fee_event(),), plan, "KXBTC15M"
    )
    assert deficits == ()
    assert intents == ()
    assert selections[0].selected_side is None
    assert selections[0].reason == "edge_below_threshold"
