from kalshi_research.research.acceptance import ResearchMetrics, evaluate_for_probability_stage


def test_rejects_weak_edge():
    metrics = ResearchMetrics(
        test_brier=0.249,
        baseline_brier=0.25,
        test_log_loss=0.692,
        baseline_log_loss=0.693,
        calibration_error=0.04,
        net_pnl=-1,
        gross_pnl=10,
        max_drawdown=5,
        trade_count=100,
        profitable_walkforward_windows=1,
        total_walkforward_windows=4,
        latency_stress_net_pnl=-2,
        cost_stress_net_pnl=-3,
    )
    decision = evaluate_for_probability_stage(metrics)
    assert not decision.accepted
    assert len(decision.reasons) >= 5
