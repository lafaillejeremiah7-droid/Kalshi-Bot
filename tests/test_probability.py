from kalshi_research.math.probability import (
    CostAdjustedEdge,
    brier_score,
    diffusion_probability_yes,
    expected_calibration_error,
    pair_arbitrage_edge,
)


def test_diffusion_probability_monotonic():
    assert diffusion_probability_yes(101, 100, 0.001, 60) > 0.5
    assert diffusion_probability_yes(99, 100, 0.001, 60) < 0.5


def test_brier_perfect_is_zero():
    assert brier_score([0, 1], [0, 1]) == 0


def test_calibration_error_perfect_grouped_example():
    assert expected_calibration_error([0, 1], [0, 1], bins=2) == 0


def test_pair_arbitrage_after_costs():
    assert round(pair_arbitrage_edge(0.47, 0.48, 0.01), 10) == 0.04


def test_cost_adjusted_edge():
    edge = CostAdjustedEdge(0.70, 0.62, fees=0.01, slippage=0.005, latency_penalty=0.005)
    assert round(edge.net_edge, 4) == 0.06
