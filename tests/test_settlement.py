from kalshi_research.math.settlement import (
    SettlementState,
    bm_probability_final_average_above_target,
    resolves_yes,
)


def test_documented_average_resolution_rule():
    samples = [100.004] * 60
    assert resolves_yes(samples, 100.00)


def test_partial_settlement_required_average():
    state = SettlementState(target=100.0, known_samples=tuple([99.0] * 50))
    assert state.remaining == 10
    assert state.required_future_average == 105.0


def test_probability_collapses_when_complete():
    yes_state = SettlementState(target=100.0, known_samples=tuple([101.0] * 60))
    no_state = SettlementState(target=100.0, known_samples=tuple([99.0] * 60))
    assert bm_probability_final_average_above_target(yes_state, 101, 1) == 1.0
    assert bm_probability_final_average_above_target(no_state, 99, 1) == 0.0


def test_more_favorable_current_index_means_higher_probability():
    state = SettlementState(target=100.0, known_samples=tuple([100.0] * 50))
    low = bm_probability_final_average_above_target(state, 99.0, 0.5)
    high = bm_probability_final_average_above_target(state, 101.0, 0.5)
    assert high > low
