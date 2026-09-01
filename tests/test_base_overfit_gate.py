from xau_company.adaptive_research import AdaptiveStrategyResearchAgent
from xau_company.canonical_strategies import STRATEGIES
from xau_company.research import Candidate, CandidateScore


def _score(index: int, strong: bool) -> CandidateScore:
    candidate = Candidate(STRATEGIES[index].strategy_id)
    common = dict(candidate=candidate, trades=180, folds=4, regime_trades={"trend_up": 80})
    if strong:
        return CandidateScore(
            train_hit_rate=0.66, valid_hit_rate=0.64, score=0.85,
            walk_forward_hit_rate=0.64, walk_forward_std=0.03, expectancy=0.001,
            profit_factor=1.60, regime_scores={"trend_up": 0.66}, avg_r_multiple=0.30,
            max_drawdown_r=3.0, max_loss_streak=3, **common,
        )
    return CandidateScore(
        train_hit_rate=0.72, valid_hit_rate=0.42, score=0.84,
        walk_forward_hit_rate=0.42, walk_forward_std=0.22, expectancy=-0.001,
        profit_factor=0.90, regime_scores={"trend_up": 0.42}, avg_r_multiple=-0.10,
        max_drawdown_r=14.0, max_loss_streak=10, **common,
    )


def test_full_canonical_batch_is_audited_and_overfit_models_are_excluded():
    lab = AdaptiveStrategyResearchAgent(max_candidates=437, catalog_size=109)
    batch = [_score(i, strong=(i % 2 == 0)) for i in range(437)]
    lab._refresh_live_catalog(batch, seed_tested_trials=437)
    assert lab.last_seed_audited == 437
    assert lab.last_seed_overfit_rejected == 218
    assert lab.last_seed_live_eligible == 219
    assert 0 < len(lab.catalog) <= 109
    assert all(result.profit_factor >= 1.15 for result in lab.catalog)
    assert all(result.avg_r_multiple >= 0.05 for result in lab.catalog)


def test_canonical_strategy_cannot_enter_live_catalog_without_explicit_overfit_pass():
    lab = AdaptiveStrategyResearchAgent(max_candidates=437)
    weak = _score(1, strong=False)
    lab._refresh_live_catalog([weak], seed_tested_trials=437)
    assert lab.last_seed_audited == 1
    assert lab.last_seed_overfit_rejected == 1
    assert lab.last_seed_live_eligible == 0
    assert lab.catalog == []
    assert lab.top == []
