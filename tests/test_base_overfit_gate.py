from xau_company.adaptive_research import AdaptiveStrategyResearchAgent
from xau_company.research import Candidate, CandidateScore


def _score(index: int, strong: bool) -> CandidateScore:
    candidate = Candidate("trend", (5 + (index % 20), 50 + index, 0.0001))
    if strong:
        return CandidateScore(
            candidate=candidate,
            train_hit_rate=0.66,
            valid_hit_rate=0.64,
            trades=180,
            score=0.85,
            walk_forward_hit_rate=0.64,
            walk_forward_std=0.03,
            expectancy=0.001,
            profit_factor=1.60,
            folds=4,
            regime_scores={"trend_up": 0.66},
            regime_trades={"trend_up": 80},
            avg_r_multiple=0.30,
            max_drawdown_r=3.0,
            max_loss_streak=3,
        )
    return CandidateScore(
        candidate=candidate,
        train_hit_rate=0.72,
        valid_hit_rate=0.42,
        trades=180,
        score=0.84,
        walk_forward_hit_rate=0.42,
        walk_forward_std=0.22,
        expectancy=-0.001,
        profit_factor=0.90,
        folds=4,
        regime_scores={"trend_up": 0.42},
        regime_trades={"trend_up": 80},
        avg_r_multiple=-0.10,
        max_drawdown_r=14.0,
        max_loss_streak=10,
    )


def test_full_base_batch_is_audited_and_overfit_variants_are_excluded(tmp_path):
    lab = AdaptiveStrategyResearchAgent(
        max_candidates=2_000,
        catalog_size=600,
        enable_evolution=False,
        strategy_library_path=str(tmp_path / "evolved.json"),
        enable_invention=False,
        invention_library_path=str(tmp_path / "invented.json"),
    )
    batch = [_score(i, strong=(i % 2 == 0)) for i in range(2_000)]

    lab._refresh_live_catalog(batch, seed_tested_trials=2_000)

    assert lab.last_seed_audited == 2_000
    assert lab.last_seed_overfit_rejected == 1_000
    assert lab.last_seed_live_eligible == 1_000
    assert 0 < len(lab.catalog) <= 600
    assert all(result.profit_factor >= 1.15 for result in lab.catalog)
    assert all(result.avg_r_multiple >= 0.05 for result in lab.catalog)


def test_base_strategy_cannot_enter_live_catalog_without_explicit_overfit_pass(tmp_path):
    lab = AdaptiveStrategyResearchAgent(
        max_candidates=1_000,
        enable_evolution=False,
        strategy_library_path=str(tmp_path / "evolved.json"),
        enable_invention=False,
        invention_library_path=str(tmp_path / "invented.json"),
    )
    weak = _score(1, strong=False)

    lab._refresh_live_catalog([weak], seed_tested_trials=20_000)

    assert lab.last_seed_audited == 1
    assert lab.last_seed_overfit_rejected == 1
    assert lab.last_seed_live_eligible == 0
    assert lab.catalog == []
    assert lab.top == []
