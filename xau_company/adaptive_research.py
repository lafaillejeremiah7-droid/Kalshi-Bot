from __future__ import annotations

import numpy as np
import pandas as pd

from .canonical_strategies import BY_ID
from .overfit import OverfitAuditor
from .research import CandidateScore, StrategyResearchAgent


class AdaptiveStrategyResearchAgent(StrategyResearchAgent):
    """Audited research lab for the survivor-only canonical strategy library."""

    def __init__(
        self,
        *args,
        overfit_min_adjusted_score: float = 0.60,
        overfit_min_profit_factor: float = 1.15,
        overfit_min_avg_r: float = 0.05,
        overfit_min_trades: int = 40,
        overfit_max_walk_forward_std: float = 0.12,
        overfit_max_train_valid_gap: float = 0.15,
        overfit_max_drawdown_r: float = 10.0,
        overfit_max_loss_streak: int = 7,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.overfit_auditor = OverfitAuditor(
            min_adjusted_score=overfit_min_adjusted_score,
            min_profit_factor=overfit_min_profit_factor,
            min_avg_r=overfit_min_avg_r,
            min_trades=overfit_min_trades,
            max_walk_forward_std=overfit_max_walk_forward_std,
            max_train_valid_gap=overfit_max_train_valid_gap,
            max_drawdown_r=overfit_max_drawdown_r,
            max_loss_streak=overfit_max_loss_streak,
            min_folds=self.min_walk_forward_folds,
        )
        self.last_seed_audited = 0
        self.last_seed_overfit_rejected = 0
        self.last_seed_live_eligible = 0

    def _refresh_live_catalog(
        self,
        research_catalog: list[CandidateScore],
        seed_tested_trials: int | None = None,
    ) -> None:
        eligible: list[CandidateScore] = []
        seed_audited = 0
        seed_rejected = 0
        seed_eligible = 0
        # The discovery universe is fixed at 437 distinct methodologies. Re-running
        # the same survivors later does not create additional multiple-testing trials.
        tested_trials = 437 if seed_tested_trials is None else max(1, int(seed_tested_trials))

        for result in research_catalog:
            if result.candidate.strategy_id not in self.HORIZONS:
                continue
            seed_audited += 1
            audit = self.overfit_auditor.audit(result, tested_trials=tested_trials)
            if audit.passed:
                eligible.append(result)
                seed_eligible += 1
            else:
                seed_rejected += 1

        self.last_seed_audited = seed_audited
        self.last_seed_overfit_rejected = seed_rejected
        self.last_seed_live_eligible = seed_eligible
        self.catalog = self._build_catalog(eligible)
        self.top = self.catalog[:25]

        strategy_buckets: dict[str, list[float]] = {}
        category_buckets: dict[str, list[float]] = {}
        for result in self.catalog:
            sid = result.candidate.strategy_id
            strategy_buckets.setdefault(sid, []).append(result.score)
            definition = BY_ID.get(sid)
            if definition is not None:
                category_buckets.setdefault(definition.category, []).append(result.score)
        self.family_quality = {
            sid: float(np.clip(np.mean(vals), 0.35, 0.90))
            for sid, vals in strategy_buckets.items()
        }
        self.category_quality = {
            category: float(np.clip(np.mean(vals), 0.35, 0.90))
            for category, vals in category_buckets.items()
        }

    def run(self, df: pd.DataFrame) -> list[CandidateScore]:
        selected = self._balanced_candidates()
        self.last_evaluated = len(selected)
        regimes = self._historical_regimes(df)

        research_catalog: list[CandidateScore] = []
        for candidate in selected:
            result = self._evaluate(df, candidate, None, regimes)
            if result is not None:
                research_catalog.append(result)
        research_catalog.sort(key=lambda x: x.score, reverse=True)

        self._refresh_live_catalog(research_catalog, seed_tested_trials=437)
        return self.top
