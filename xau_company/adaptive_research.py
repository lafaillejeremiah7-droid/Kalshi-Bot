from __future__ import annotations

import numpy as np
import pandas as pd

from .overfit import OverfitAuditor
from .research import Candidate, CandidateScore, StrategyResearchAgent
from .strategy_invention import StrategyInventionAgent


class AdaptiveStrategyResearchAgent(StrategyResearchAgent):
    """Research seed strategies plus independently audited invented strategies.

    Strategy Evolution was intentionally removed. Legacy evolution constructor
    arguments and telemetry fields remain as inert compatibility shims so older
    launch/backtest code cannot accidentally re-enable the deleted behavior.
    """

    def __init__(
        self,
        *args,
        enable_evolution: bool | None = None,
        strategy_library_path: str | None = None,
        discoveries_per_cycle: int | None = None,
        discovery_library_size: int | None = None,
        enable_invention: bool = False,
        invention_library_path: str = "data/invented_strategies.json",
        invented_families_per_cycle: int = 6,
        invented_variants_per_family: int = 8,
        invention_library_size: int = 4_000,
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
        # These values are accepted only so old callers keep starting safely.
        # They have no effect and there is no evolution engine attached here.
        _ = (enable_evolution, strategy_library_path, discoveries_per_cycle, discovery_library_size)

        self.enable_invention = bool(enable_invention)
        self.invention = StrategyInventionAgent(
            library_path=invention_library_path,
            families_per_cycle=invented_families_per_cycle,
            variants_per_family=invented_variants_per_family,
            max_library_size=invention_library_size,
        )
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

        self.last_invented_catalog_size = 0
        self.last_invented_families = 0
        self.last_invented_variants = 0
        self.last_invention_promoted = 0
        self.last_invention_quarantined = 0
        self.last_seed_audited = 0
        self.last_seed_overfit_rejected = 0
        self.last_seed_live_eligible = 0
        self.invention_library_size = self.invention.size()
        self.invention_family_count = self.invention.family_count()
        self.invention_promoted_family_count = self.invention.promoted_family_count()

        # Inert compatibility telemetry. Evolution cannot be enabled from these.
        self.last_lifetime_trials = 0
        self.dynamic_library_size = 0
        self.last_discovered = 0
        self.last_promoted = 0
        self.last_quarantined = 0
        self.last_experimental_catalog_size = 0

    def candidates(self):
        yield from super().candidates()
        if not self.enable_invention:
            return
        seen: set[Candidate] = set()
        for family, params in self.invention.candidate_specs():
            candidate = Candidate(family, params)
            if candidate not in seen:
                seen.add(candidate)
                yield candidate

    def _signal(
        self,
        df: pd.DataFrame,
        candidate: Candidate,
        cache: dict[tuple, pd.Series | tuple[pd.Series, pd.Series]] | None = None,
    ) -> pd.Series:
        if candidate.family == "invented":
            return self.invention.signal(df, candidate.params, cache)
        # Evolved/ensemble strategies are deliberately unsupported after the
        # Strategy Evolution bot removal and therefore fail closed to HOLD.
        if candidate.family == "ensemble":
            return pd.Series(0, index=df.index, dtype="int8")
        return super()._signal(df, candidate, cache)

    def _refresh_live_catalog(
        self,
        research_catalog: list[CandidateScore],
        seed_tested_trials: int | None = None,
    ) -> None:
        """Build live catalog only from seed or invented strategies that pass audit."""
        invented_promoted = self.invention.promoted_keys() if self.enable_invention else set()
        eligible: list[CandidateScore] = []
        seed_audited = 0
        seed_rejected = 0
        seed_eligible = 0
        resolved_seed_trials = max(
            1,
            int(seed_tested_trials)
            if seed_tested_trials is not None
            else max(self.last_evaluated, len(research_catalog)),
        )

        for result in research_catalog:
            family = result.candidate.family
            key = self.invention.spec_key(family, result.candidate.params)
            if family in self.HORIZONS:
                seed_audited += 1
                audit = self.overfit_auditor.audit(result, tested_trials=resolved_seed_trials)
                if audit.passed:
                    eligible.append(result)
                    seed_eligible += 1
                else:
                    seed_rejected += 1
            elif family == "invented" and key in invented_promoted:
                eligible.append(result)

        self.last_seed_audited = seed_audited
        self.last_seed_overfit_rejected = seed_rejected
        self.last_seed_live_eligible = seed_eligible
        self.catalog = self._build_catalog(eligible)
        self.top = self.catalog[:25]

        buckets: dict[str, list[float]] = {}
        for result in self.catalog:
            buckets.setdefault(result.candidate.family, []).append(result.score)
        self.family_quality = {
            family: float(np.clip(np.mean(vals[:20]), 0.35, 0.90))
            for family, vals in buckets.items()
        }

    def run(self, df: pd.DataFrame) -> list[CandidateScore]:
        selected = self._balanced_candidates()
        self.last_evaluated = len(selected)
        self.last_lifetime_trials += self.last_evaluated
        cache: dict[tuple, pd.Series | tuple[pd.Series, pd.Series]] = {}
        regimes = self._historical_regimes(df, cache)

        research_catalog: list[CandidateScore] = []
        for candidate in selected:
            result = self._evaluate(df, candidate, cache, regimes)
            if result is not None:
                research_catalog.append(result)
        research_catalog.sort(key=lambda x: x.score, reverse=True)

        self.last_experimental_catalog_size = 0
        self.last_invented_catalog_size = sum(
            result.candidate.family == "invented" for result in research_catalog
        )
        seed_trials = max(1, self.last_evaluated)

        if self.enable_invention:
            self.last_invention_promoted, self.last_invention_quarantined = self.invention.audit_promotions(
                research_catalog,
                self.overfit_auditor,
                tested_trials=max(1, self.last_lifetime_trials),
            )
        else:
            self.last_invention_promoted = self.last_invention_quarantined = 0

        self._refresh_live_catalog(research_catalog, seed_tested_trials=seed_trials)

        if self.enable_invention:
            self.last_invented_families, self.last_invented_variants = self.invention.invent()
        else:
            self.last_invented_families = self.last_invented_variants = 0

        self.invention_library_size = self.invention.size()
        self.invention_family_count = self.invention.family_count()
        self.invention_promoted_family_count = self.invention.promoted_family_count()
        return self.top
