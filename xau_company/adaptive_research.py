from __future__ import annotations

import numpy as np
import pandas as pd

from .overfit import OverfitAuditor
from .research import Candidate, CandidateScore, StrategyResearchAgent
from .strategy_evolution import StrategyEvolutionAgent
from .strategy_invention import StrategyInventionAgent


class AdaptiveStrategyResearchAgent(StrategyResearchAgent):
    """Research seeds plus audited evolved and invented strategy libraries."""

    def __init__(
        self,
        *args,
        enable_evolution: bool = True,
        strategy_library_path: str = "data/discovered_strategies.json",
        discoveries_per_cycle: int = 250,
        discovery_library_size: int = 5_000,
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
        self.enable_evolution = bool(enable_evolution)
        self.enable_invention = bool(enable_invention)
        self.evolution = StrategyEvolutionAgent(
            library_path=strategy_library_path,
            discoveries_per_cycle=discoveries_per_cycle,
            max_library_size=discovery_library_size,
        )
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
        self.last_discovered = 0
        self.last_promoted = 0
        self.last_quarantined = 0
        self.last_experimental_catalog_size = 0
        self.last_invented_catalog_size = 0
        self.last_invented_families = 0
        self.last_invented_variants = 0
        self.last_invention_promoted = 0
        self.last_invention_quarantined = 0
        self.last_lifetime_trials = self.evolution.tested_trials_lifetime()
        self.dynamic_library_size = self.evolution.size()
        self.invention_library_size = self.invention.size()
        self.invention_family_count = self.invention.family_count()
        self.invention_promoted_family_count = self.invention.promoted_family_count()

    def candidates(self):
        yield from super().candidates()
        seen: set[Candidate] = set()
        dynamic_sources = (
            self.evolution.candidate_specs() if self.enable_evolution else [],
            self.invention.candidate_specs() if self.enable_invention else [],
        )
        for specs in dynamic_sources:
            for family, params in specs:
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

        if candidate.family != "ensemble":
            return super()._signal(df, candidate, cache)

        cache = {} if cache is None else cache
        try:
            family_a, params_a, family_b, params_b, mode = candidate.params
        except (TypeError, ValueError):
            return pd.Series(0, index=df.index, dtype="int8")

        # Evolved signals remain one level deep. Invented parents are allowed
        # because they are finite primitive formulas; nested ensembles are not.
        if family_a == "ensemble" or family_b == "ensemble":
            return pd.Series(0, index=df.index, dtype="int8")

        left = Candidate(str(family_a), tuple(params_a))
        right = Candidate(str(family_b), tuple(params_b))
        a = self.invention.signal(df, left.params, cache) if left.family == "invented" else super()._signal(df, left, cache)
        b = self.invention.signal(df, right.params, cache) if right.family == "invented" else super()._signal(df, right, cache)
        out = pd.Series(0, index=df.index, dtype="int8")

        if mode == "confirm":
            out[(a == 1) & (b == 1)] = 1
            out[(a == -1) & (b == -1)] = -1
        elif mode == "primary_filter":
            out[(a == 1) & (b != -1)] = 1
            out[(a == -1) & (b != 1)] = -1
        elif mode == "consensus_or":
            out[((a == 1) & (b != -1)) | ((b == 1) & (a != -1))] = 1
            out[((a == -1) & (b != 1)) | ((b == -1) & (a != 1))] = -1
        return out

    def _refresh_live_catalog(self, research_catalog: list[CandidateScore]) -> None:
        evolved_promoted = self.evolution.promoted_keys() if self.enable_evolution else set()
        invented_promoted = self.invention.promoted_keys() if self.enable_invention else set()
        eligible: list[CandidateScore] = []

        for result in research_catalog:
            family = result.candidate.family
            key = StrategyEvolutionAgent.spec_key(family, result.candidate.params)
            if family in self.HORIZONS:
                eligible.append(result)
            elif family == "ensemble" and key in evolved_promoted:
                eligible.append(result)
            elif family == "invented" and key in invented_promoted:
                eligible.append(result)

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
        super().run(df)
        research_catalog = list(self.catalog)
        self.last_experimental_catalog_size = sum(result.candidate.family == "ensemble" for result in research_catalog)
        self.last_invented_catalog_size = sum(result.candidate.family == "invented" for result in research_catalog)

        if not self.enable_evolution and not self.enable_invention:
            self.last_discovered = self.last_promoted = self.last_quarantined = 0
            self.last_invented_families = self.last_invented_variants = 0
            self.last_invention_promoted = self.last_invention_quarantined = 0
            self.dynamic_library_size = self.evolution.size()
            self.invention_library_size = self.invention.size()
            self.invention_family_count = self.invention.family_count()
            self.invention_promoted_family_count = self.invention.promoted_family_count()
            seed_only = [r for r in research_catalog if r.candidate.family in self.HORIZONS]
            self.catalog = self._build_catalog(seed_only)
            self.top = self.catalog[:25]
            return self.top

        # One global lifetime trial ledger feeds both dynamic sources so adding
        # Invention cannot reset the multiple-testing penalty.
        self.last_lifetime_trials = self.evolution.increment_tested_trials(self.last_evaluated)

        if self.enable_evolution:
            self.last_promoted, self.last_quarantined = self.evolution.audit_promotions(
                research_catalog, self.overfit_auditor, tested_trials=max(1, self.last_lifetime_trials)
            )
        else:
            self.last_promoted = self.last_quarantined = 0

        if self.enable_invention:
            self.last_invention_promoted, self.last_invention_quarantined = self.invention.audit_promotions(
                research_catalog, self.overfit_auditor, tested_trials=max(1, self.last_lifetime_trials)
            )
        else:
            self.last_invention_promoted = self.last_invention_quarantined = 0

        # Only seed strategies and explicitly promoted dynamic variants may reach Selector.
        self._refresh_live_catalog(research_catalog)

        self.last_discovered = self.evolution.propose(self.catalog) if self.enable_evolution else 0
        if self.enable_invention:
            self.last_invented_families, self.last_invented_variants = self.invention.invent()
        else:
            self.last_invented_families = self.last_invented_variants = 0

        self.dynamic_library_size = self.evolution.size()
        self.invention_library_size = self.invention.size()
        self.invention_family_count = self.invention.family_count()
        self.invention_promoted_family_count = self.invention.promoted_family_count()
        return self.top
