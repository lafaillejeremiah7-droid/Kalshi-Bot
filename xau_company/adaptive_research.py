from __future__ import annotations

import numpy as np
import pandas as pd

from .overfit import OverfitAuditor
from .research import Candidate, CandidateScore, StrategyResearchAgent
from .strategy_evolution import StrategyEvolutionAgent


class AdaptiveStrategyResearchAgent(StrategyResearchAgent):
    """Research the fixed seed universe plus a persistent, growing strategy library."""

    def __init__(
        self,
        *args,
        enable_evolution: bool = True,
        strategy_library_path: str = "data/discovered_strategies.json",
        discoveries_per_cycle: int = 250,
        discovery_library_size: int = 5_000,
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
        self.evolution = StrategyEvolutionAgent(
            library_path=strategy_library_path,
            discoveries_per_cycle=discoveries_per_cycle,
            max_library_size=discovery_library_size,
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
            min_folds=max(3, self.min_walk_forward_folds),
        )
        self.last_discovered = 0
        self.last_promoted = 0
        self.last_quarantined = 0
        self.last_experimental_catalog_size = 0
        self.dynamic_library_size = self.evolution.size()

    def candidates(self):
        # The original 27k+ universe remains the stable seed set.
        yield from super().candidates()

        # Then append persistent discoveries. The universe therefore grows over
        # time rather than being hard-coded to the seed grids forever.
        seen: set[Candidate] = set()
        for family, params in self.evolution.candidate_specs():
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
        if candidate.family != "ensemble":
            return super()._signal(df, candidate, cache)

        cache = {} if cache is None else cache
        try:
            family_a, params_a, family_b, params_b, mode = candidate.params
        except (TypeError, ValueError):
            return pd.Series(0, index=df.index, dtype="int8")

        if family_a == "ensemble" or family_b == "ensemble":
            return pd.Series(0, index=df.index, dtype="int8")

        left = Candidate(str(family_a), tuple(params_a))
        right = Candidate(str(family_b), tuple(params_b))
        a = super()._signal(df, left, cache)
        b = super()._signal(df, right, cache)
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
        promoted = self.evolution.promoted_keys()
        eligible = [
            result
            for result in research_catalog
            if result.candidate.family != "ensemble"
            or self.evolution.spec_key(result.candidate.family, result.candidate.params) in promoted
        ]
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
        # Base research may evaluate experimental entries, but the resulting
        # catalog is treated as a research-only staging area until audit completes.
        super().run(df)
        research_catalog = list(self.catalog)
        self.last_experimental_catalog_size = sum(
            result.candidate.family == "ensemble" for result in research_catalog
        )

        if not self.enable_evolution:
            self.last_discovered = 0
            self.last_promoted = 0
            self.last_quarantined = 0
            self.dynamic_library_size = self.evolution.size()
            # Evolution disabled means no dynamically discovered strategy may leak
            # into live selection even if an old library file exists.
            seed_only = [r for r in research_catalog if r.candidate.family != "ensemble"]
            self.catalog = self._build_catalog(seed_only)
            self.top = self.catalog[:25]
            return self.top

        self.last_promoted, self.last_quarantined = self.evolution.audit_promotions(
            research_catalog,
            self.overfit_auditor,
            tested_trials=max(1, self.last_evaluated),
        )

        # Rebuild the Boss-visible catalog after audit. EXPERIMENTAL and
        # QUARANTINED strategies stay available to research but cannot trade.
        self._refresh_live_catalog(research_catalog)

        # New experiments are generated only from the currently live-eligible
        # robust catalog and are not evaluated until a later research cycle.
        self.last_discovered = self.evolution.propose(self.catalog)
        self.dynamic_library_size = self.evolution.size()
        return self.top
