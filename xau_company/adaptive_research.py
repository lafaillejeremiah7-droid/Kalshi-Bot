from __future__ import annotations

from typing import Callable

import pandas as pd

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
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.enable_evolution = bool(enable_evolution)
        self.evolution = StrategyEvolutionAgent(
            library_path=strategy_library_path,
            discoveries_per_cycle=discoveries_per_cycle,
            max_library_size=discovery_library_size,
        )
        self.last_discovered = 0
        self.last_promoted = 0
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

    def run(self, df: pd.DataFrame) -> list[CandidateScore]:
        top = super().run(df)
        if not self.enable_evolution:
            self.last_discovered = 0
            self.last_promoted = 0
            self.dynamic_library_size = self.evolution.size()
            return top

        # Promotion is based only on candidates that already survived the full
        # lifecycle/walk-forward research pipeline in this run.
        self.last_promoted = self.evolution.mark_promoted(self.catalog)

        # New experiments are generated after scoring, so they cannot be used
        # immediately. They enter the candidate universe on the next research
        # cycle and must earn their way into the catalog.
        self.last_discovered = self.evolution.propose(self.catalog)
        self.dynamic_library_size = self.evolution.size()
        return top
