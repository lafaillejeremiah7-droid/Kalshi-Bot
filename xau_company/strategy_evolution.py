from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class StrategyEvolutionAgent:
    """Build and persist new experimental strategy structures from validated parents.

    The original parameter grids remain the seed universe. This agent expands the
    search space by recombining strong, structurally different parent strategies
    into ensembles. Newly created entries are EXPERIMENTAL: merely being in this
    library never authorizes live use. The research/backtest pipeline must score
    them out of sample before they can be promoted into the robust catalog.
    """

    name = "Strategy Discovery & Evolution Bot"
    MODES = ("confirm", "primary_filter", "consensus_or")

    def __init__(
        self,
        library_path: str = "data/discovered_strategies.json",
        discoveries_per_cycle: int = 250,
        max_library_size: int = 5_000,
    ) -> None:
        self.library_path = library_path
        self.discoveries_per_cycle = max(0, int(discoveries_per_cycle))
        self.max_library_size = max(100, int(max_library_size))
        if library_path != ":memory:":
            Path(library_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, tuple):
            return [StrategyEvolutionAgent._jsonable(x) for x in value]
        if isinstance(value, list):
            return [StrategyEvolutionAgent._jsonable(x) for x in value]
        if isinstance(value, dict):
            return {str(k): StrategyEvolutionAgent._jsonable(v) for k, v in value.items()}
        return value

    @staticmethod
    def _tupleize(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(StrategyEvolutionAgent._tupleize(x) for x in value)
        if isinstance(value, dict):
            return {k: StrategyEvolutionAgent._tupleize(v) for k, v in value.items()}
        return value

    @classmethod
    def spec_key(cls, family: str, params: Any) -> str:
        payload = {"family": family, "params": cls._jsonable(params)}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _read(self) -> list[dict[str, Any]]:
        if self.library_path == ":memory:":
            return getattr(self, "_memory", [])
        path = Path(self.library_path)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _write(self, rows: list[dict[str, Any]]) -> None:
        rows = rows[-self.max_library_size :]
        if self.library_path == ":memory:":
            self._memory = rows
            return
        path = Path(self.library_path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def entries(self) -> list[dict[str, Any]]:
        return self._read()

    def candidate_specs(self) -> list[tuple[str, tuple]]:
        specs: list[tuple[str, tuple]] = []
        for row in self._read():
            family = row.get("family")
            params = row.get("params")
            if isinstance(family, str) and isinstance(params, list):
                specs.append((family, self._tupleize(params)))
        return specs

    @staticmethod
    def _parent_payload(score: Any) -> dict[str, Any]:
        candidate = score.candidate
        return {
            "family": candidate.family,
            "params": StrategyEvolutionAgent._jsonable(candidate.params),
            "research_score": round(float(score.score), 6),
            "oos_hit_rate": round(float(score.walk_forward_hit_rate or score.valid_hit_rate), 6),
            "profit_factor": round(float(score.profit_factor), 6),
            "avg_r": round(float(score.avg_r_multiple), 6),
        }

    def propose(self, scores: Iterable[Any]) -> int:
        """Create novel ensembles from strong non-ensemble parents.

        We intentionally recombine different strategy families. Combining nearly
        identical parameterizations creates many correlated trials without adding
        much structural novelty, which increases multiple-testing risk.
        """
        if self.discoveries_per_cycle <= 0:
            return 0

        rows = self._read()
        existing = {
            self.spec_key(str(row.get("family", "")), row.get("params", []))
            for row in rows
        }
        parents = []
        seen_parents: set[str] = set()
        for score in scores:
            candidate = score.candidate
            if candidate.family == "ensemble":
                continue
            key = self.spec_key(candidate.family, candidate.params)
            if key in seen_parents:
                continue
            seen_parents.add(key)
            parents.append(score)
            if len(parents) >= 120:
                break

        added = 0
        now = datetime.now(timezone.utc).isoformat()
        for i, left in enumerate(parents):
            for right in parents[i + 1 :]:
                a = left.candidate
                b = right.candidate
                if a.family == b.family:
                    continue

                variants = [
                    (a, b, "confirm"),
                    (a, b, "primary_filter"),
                    (b, a, "primary_filter"),
                    (a, b, "consensus_or"),
                ]
                for primary, secondary, mode in variants:
                    params = (
                        primary.family,
                        primary.params,
                        secondary.family,
                        secondary.params,
                        mode,
                    )
                    key = self.spec_key("ensemble", params)
                    if key in existing:
                        continue
                    existing.add(key)
                    rows.append(
                        {
                            "family": "ensemble",
                            "params": self._jsonable(params),
                            "status": "EXPERIMENTAL",
                            "origin": "evolved_from_validated_catalog",
                            "created_at": now,
                            "parents": [self._parent_payload(left), self._parent_payload(right)],
                        }
                    )
                    added += 1
                    if added >= self.discoveries_per_cycle or len(rows) >= self.max_library_size:
                        self._write(rows)
                        return added

        self._write(rows)
        return added

    def mark_promoted(self, catalog: Iterable[Any]) -> int:
        rows = self._read()
        if not rows:
            return 0
        promoted_scores = {
            self.spec_key(result.candidate.family, result.candidate.params): result
            for result in catalog
            if result.candidate.family == "ensemble"
        }
        changed = 0
        for row in rows:
            key = self.spec_key(str(row.get("family", "")), row.get("params", []))
            result = promoted_scores.get(key)
            if result is None:
                continue
            if row.get("status") != "PROMOTED":
                changed += 1
            row["status"] = "PROMOTED"
            row["last_promoted_at"] = datetime.now(timezone.utc).isoformat()
            row["last_score"] = round(float(result.score), 6)
            row["last_oos_hit_rate"] = round(
                float(result.walk_forward_hit_rate or result.valid_hit_rate), 6
            )
            row["last_profit_factor"] = round(float(result.profit_factor), 6)
            row["last_avg_r"] = round(float(result.avg_r_multiple), 6)
        if changed:
            self._write(rows)
        return changed

    def size(self) -> int:
        return len(self._read())
