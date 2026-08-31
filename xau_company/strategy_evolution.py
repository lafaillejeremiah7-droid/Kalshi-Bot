from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .overfit import OverfitAuditor

try:  # Linux/Unix production hosts.
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback remains atomic per write.
    fcntl = None


class StrategyEvolutionAgent:
    """Build, persist, audit and safely synchronize evolved strategy experiments."""

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

    def _path(self) -> Path:
        return Path(self.library_path)

    def _meta_path(self) -> Path:
        path = self._path()
        return path.with_name(path.name + ".meta.json")

    @contextmanager
    def _exclusive_lock(self):
        if self.library_path == ":memory:":
            yield
            return
        lock_path = self._path().with_name(self._path().name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> list[dict[str, Any]]:
        if self.library_path == ":memory:":
            return list(getattr(self, "_memory", []))
        path = self._path()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _read(self) -> list[dict[str, Any]]:
        with self._exclusive_lock():
            return self._read_unlocked()

    def _read_meta_unlocked(self) -> dict[str, Any]:
        if self.library_path == ":memory:":
            return dict(getattr(self, "_meta_memory", {}))
        path = self._meta_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_meta_unlocked(self, meta: dict[str, Any]) -> None:
        if self.library_path == ":memory:":
            self._meta_memory = dict(meta)
            return
        path = self._meta_path()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def increment_tested_trials(self, trials: int) -> int:
        """Persist cumulative search effort so multiplicity pressure grows over time."""
        add = max(0, int(trials))
        with self._exclusive_lock():
            meta = self._read_meta_unlocked()
            total = max(0, int(meta.get("tested_trials_lifetime", 0))) + add
            meta["tested_trials_lifetime"] = total
            meta["last_research_at"] = datetime.now(timezone.utc).isoformat()
            self._write_meta_unlocked(meta)
            return total

    def tested_trials_lifetime(self) -> int:
        with self._exclusive_lock():
            return max(0, int(self._read_meta_unlocked().get("tested_trials_lifetime", 0)))

    def _bounded_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # PROMOTED and QUARANTINED entries are durable research history. The size
        # limit applies to disposable experiments; protected entries are never
        # silently evicted merely because the experimental queue is full.
        protected = [row for row in rows if row.get("status") in {"PROMOTED", "QUARANTINED"}]
        experiments = [row for row in rows if row.get("status") not in {"PROMOTED", "QUARANTINED"}]
        remaining = max(0, self.max_library_size - len(protected))
        return protected + (experiments[-remaining:] if remaining > 0 else [])

    def _write_unlocked(self, rows: list[dict[str, Any]]) -> None:
        rows = self._bounded_rows(rows)
        if self.library_path == ":memory:":
            self._memory = rows
            return
        path = self._path()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _write(self, rows: list[dict[str, Any]]) -> None:
        with self._exclusive_lock():
            self._write_unlocked(rows)

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

    def promoted_keys(self) -> set[str]:
        return {
            self.spec_key(str(row.get("family", "")), row.get("params", []))
            for row in self._read()
            if row.get("status") == "PROMOTED"
        }

    @staticmethod
    def _parent_payload(score: Any) -> dict[str, Any]:
        candidate = score.candidate
        oos_trades = sum(max(0, int(n)) for n in getattr(score, "regime_trades", {}).values())
        return {
            "family": candidate.family,
            "params": StrategyEvolutionAgent._jsonable(candidate.params),
            "research_score": round(float(score.score), 6),
            "oos_hit_rate": round(float(score.walk_forward_hit_rate or score.valid_hit_rate), 6),
            "oos_trades": oos_trades if oos_trades > 0 else int(score.trades),
            "profit_factor": round(float(score.profit_factor), 6),
            "avg_r": round(float(score.avg_r_multiple), 6),
        }

    def propose(self, scores: Iterable[Any]) -> int:
        """Continuously rotate experiments without ever evicting protected history."""
        if self.discoveries_per_cycle <= 0:
            return 0

        with self._exclusive_lock():
            rows = self._bounded_rows(self._read_unlocked())
            protected = [row for row in rows if row.get("status") in {"PROMOTED", "QUARANTINED"}]
            experiments = [row for row in rows if row.get("status") not in {"PROMOTED", "QUARANTINED"}]
            experiment_capacity = max(0, self.max_library_size - len(protected))
            if experiment_capacity <= 0:
                return 0

            limit = min(self.discoveries_per_cycle, experiment_capacity)
            # Preserve the keys of entries rotated out during this cycle so they
            # are not immediately regenerated from the same parent pair.
            prior_keys = {
                self.spec_key(str(row.get("family", "")), row.get("params", []))
                for row in rows
            }
            slots_needed = max(0, limit - max(0, experiment_capacity - len(experiments)))
            if slots_needed:
                experiments = experiments[min(slots_needed, len(experiments)) :]
            rows = protected + experiments
            existing = set(prior_keys)

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
                        if added >= limit:
                            self._write_unlocked(rows)
                            return added
            self._write_unlocked(rows)
            return added

    def audit_promotions(
        self,
        catalog: Iterable[Any],
        auditor: OverfitAuditor,
        tested_trials: int,
    ) -> tuple[int, int]:
        """Promote only audited entries and quarantine degraded promotions."""
        with self._exclusive_lock():
            rows = self._read_unlocked()
            if not rows:
                return 0, 0

            scores = {
                self.spec_key(result.candidate.family, result.candidate.params): result
                for result in catalog
                if result.candidate.family == "ensemble"
            }
            newly_promoted = 0
            newly_quarantined = 0
            touched = False
            now = datetime.now(timezone.utc).isoformat()

            for row in rows:
                key = self.spec_key(str(row.get("family", "")), row.get("params", []))
                result = scores.get(key)
                if result is None:
                    continue

                audit = auditor.audit(result, tested_trials)
                previous = str(row.get("status", "EXPERIMENTAL"))
                next_status = "PROMOTED" if audit.passed else (
                    "QUARANTINED" if previous == "PROMOTED" else "EXPERIMENTAL"
                )
                if next_status == "PROMOTED" and previous != "PROMOTED":
                    newly_promoted += 1
                if next_status == "QUARANTINED" and previous == "PROMOTED":
                    newly_quarantined += 1

                row["status"] = next_status
                row["last_audited_at"] = now
                row["last_score"] = round(float(result.score), 6)
                row["last_oos_hit_rate"] = round(float(result.walk_forward_hit_rate or result.valid_hit_rate), 6)
                row["last_oos_trades"] = auditor.oos_trade_count(result)
                row["last_profit_factor"] = round(float(result.profit_factor), 6)
                row["last_avg_r"] = round(float(result.avg_r_multiple), 6)
                row["overfit_adjusted_score"] = round(float(audit.adjusted_score), 6)
                row["overfit_multiplicity_penalty"] = round(float(audit.multiplicity_penalty), 6)
                row["overfit_tested_trials"] = int(audit.tested_trials)
                row["overfit_passed"] = bool(audit.passed)
                row["overfit_reasons"] = list(audit.reasons)
                if audit.passed:
                    row["last_promoted_at"] = now
                elif previous == "PROMOTED":
                    row["last_quarantined_at"] = now
                touched = True

            if touched:
                self._write_unlocked(rows)
            return newly_promoted, newly_quarantined

    def size(self) -> int:
        return len(self._read())
