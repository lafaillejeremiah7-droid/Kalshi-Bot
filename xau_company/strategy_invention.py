from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .indicators import atr, donchian, ema, roc, rsi, rolling_zscore
from .overfit import OverfitAuditor
from .strategy_evolution import StrategyEvolutionAgent


class StrategyInventionAgent(StrategyEvolutionAgent):
    """Invent new signal families from primitive market features.

    Evolution combines existing strategy outputs. Invention creates fresh
    indicator/price formulas. Every invented variant remains research-only until
    the shared OverfitAuditor explicitly promotes it.
    """

    name = "Strategy Invention Bot"
    DIRECTIONAL_FEATURES = (
        "ema_gap", "ema_slope", "momentum", "rsi_trend", "rsi_reversion",
        "zscore_trend", "zscore_reversion", "donchian_break", "donchian_fade",
        "candle_impulse", "range_location",
    )
    GATES = ("none", "atr_expansion", "atr_normal", "atr_compression")
    LOGICS = ("all", "majority", "lead_confirm")

    def __init__(
        self,
        library_path: str = "data/invented_strategies.json",
        families_per_cycle: int = 6,
        variants_per_family: int = 8,
        max_library_size: int = 4_000,
    ) -> None:
        super().__init__(library_path=library_path, discoveries_per_cycle=0, max_library_size=max_library_size)
        self.families_per_cycle = max(0, int(families_per_cycle))
        self.variants_per_family = max(1, int(variants_per_family))

    @classmethod
    def family_templates(cls) -> list[tuple[tuple[str, str, str], str, str]]:
        return [
            (features, gate, logic)
            for features in combinations(cls.DIRECTIONAL_FEATURES, 3)
            for gate in cls.GATES
            for logic in cls.LOGICS
        ]

    @staticmethod
    def _family_name(features: tuple[str, str, str], gate: str, logic: str) -> str:
        parts = " + ".join(x.replace("_", " ").title() for x in features)
        return f"{parts} | {gate.replace('_', ' ').title()} | {logic.replace('_', ' ').title()}"

    @staticmethod
    def _pick(options: tuple[tuple[Any, ...], ...], variant: int, salt: int) -> tuple[Any, ...]:
        return options[(variant + salt) % len(options)]

    @classmethod
    def _feature_spec(cls, kind: str, variant: int, salt: int) -> tuple[Any, ...]:
        presets: dict[str, tuple[tuple[Any, ...], ...]] = {
            "ema_gap": ((5,30,.0001),(8,50,.0002),(10,75,.0003),(12,100,.0005),(15,150,.0008),(20,200,.0010)),
            "ema_slope": ((20,3,.0001),(30,5,.0002),(50,5,.0003),(75,8,.0004),(100,10,.0005),(150,14,.0007)),
            "momentum": ((2,.0002),(3,.0003),(5,.0005),(8,.0008),(14,.0012),(20,.0018)),
            "rsi_trend": ((7,55,45),(9,58,42),(14,55,45),(14,60,40),(21,58,42),(28,60,40)),
            "rsi_reversion": ((5,25,75),(7,30,70),(9,30,70),(14,25,75),(14,35,65),(21,30,70)),
            "zscore_trend": ((15,.6),(20,.8),(30,1.0),(40,1.2),(60,1.5),(90,1.8)),
            "zscore_reversion": ((15,.6),(20,.8),(30,1.0),(40,1.2),(60,1.5),(90,1.8)),
            "donchian_break": ((10,0.0),(15,.0001),(20,.0002),(30,.0003),(50,.0005),(75,.0008)),
            "donchian_fade": ((10,.08),(15,.10),(20,.12),(30,.15),(50,.18),(75,.22)),
            "candle_impulse": ((.45,),(.50,),(.55,),(.60,),(.65,),(.70,)),
            "range_location": ((10,.15),(15,.18),(20,.20),(30,.22),(50,.25),(75,.30)),
        }
        if kind not in presets:
            raise ValueError(f"Unknown invention feature: {kind}")
        return (kind, *cls._pick(presets[kind], variant, salt))

    @classmethod
    def _gate_spec(cls, gate: str, variant: int, salt: int) -> tuple[Any, ...]:
        if gate == "none":
            return ("none",)
        presets = {
            "atr_expansion": ((7,60,1.05),(10,90,1.10),(14,120,1.15),(18,160,1.20),(21,200,1.25),(28,240,1.30)),
            "atr_normal": ((7,60,1.60),(10,90,1.70),(14,120,1.80),(18,160,1.90),(21,200,2.00),(28,240,2.10)),
            "atr_compression": ((7,60,.95),(10,90,.90),(14,120,.85),(18,160,.80),(21,200,.78),(28,240,.75)),
        }
        if gate not in presets:
            raise ValueError(f"Unknown invention gate: {gate}")
        return (gate, *cls._pick(presets[gate], variant, salt))

    @classmethod
    def _candidate_params(
        cls,
        family_index: int,
        features: tuple[str, str, str],
        gate: str,
        logic: str,
        variant: int,
    ) -> tuple[Any, ...]:
        family_id = f"INV-{family_index + 1:04d}"
        specs = tuple(cls._feature_spec(kind, variant, family_index + i * 7) for i, kind in enumerate(features))
        gate_spec = cls._gate_spec(gate, variant, family_index + 31)
        return (family_id, variant + 1, logic, specs, gate_spec)

    @staticmethod
    def _row_family_index(row: dict[str, Any]) -> int | None:
        family_id = str(row.get("family_id", ""))
        if not family_id.startswith("INV-"):
            return None
        try:
            return max(0, int(family_id.split("-", 1)[1]) - 1)
        except (TypeError, ValueError):
            return None

    def invent(self) -> tuple[int, int]:
        """Persist unseen family structures and variants; return family/variant counts."""
        if self.families_per_cycle <= 0:
            return 0, 0
        templates = self.family_templates()
        with self._exclusive_lock():
            rows = self._bounded_rows(self._read_unlocked())
            protected = [r for r in rows if r.get("status") in {"PROMOTED", "QUARANTINED"}]
            experiments = [r for r in rows if r.get("status") not in {"PROMOTED", "QUARANTINED"}]
            capacity = max(0, self.max_library_size - len(protected))
            if capacity < self.variants_per_family:
                return 0, 0

            meta = self._read_meta_unlocked()
            persisted_indices = [i for i in (self._row_family_index(r) for r in rows) if i is not None]
            recovered_cursor = (max(persisted_indices) + 1) if persisted_indices else 0
            # If a crash wrote the library but not metadata, recovered_cursor wins.
            cursor = max(max(0, int(meta.get("next_family_index", 0))), recovered_cursor)
            family_limit = min(
                self.families_per_cycle,
                capacity // self.variants_per_family,
                max(0, len(templates) - cursor),
            )
            if family_limit <= 0:
                return 0, 0

            existing = {
                self.spec_key(str(row.get("family", "")), row.get("params", []))
                for row in rows
            }
            proposed: list[dict[str, Any]] = []
            actual_families = 0
            now = datetime.now(timezone.utc).isoformat()
            next_cursor = cursor
            for family_index in range(cursor, cursor + family_limit):
                features, gate, logic = templates[family_index]
                family_id = f"INV-{family_index + 1:04d}"
                family_name = self._family_name(features, gate, logic)
                family_rows: list[dict[str, Any]] = []
                for variant in range(self.variants_per_family):
                    params = self._candidate_params(family_index, features, gate, logic, variant)
                    key = self.spec_key("invented", params)
                    if key in existing:
                        continue
                    existing.add(key)
                    family_rows.append({
                        "family": "invented",
                        "family_id": family_id,
                        "family_name": family_name,
                        "variant_id": variant + 1,
                        "params": self._jsonable(params),
                        "status": "EXPERIMENTAL",
                        "origin": "invented_from_primitive_feature_grammar",
                        "created_at": now,
                        "formula": {"directional_features": list(features), "gate": gate, "logic": logic},
                    })
                if family_rows:
                    proposed.extend(family_rows)
                    actual_families += 1
                next_cursor = family_index + 1

            if not proposed:
                meta["next_family_index"] = next_cursor
                self._write_meta_unlocked(meta)
                return 0, 0

            overflow = max(0, len(experiments) + len(proposed) - capacity)
            if overflow:
                experiments = experiments[min(overflow, len(experiments)):]
            self._write_unlocked(protected + experiments + proposed)
            meta["next_family_index"] = next_cursor
            meta["families_invented_lifetime"] = max(next_cursor, int(meta.get("families_invented_lifetime", 0)))
            meta["last_invention_at"] = now
            self._write_meta_unlocked(meta)
            return actual_families, len(proposed)

    def family_count(self) -> int:
        return len({str(r.get("family_id")) for r in self._read() if r.get("family") == "invented" and r.get("family_id")})

    def promoted_family_count(self) -> int:
        return len({str(r.get("family_id")) for r in self._read() if r.get("family") == "invented" and r.get("family_id") and r.get("status") == "PROMOTED"})

    def audit_promotions(self, catalog: Iterable[Any], auditor: OverfitAuditor, tested_trials: int) -> tuple[int, int]:
        with self._exclusive_lock():
            rows = self._read_unlocked()
            if not rows:
                return 0, 0
            scores = {
                self.spec_key(result.candidate.family, result.candidate.params): result
                for result in catalog if result.candidate.family == "invented"
            }
            newly_promoted = newly_quarantined = 0
            now = datetime.now(timezone.utc).isoformat()
            touched = False
            for row in rows:
                key = self.spec_key(str(row.get("family", "")), row.get("params", []))
                result = scores.get(key)
                if result is None:
                    continue
                audit = auditor.audit(result, tested_trials)
                previous = str(row.get("status", "EXPERIMENTAL"))
                next_status = "PROMOTED" if audit.passed else ("QUARANTINED" if previous in {"PROMOTED", "QUARANTINED"} else "EXPERIMENTAL")
                if next_status == "PROMOTED" and previous != "PROMOTED":
                    newly_promoted += 1
                if next_status == "QUARANTINED" and previous == "PROMOTED":
                    newly_quarantined += 1
                row.update({
                    "status": next_status,
                    "last_audited_at": now,
                    "last_score": round(float(result.score), 6),
                    "last_oos_hit_rate": round(float(result.walk_forward_hit_rate or result.valid_hit_rate), 6),
                    "last_oos_trades": auditor.oos_trade_count(result),
                    "last_profit_factor": round(float(result.profit_factor), 6),
                    "last_avg_r": round(float(result.avg_r_multiple), 6),
                    "overfit_adjusted_score": round(float(audit.adjusted_score), 6),
                    "overfit_multiplicity_penalty": round(float(audit.multiplicity_penalty), 6),
                    "overfit_tested_trials": int(audit.tested_trials),
                    "overfit_passed": bool(audit.passed),
                    "overfit_reasons": list(audit.reasons),
                })
                if audit.passed:
                    row["last_promoted_at"] = now
                elif previous == "PROMOTED":
                    row["last_quarantined_at"] = now
                touched = True
            if touched:
                self._write_unlocked(rows)
            return newly_promoted, newly_quarantined

    @staticmethod
    def _cached(cache: dict[tuple[Any, ...], Any], key: tuple[Any, ...], builder):
        if key not in cache:
            cache[key] = builder()
        return cache[key]

    @classmethod
    def _directional_feature(cls, df: pd.DataFrame, spec: tuple[Any, ...], cache: dict[tuple[Any, ...], Any]) -> pd.Series:
        close = df["close"]
        kind = str(spec[0])
        out = pd.Series(0, index=df.index, dtype="int8")
        if kind == "ema_gap":
            _, fast, slow, threshold = spec
            ef = cls._cached(cache, ("ema", int(fast)), lambda: ema(close, int(fast)))
            es = cls._cached(cache, ("ema", int(slow)), lambda: ema(close, int(slow)))
            gap = (ef - es) / close.replace(0, np.nan)
            out[gap > float(threshold)] = 1
            out[gap < -float(threshold)] = -1
        elif kind == "ema_slope":
            _, span, lookback, threshold = spec
            e = cls._cached(cache, ("ema", int(span)), lambda: ema(close, int(span)))
            slope = e.pct_change(int(lookback))
            out[slope > float(threshold)] = 1
            out[slope < -float(threshold)] = -1
        elif kind == "momentum":
            _, period, threshold = spec
            m = cls._cached(cache, ("roc", int(period)), lambda: roc(close, int(period)))
            out[m > float(threshold)] = 1
            out[m < -float(threshold)] = -1
        elif kind in {"rsi_trend", "rsi_reversion"}:
            _, period, a, b = spec
            rs = cls._cached(cache, ("rsi", int(period)), lambda: rsi(close, int(period)))
            if kind == "rsi_trend":
                out[rs > float(a)] = 1
                out[rs < float(b)] = -1
            else:
                out[rs < float(a)] = 1
                out[rs > float(b)] = -1
        elif kind in {"zscore_trend", "zscore_reversion"}:
            _, period, threshold = spec
            z = cls._cached(cache, ("z", int(period)), lambda: rolling_zscore(close, int(period)))
            if kind == "zscore_trend":
                out[z > float(threshold)] = 1
                out[z < -float(threshold)] = -1
            else:
                out[z < -float(threshold)] = 1
                out[z > float(threshold)] = -1
        elif kind in {"donchian_break", "donchian_fade"}:
            _, lookback, amount = spec
            hi, lo = cls._cached(cache, ("donchian", int(lookback)), lambda: donchian(df, int(lookback)))
            if kind == "donchian_break":
                out[close > hi * (1 + float(amount))] = 1
                out[close < lo * (1 - float(amount))] = -1
            else:
                location = (close - lo) / (hi - lo).replace(0, np.nan)
                out[location < float(amount)] = 1
                out[location > 1 - float(amount)] = -1
        elif kind == "candle_impulse":
            _, min_body_ratio = spec
            ratio = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)
            out[(df["close"] > df["open"]) & (ratio >= float(min_body_ratio))] = 1
            out[(df["close"] < df["open"]) & (ratio >= float(min_body_ratio))] = -1
        elif kind == "range_location":
            _, lookback, edge = spec
            hi = cls._cached(cache, ("range_hi", int(lookback)), lambda: df["high"].rolling(int(lookback)).max().shift(1))
            lo = cls._cached(cache, ("range_lo", int(lookback)), lambda: df["low"].rolling(int(lookback)).min().shift(1))
            location = (close - lo) / (hi - lo).replace(0, np.nan)
            out[location > 1 - float(edge)] = 1
            out[location < float(edge)] = -1
        return out

    @classmethod
    def _gate(cls, df: pd.DataFrame, spec: tuple[Any, ...], cache: dict[tuple[Any, ...], Any]) -> pd.Series:
        kind = str(spec[0])
        if kind == "none":
            return pd.Series(True, index=df.index, dtype=bool)
        try:
            _, period, baseline, multiple = spec
        except (TypeError, ValueError):
            return pd.Series(False, index=df.index, dtype=bool)
        ratio = cls._cached(cache, ("atr", int(period)), lambda: atr(df, int(period))) / df["close"].replace(0, np.nan)
        median = ratio.rolling(int(baseline)).median()
        mult = float(multiple)
        if kind == "atr_expansion":
            return (ratio >= median * mult).fillna(False)
        if kind in {"atr_normal", "atr_compression"}:
            return (ratio <= median * mult).fillna(False)
        return pd.Series(False, index=df.index, dtype=bool)

    @classmethod
    def signal(cls, df: pd.DataFrame, params: tuple[Any, ...], cache: dict[tuple[Any, ...], Any] | None = None) -> pd.Series:
        """Evaluate one invented formula. Malformed specs fail closed to HOLD."""
        out = pd.Series(0, index=df.index, dtype="int8")
        cache = {} if cache is None else cache
        try:
            _family_id, _variant_id, logic, feature_specs, gate_spec = params
            features = tuple(tuple(x) for x in feature_specs)
            gate_tuple = tuple(gate_spec)
        except (TypeError, ValueError):
            return out
        if len(features) != 3:
            return out
        votes = [cls._directional_feature(df, spec, cache) for spec in features]
        gate = cls._gate(df, gate_tuple, cache)
        stacked = pd.concat(votes, axis=1)
        positive = (stacked == 1).sum(axis=1)
        negative = (stacked == -1).sum(axis=1)
        if logic == "all":
            long_mask, short_mask = positive == 3, negative == 3
        elif logic == "majority":
            long_mask, short_mask = positive >= 2, negative >= 2
        elif logic == "lead_confirm":
            confirmations = pd.concat(votes[1:], axis=1)
            long_mask = (votes[0] == 1) & ((confirmations == 1).sum(axis=1) >= 1)
            short_mask = (votes[0] == -1) & ((confirmations == -1).sum(axis=1) >= 1)
        else:
            return out
        out[gate & long_mask & ~short_mask] = 1
        out[gate & short_mask & ~long_mask] = -1
        return out
