from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .models import Direction, TradeSignal


@dataclass(frozen=True)
class CalibrationResult:
    probability: float
    samples: int
    wins: int
    brier_score: float | None = None


@dataclass(frozen=True)
class ReservationResult:
    reserved: bool
    reason: str
    trades_today: int


class OutcomeCalibrationAgent:
    """Persist signals, synchronize delivery, resolve outcomes and calibrate confidence."""

    name = "Outcome & Calibration Desk"

    def __init__(
        self,
        db_path: str = "data/xau_outcomes.sqlite3",
        max_age_hours: int = 72,
        bin_width: float = 0.05,
        prior_strength: float = 20.0,
    ) -> None:
        self.db_path = db_path
        self.max_age_hours = max(1, int(max_age_hours))
        self.bin_width = float(np.clip(bin_width, 0.02, 0.20))
        self.prior_strength = max(5.0, float(prior_strength))
        if db_path != ":memory:":
            Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, name: str, ddl: str) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(signal_outcomes)").fetchall()}
        if name not in columns:
            conn.execute(f"ALTER TABLE signal_outcomes ADD COLUMN {ddl}")

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_key TEXT NOT NULL UNIQUE,
                    observed_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    selection_confidence REAL NOT NULL,
                    calibrated_confidence REAL NOT NULL,
                    strategy TEXT NOT NULL DEFAULT '',
                    strategy_family TEXT NOT NULL DEFAULT '',
                    regime TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    resolved_at TEXT,
                    resolved_price REAL,
                    delivery_state TEXT NOT NULL DEFAULT 'SENT',
                    max_holding_minutes INTEGER NOT NULL DEFAULT 4320,
                    resolution_interval_minutes INTEGER NOT NULL DEFAULT 1,
                    telegram_message_id TEXT
                )
                """
            )
            self._ensure_column(conn, "delivery_state", "delivery_state TEXT NOT NULL DEFAULT 'SENT'")
            self._ensure_column(conn, "max_holding_minutes", "max_holding_minutes INTEGER NOT NULL DEFAULT 4320")
            self._ensure_column(conn, "resolution_interval_minutes", "resolution_interval_minutes INTEGER NOT NULL DEFAULT 1")
            self._ensure_column(conn, "telegram_message_id", "telegram_message_id TEXT")
            # A reservation left behind by a crashed process may already have been
            # delivered externally. Treat it as UNKNOWN: suppress duplicates and
            # daily-cap reuse, but exclude it from probability calibration.
            conn.execute(
                "UPDATE signal_outcomes SET delivery_state='UNKNOWN' WHERE delivery_state='RESERVED'"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_outcomes_status ON signal_outcomes(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_outcomes_confidence ON signal_outcomes(selection_confidence)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_outcomes_context ON signal_outcomes(strategy_family, regime, status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_outcomes_observed_at ON signal_outcomes(observed_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_outcomes_delivery ON signal_outcomes(delivery_state)")

    @staticmethod
    def utc_iso(value: datetime | pd.Timestamp | str) -> str:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat()

    @staticmethod
    def _family(strategy: str) -> str:
        return strategy.split("(", 1)[0].strip() if strategy else ""

    @staticmethod
    def _signal_key(signal: TradeSignal, setup_at: str) -> str:
        # Identity represents one strategy/direction decision on one completed
        # setup candle. Entry/SL/TP can refresh without creating a second setup.
        raw = "|".join(
            [setup_at, signal.symbol, signal.direction.value, signal.selected_strategy]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _runtime_limits(self, signal: TradeSignal) -> tuple[int, int]:
        stats = signal.strategy_stats or {}
        holding = int(stats.get("max_holding_minutes", self.max_age_hours * 60))
        resolution = int(stats.get("resolution_interval_minutes", 1))
        return max(1, min(holding, self.max_age_hours * 60)), max(1, resolution)

    def exists(self, signal: TradeSignal, setup_at: datetime | pd.Timestamp | str) -> bool:
        setup = self.utc_iso(setup_at)
        key = self._signal_key(signal, setup)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM signal_outcomes WHERE signal_key=? AND delivery_state!='FAILED' LIMIT 1",
                (key,),
            ).fetchone()
        return row is not None

    def count_emitted_between(
        self,
        start_at: datetime | pd.Timestamp | str,
        end_at: datetime | pd.Timestamp | str,
    ) -> int:
        start = self.utc_iso(start_at)
        end = self.utc_iso(end_at)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n FROM signal_outcomes
                WHERE observed_at >= ? AND observed_at < ?
                  AND delivery_state IN ('RESERVED','SENT','UNKNOWN')
                """,
                (start, end),
            ).fetchone()
        return int(row["n"] or 0)

    def reserve_if_under_cap(
        self,
        signal: TradeSignal,
        observed_at: datetime | pd.Timestamp | str,
        selection_confidence: float,
        setup_at: datetime | pd.Timestamp | str,
        day_start: datetime | pd.Timestamp | str,
        day_end: datetime | pd.Timestamp | str,
        max_per_day: int,
    ) -> ReservationResult:
        """Atomically reserve dedupe identity and a daily signal slot."""
        observed = self.utc_iso(observed_at)
        setup = self.utc_iso(setup_at)
        start = self.utc_iso(day_start)
        end = self.utc_iso(day_end)
        key = self._signal_key(signal, setup)
        holding, resolution = self._runtime_limits(signal)
        raw_conf = float(selection_confidence)
        calibrated = float(signal.confidence)

        conn = self._connect()
        try:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            duplicate = conn.execute(
                "SELECT delivery_state FROM signal_outcomes WHERE signal_key=? LIMIT 1", (key,)
            ).fetchone()
            if duplicate is not None and duplicate["delivery_state"] != "FAILED":
                conn.execute("ROLLBACK")
                return ReservationResult(False, "duplicate setup", 0)
            if duplicate is not None:
                conn.execute("DELETE FROM signal_outcomes WHERE signal_key=?", (key,))

            row = conn.execute(
                """
                SELECT COUNT(*) AS n FROM signal_outcomes
                WHERE observed_at >= ? AND observed_at < ?
                  AND delivery_state IN ('RESERVED','SENT','UNKNOWN')
                """,
                (start, end),
            ).fetchone()
            count = int(row["n"] or 0)
            if count >= max(1, int(max_per_day)):
                conn.execute("ROLLBACK")
                return ReservationResult(False, "daily trade cap reached", count)

            conn.execute(
                """
                INSERT INTO signal_outcomes (
                    signal_key, observed_at, symbol, direction, entry, stop_loss,
                    take_profit, selection_confidence, calibrated_confidence,
                    strategy, strategy_family, regime, status, delivery_state,
                    max_holding_minutes, resolution_interval_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', 'RESERVED', ?, ?)
                """,
                (
                    key,
                    observed,
                    signal.symbol,
                    signal.direction.value,
                    float(signal.entry),
                    float(signal.stop_loss),
                    float(signal.take_profit),
                    raw_conf,
                    calibrated,
                    signal.selected_strategy or "",
                    self._family(signal.selected_strategy),
                    signal.regime or "",
                    holding,
                    resolution,
                ),
            )
            conn.execute("COMMIT")
            return ReservationResult(True, "reserved", count)
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def mark_delivery_state(
        self,
        signal: TradeSignal,
        setup_at: datetime | pd.Timestamp | str,
        state: str,
        telegram_message_id: str | int | None = None,
    ) -> None:
        allowed = {"RESERVED", "SENT", "UNKNOWN", "FAILED"}
        if state not in allowed:
            raise ValueError(f"Unknown delivery state: {state}")
        key = self._signal_key(signal, self.utc_iso(setup_at))
        with self._connect() as conn:
            conn.execute(
                "UPDATE signal_outcomes SET delivery_state=?, telegram_message_id=? WHERE signal_key=?",
                (state, None if telegram_message_id is None else str(telegram_message_id), key),
            )

    def record(
        self,
        signal: TradeSignal,
        observed_at: datetime | pd.Timestamp | str,
        selection_confidence: float | None = None,
        setup_at: datetime | pd.Timestamp | str | None = None,
    ) -> bool:
        """Record a known-emitted signal (used by paper mode and tests)."""
        observed = self.utc_iso(observed_at)
        setup = self.utc_iso(setup_at if setup_at is not None else observed_at)
        raw_conf = float(signal.confidence if selection_confidence is None else selection_confidence)
        calibrated = float(signal.confidence)
        key = self._signal_key(signal, setup)
        holding, resolution = self._runtime_limits(signal)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO signal_outcomes (
                    signal_key, observed_at, symbol, direction, entry, stop_loss,
                    take_profit, selection_confidence, calibrated_confidence,
                    strategy, strategy_family, regime, status, delivery_state,
                    max_holding_minutes, resolution_interval_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', 'SENT', ?, ?)
                """,
                (
                    key,
                    observed,
                    signal.symbol,
                    signal.direction.value,
                    float(signal.entry),
                    float(signal.stop_loss),
                    float(signal.take_profit),
                    raw_conf,
                    calibrated,
                    signal.selected_strategy or "",
                    self._family(signal.selected_strategy),
                    signal.regime or "",
                    holding,
                    resolution,
                ),
            )
            return cursor.rowcount > 0

    def oldest_unresolved_observed_at(self) -> pd.Timestamp | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MIN(observed_at) AS oldest FROM signal_outcomes
                WHERE status='OPEN' AND delivery_state IN ('RESERVED','SENT','UNKNOWN')
                """
            ).fetchone()
        return pd.Timestamp(row["oldest"]) if row and row["oldest"] else None

    def resolve_open(self, df: pd.DataFrame, interval_minutes: int = 1) -> dict[str, int]:
        """Resolve open signals without pretending partial pre-emission bars are known."""
        counts = {"wins": 0, "losses": 0, "expired": 0, "ambiguous": 0}
        if df is None or df.empty or not {"datetime", "high", "low"}.issubset(df.columns):
            return counts

        candles = df[["datetime", "high", "low"]].copy()
        candles["datetime"] = pd.to_datetime(candles["datetime"], utc=True, errors="coerce")
        candles["high"] = pd.to_numeric(candles["high"], errors="coerce")
        candles["low"] = pd.to_numeric(candles["low"], errors="coerce")
        candles = candles.dropna().sort_values("datetime")
        if candles.empty:
            return counts

        delta = pd.Timedelta(minutes=max(1, int(interval_minutes)))
        latest_end = candles["datetime"].iloc[-1] + delta
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM signal_outcomes
                WHERE status='OPEN' AND delivery_state IN ('SENT','UNKNOWN')
                ORDER BY observed_at
                """
            ).fetchall()

            for row in rows:
                observed = pd.Timestamp(row["observed_at"])
                if observed.tzinfo is None:
                    observed = observed.tz_localize("UTC")
                else:
                    observed = observed.tz_convert("UTC")
                holding_minutes = min(
                    max(1, int(row["max_holding_minutes"] or self.max_age_hours * 60)),
                    self.max_age_hours * 60,
                )
                expiry = observed + pd.Timedelta(minutes=holding_minutes)
                relevant = candles[
                    (candles["datetime"] + delta > observed)
                    & (candles["datetime"] <= expiry)
                ]
                resolved_status: str | None = None
                resolved_at: str | None = None
                resolved_price: float | None = None

                for candle in relevant.itertuples(index=False):
                    candle_start = pd.Timestamp(candle.datetime)
                    candle_end = candle_start + delta
                    high = float(candle.high)
                    low = float(candle.low)
                    if row["direction"] == Direction.BUY.value:
                        stop_hit = low <= float(row["stop_loss"])
                        target_hit = high >= float(row["take_profit"])
                    else:
                        stop_hit = high >= float(row["stop_loss"])
                        target_hit = low <= float(row["take_profit"])

                    overlaps_emission = candle_start < observed < candle_end
                    if overlaps_emission and (stop_hit or target_hit):
                        # OHLC cannot tell whether the hit happened before or after
                        # the actual send inside this candle. Exclude it from calibration.
                        resolved_status = "AMBIGUOUS"
                        resolved_at = self.utc_iso(candle_start)
                        counts["ambiguous"] += 1
                        break
                    if overlaps_emission:
                        continue
                    if stop_hit:
                        resolved_status = "LOSS"
                        resolved_at = self.utc_iso(candle_start)
                        resolved_price = float(row["stop_loss"])
                        counts["losses"] += 1
                        break
                    if target_hit:
                        resolved_status = "WIN"
                        resolved_at = self.utc_iso(candle_start)
                        resolved_price = float(row["take_profit"])
                        counts["wins"] += 1
                        break

                if resolved_status is None and latest_end >= expiry:
                    resolved_status = "EXPIRED"
                    resolved_at = self.utc_iso(expiry)
                    counts["expired"] += 1

                if resolved_status is not None:
                    conn.execute(
                        """
                        UPDATE signal_outcomes
                        SET status=?, resolved_at=?, resolved_price=?
                        WHERE id=? AND status='OPEN'
                        """,
                        (resolved_status, resolved_at, resolved_price, int(row["id"])),
                    )
        return counts

    def _bucket_bounds(self, probability: float) -> tuple[float, float]:
        p = float(np.clip(probability, 0.0, 0.999999))
        lower = np.floor(p / self.bin_width) * self.bin_width
        upper = min(1.000001, lower + self.bin_width)
        return float(lower), float(upper)

    def _resolved_stats(
        self,
        lower: float,
        upper: float,
        strategy_family: str = "",
        regime: str = "",
    ) -> tuple[int, int]:
        where = [
            "status IN ('WIN','LOSS')",
            "delivery_state='SENT'",
            "selection_confidence >= ?",
            "selection_confidence < ?",
        ]
        params: list[object] = [lower, upper]
        if strategy_family:
            where.append("strategy_family = ?")
            params.append(strategy_family)
        if regime:
            where.append("regime = ?")
            params.append(regime)
        sql = f"SELECT COUNT(*) AS n, SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) AS wins FROM signal_outcomes WHERE {' AND '.join(where)}"
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["n"] or 0), int(row["wins"] or 0)

    def _brier_score(self) -> float | None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT calibrated_confidence, status FROM signal_outcomes
                WHERE status IN ('WIN','LOSS') AND delivery_state='SENT'
                """
            ).fetchall()
        if not rows:
            return None
        errors = []
        for row in rows:
            outcome = 1.0 if row["status"] == "WIN" else 0.0
            errors.append((float(row["calibrated_confidence"]) - outcome) ** 2)
        return float(np.mean(errors))

    def calibrate(self, probability: float, strategy: str = "", regime: str = "") -> CalibrationResult:
        raw = float(np.clip(probability, 0.01, 0.99))
        lower, upper = self._bucket_bounds(raw)
        n, wins = self._resolved_stats(lower, upper)
        if n == 0:
            return CalibrationResult(raw, 0, 0, self._brier_score())

        global_posterior = (wins + raw * self.prior_strength) / (n + self.prior_strength)
        posterior = global_posterior
        used_n, used_wins = n, wins

        family = self._family(strategy)
        context_n, context_wins = self._resolved_stats(lower, upper, family, regime)
        if context_n >= 8 and (family or regime):
            context_posterior = (context_wins + raw * self.prior_strength) / (
                context_n + self.prior_strength
            )
            context_weight = min(0.65, context_n / (context_n + 20.0))
            posterior = global_posterior * (1.0 - context_weight) + context_posterior * context_weight
            used_n, used_wins = context_n, context_wins

        calibrated = float(np.clip(posterior, 0.05, 0.95))
        return CalibrationResult(calibrated, used_n, used_wins, self._brier_score())

    def summary(self) -> dict[str, float | int | None]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS total,
                    SUM(CASE WHEN status='OPEN' THEN 1 ELSE 0 END) AS open_count,
                    SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN status='LOSS' THEN 1 ELSE 0 END) AS losses,
                    SUM(CASE WHEN status='EXPIRED' THEN 1 ELSE 0 END) AS expired,
                    SUM(CASE WHEN status='AMBIGUOUS' THEN 1 ELSE 0 END) AS ambiguous,
                    SUM(CASE WHEN delivery_state='UNKNOWN' THEN 1 ELSE 0 END) AS delivery_unknown
                FROM signal_outcomes
                """
            ).fetchone()
        wins = int(row["wins"] or 0)
        losses = int(row["losses"] or 0)
        resolved = wins + losses
        return {
            "total": int(row["total"] or 0),
            "open": int(row["open_count"] or 0),
            "wins": wins,
            "losses": losses,
            "expired": int(row["expired"] or 0),
            "ambiguous": int(row["ambiguous"] or 0),
            "delivery_unknown": int(row["delivery_unknown"] or 0),
            "resolved": resolved,
            "win_rate": (wins / resolved) if resolved else None,
            "brier_score": self._brier_score(),
        }
