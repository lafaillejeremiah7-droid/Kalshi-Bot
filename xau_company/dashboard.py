from __future__ import annotations

import json
import logging
import os
import re
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EMPLOYEES: tuple[dict[str, str], ...] = (
    {"id": "market_data", "name": "Market Data", "room": "market"},
    {"id": "data_quality", "name": "Data Quality", "room": "market"},
    {"id": "tf_1m", "name": "1m Desk", "room": "market"},
    {"id": "tf_5m", "name": "5m Desk", "room": "market"},
    {"id": "tf_15m", "name": "15m Desk", "room": "market"},
    {"id": "tf_1h", "name": "1h Desk", "room": "market"},
    {"id": "tf_4h", "name": "4h Desk", "room": "market"},
    {"id": "research_lab", "name": "Research Lab", "room": "research"},
    {"id": "strategy_evolution", "name": "Strategy Evolution", "room": "research"},
    {"id": "backtest_auditor", "name": "Backtest Auditor", "room": "research"},
    {"id": "overfit_auditor", "name": "Overfit Auditor", "room": "research"},
    {"id": "trend", "name": "Trend Desk", "room": "analysis"},
    {"id": "breakout", "name": "Breakout Desk", "room": "analysis"},
    {"id": "mean_reversion", "name": "Mean Reversion", "room": "analysis"},
    {"id": "momentum", "name": "Momentum Desk", "room": "analysis"},
    {"id": "price_action", "name": "Price Action", "room": "analysis"},
    {"id": "structure", "name": "Structure Desk", "room": "analysis"},
    {"id": "regime", "name": "Regime Analyst", "room": "risk"},
    {"id": "usd", "name": "USD Strength", "room": "risk"},
    {"id": "yield", "name": "Treasury Yield", "room": "risk"},
    {"id": "news", "name": "News Risk", "room": "risk"},
    {"id": "volatility", "name": "Volatility Guard", "room": "risk"},
    {"id": "session", "name": "Session Desk", "room": "risk"},
    {"id": "selector", "name": "Strategy Selector", "room": "decision"},
    {"id": "boss", "name": "Boss / CEO", "room": "decision"},
    {"id": "outcomes", "name": "Outcome & Calibration", "room": "performance"},
    {"id": "frequency", "name": "Trade Frequency", "room": "performance"},
)

ROOMS: tuple[dict[str, str], ...] = (
    {"id": "market", "name": "Market Intelligence", "accent": "blue"},
    {"id": "research", "name": "Strategy Laboratory", "accent": "purple"},
    {"id": "analysis", "name": "Market Analysis", "accent": "green"},
    {"id": "risk", "name": "Macro & Risk", "accent": "gold"},
    {"id": "decision", "name": "Decision Room", "accent": "red"},
    {"id": "performance", "name": "Performance Room", "accent": "cyan"},
)

_ANALYSIS_IDS = ("trend", "breakout", "mean_reversion", "momentum", "price_action", "structure")
_RISK_IDS = ("regime", "usd", "yield", "news", "volatility", "session")
_TIMEFRAME_IDS = ("tf_1m", "tf_5m", "tf_15m", "tf_1h", "tf_4h")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def default_state() -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": utc_now_iso(),
        "system": {
            "status": "WAITING",
            "headline": "Waiting for company runtime",
            "market_open": None,
            "telegram": "UNKNOWN",
            "cycle": None,
            "last_error": None,
        },
        "rooms": [dict(room) for room in ROOMS],
        "employees": [
            {
                **dict(employee),
                "state": "idle",
                "task": "Waiting for runtime activity",
                "detail": "",
                "direction": "HOLD",
                "confidence": None,
                "updated_at": None,
            }
            for employee in EMPLOYEES
        ],
        "handoffs": [],
        "activity": [],
        "boss": {
            "decision": "HOLD",
            "symbol": "XAU/USD",
            "strategy": "No decision yet",
            "confidence": None,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "regime": None,
            "reason": "Waiting for synchronized bot output",
        },
        "performance": {
            "signals": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "brier": None,
            "recent_signals": [],
        },
    }


class DashboardPublisher:
    """Small, atomic JSON state publisher for the visual operations floor.

    The trading runtime remains the source of truth. This class only mirrors
    observable work states into a dashboard-safe file; it never influences a
    trading decision.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None, max_events: int = 80) -> None:
        self.path = Path(path or os.getenv("DASHBOARD_STATE_PATH", "data/dashboard_state.json"))
        self.max_events = max(10, int(max_events))
        self._lock = threading.RLock()
        self._state: dict[str, Any] | None = None
        self._event_counter = 0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_loaded()
            return deepcopy(self._state or default_state())

    def _ensure_loaded(self) -> None:
        if self._state is not None:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("employees"), list):
                self._state = raw
                return
        except (OSError, ValueError, TypeError):
            pass
        self._state = default_state()

    def _write(self) -> None:
        assert self._state is not None
        self._state["updated_at"] = utc_now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(self._state, indent=2, sort_keys=False, default=str)
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)

    def _employee(self, employee_id: str) -> dict[str, Any] | None:
        assert self._state is not None
        for employee in self._state.get("employees", []):
            if employee.get("id") == employee_id:
                return employee
        return None

    def system(
        self,
        *,
        status: str | None = None,
        headline: str | None = None,
        market_open: bool | None = None,
        telegram: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._ensure_loaded()
            assert self._state is not None
            sys = self._state.setdefault("system", {})
            if status is not None:
                sys["status"] = status
            if headline is not None:
                sys["headline"] = headline
            if market_open is not None:
                sys["market_open"] = bool(market_open)
            if telegram is not None:
                sys["telegram"] = telegram
            if error is not None:
                sys["last_error"] = error
            self._write()

    def employee(
        self,
        employee_id: str,
        state: str,
        task: str,
        *,
        detail: str = "",
        direction: str | None = None,
        confidence: float | None = None,
    ) -> None:
        with self._lock:
            self._ensure_loaded()
            assert self._state is not None
            row = self._employee(employee_id)
            if row is None:
                return
            row["state"] = state
            row["task"] = task
            row["detail"] = detail
            row["updated_at"] = utc_now_iso()
            if direction is not None:
                row["direction"] = direction
            if confidence is not None:
                row["confidence"] = max(0.0, min(1.0, float(confidence)))
            self._write()

    def employees(
        self,
        employee_ids: tuple[str, ...] | list[str],
        state: str,
        task: str,
        *,
        detail: str = "",
    ) -> None:
        with self._lock:
            self._ensure_loaded()
            assert self._state is not None
            stamp = utc_now_iso()
            for employee_id in employee_ids:
                row = self._employee(employee_id)
                if row is None:
                    continue
                row.update({"state": state, "task": task, "detail": detail, "updated_at": stamp})
            self._write()

    def handoff(self, source: str, target: str, document: str, detail: str = "") -> None:
        with self._lock:
            self._ensure_loaded()
            assert self._state is not None
            self._event_counter += 1
            event = {
                "id": f"{int(datetime.now(timezone.utc).timestamp() * 1000)}-{self._event_counter}",
                "at": utc_now_iso(),
                "from": source,
                "to": target,
                "document": document,
                "detail": detail,
            }
            handoffs = self._state.setdefault("handoffs", [])
            handoffs.append(event)
            del handoffs[:-self.max_events]
            self._activity(f"{document}: {source} → {target}", detail, event_id=event["id"])
            source_row = self._employee(source)
            target_row = self._employee(target)
            if source_row is not None:
                source_row["state"] = "handoff"
                source_row["task"] = f"Sending {document}"
                source_row["updated_at"] = event["at"]
            if target_row is not None:
                target_row["state"] = "receiving"
                target_row["task"] = f"Receiving {document}"
                target_row["updated_at"] = event["at"]
            self._write()

    def _activity(self, text: str, detail: str = "", event_id: str | None = None) -> None:
        assert self._state is not None
        activity = self._state.setdefault("activity", [])
        activity.append(
            {
                "id": event_id or f"activity-{len(activity) + 1}-{int(datetime.now().timestamp() * 1000)}",
                "at": utc_now_iso(),
                "text": text,
                "detail": detail,
            }
        )
        del activity[:-self.max_events]

    def boss_decision(
        self,
        direction: str,
        *,
        strategy: str = "",
        confidence: float | None = None,
        symbol: str = "XAU/USD",
        entry: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        regime: str | None = None,
        reason: str = "",
    ) -> None:
        with self._lock:
            self._ensure_loaded()
            assert self._state is not None
            boss = self._state.setdefault("boss", {})
            boss.update(
                {
                    "decision": direction,
                    "symbol": symbol,
                    "strategy": strategy or boss.get("strategy") or "",
                    "confidence": confidence,
                    "entry": entry,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "regime": regime,
                    "reason": reason,
                }
            )
            boss_row = self._employee("boss")
            if boss_row is not None:
                boss_row.update(
                    {
                        "state": "done" if direction in {"BUY", "SELL"} else "idle",
                        "task": f"Final decision: {direction}",
                        "direction": direction,
                        "confidence": confidence,
                        "updated_at": utc_now_iso(),
                    }
                )
            self._write()


_SIGNAL_RE = re.compile(
    r"Signal\s+(BUY|SELL)\s+using\s+(.+?)\s+raw_confidence\s+([0-9.]+)%\s+calibrated_confidence\s+([0-9.]+)%",
    re.IGNORECASE,
)
_STRATEGY_RE = re.compile(r"Strategy:\s*([^|]+)", re.IGNORECASE)
_ENTRY_RE = re.compile(r"Entry:\s*([0-9.,]+)", re.IGNORECASE)
_SL_RE = re.compile(r"(?:SL|Stop(?: Loss)?):\s*([0-9.,]+)", re.IGNORECASE)
_TP_RE = re.compile(r"(?:TP|Take(?: Profit)?):\s*([0-9.,]+)", re.IGNORECASE)
_DIRECTION_RE = re.compile(r"\b(BUY|SELL)\b")


def _number(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except (TypeError, ValueError):
        return None


class DashboardLogHandler(logging.Handler):
    """Translate truthful runtime log milestones into visible office actions."""

    def __init__(self, publisher: DashboardPublisher | None = None) -> None:
        super().__init__(level=logging.INFO)
        self.publisher = publisher or DashboardPublisher()

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - logging must never break trading
        try:
            if record.name != "xau-company":
                return
            message = record.getMessage()
            self._handle(message, record.levelno)
        except Exception:
            return

    def _handle(self, message: str, level: int) -> None:
        p = self.publisher

        if level >= logging.ERROR or message.startswith("Cycle failed"):
            p.system(status="ERROR", headline="Runtime cycle failed", error=message[:500])
            p.employee("boss", "blocked", "Cycle interrupted", detail=message[:220])
            return

        if "XAU/USD session is closed" in message:
            p.system(status="HOLD", headline="Market closed — company standing by", market_open=False)
            p.employee("session", "blocked", "Market closed", detail="Session guard veto")
            p.employee("boss", "blocked", "No new authorization", detail="Session guard veto")
            p.handoff("session", "boss", "Market Closed Veto")
            return

        if message.startswith("Market-quality veto:"):
            p.system(status="HOLD", headline="Market data quality veto", market_open=True)
            p.employee("market_data", "blocked", "Reference data unavailable", detail=message)
            p.employee("data_quality", "blocked", "Rejected market input", detail=message)
            p.handoff("data_quality", "boss", "Data Quality Veto", message)
            return

        if message.startswith("Outcome desk resolved"):
            p.system(status="ONLINE", headline="Company runtime active", market_open=True)
            p.employee("outcomes", "done", "Resolved forward outcomes", detail=message)
            p.handoff("outcomes", "research_lab", "Calibration Report", message)
            return

        if message.startswith("Strategy lab universe="):
            p.system(status="ONLINE", headline="Research cycle completed", market_open=True)
            p.employee("research_lab", "done", "Research universe evaluated", detail=message)
            p.employee("strategy_evolution", "done", "Discovery cycle completed")
            p.employee("backtest_auditor", "done", "Backtests audited")
            p.employee("overfit_auditor", "done", "Overfit gates applied")
            p.handoff("research_lab", "strategy_evolution", "Candidate Research Pack")
            p.handoff("strategy_evolution", "backtest_auditor", "Novel Strategy Pack")
            p.handoff("backtest_auditor", "overfit_auditor", "Backtest Audit")
            p.handoff("overfit_auditor", "selector", "Approved Strategy Catalog")
            return

        if message.startswith("Signal timing veto:"):
            p.employee("session", "blocked", "Setup timing rejected", detail=message)
            p.employee("selector", "blocked", "Stale setup cannot be selected", detail=message)
            p.employee("boss", "blocked", "Authorization withheld", detail=message)
            p.handoff("session", "boss", "Timing Veto", message)
            return

        if message.startswith("Calibration veto:"):
            p.employee("outcomes", "blocked", "Forward calibration veto", detail=message)
            p.employee("boss", "blocked", "Confidence below release threshold", detail=message)
            p.handoff("outcomes", "boss", "Calibration Veto", message)
            return

        if message.startswith("Frequency veto:") or message.startswith("Signal reservation veto:"):
            p.employee("frequency", "blocked", "Daily signal guard veto", detail=message)
            p.employee("boss", "blocked", "Signal release withheld", detail=message)
            p.handoff("frequency", "boss", "Frequency Veto", message)
            return

        signal_match = _SIGNAL_RE.search(message)
        if signal_match:
            direction = signal_match.group(1).upper()
            strategy = signal_match.group(2).strip()
            confidence = float(signal_match.group(4)) / 100.0
            p.system(status="ONLINE", headline="Boss authorization prepared", market_open=True)
            p.employees(_TIMEFRAME_IDS, "done", "Timeframe report submitted")
            p.employees(_ANALYSIS_IDS, "done", "Specialist vote submitted")
            p.employees(_RISK_IDS, "done", "Risk/macro check submitted")
            p.employee("selector", "done", f"Selected {strategy}", direction=direction, confidence=confidence)
            p.handoff("tf_15m", "selector", "Structure Report")
            p.handoff("structure", "selector", "Market Analysis Vote")
            p.handoff("regime", "selector", "Regime Report")
            p.handoff("selector", "boss", "Strategy Recommendation", strategy)
            p.boss_decision(direction, strategy=strategy, confidence=confidence, reason="Strategy passed company gates")
            return

        if message.startswith("PAPER_MODE:"):
            payload = message.split("PAPER_MODE:", 1)[1].strip()
            direction_match = _DIRECTION_RE.search(payload)
            strategy_match = _STRATEGY_RE.search(payload)
            p.system(status="ONLINE", headline="Paper signal recorded", telegram="PAPER")
            if direction_match:
                current = p.snapshot().get("boss", {})
                p.boss_decision(
                    direction_match.group(1).upper(),
                    strategy=(strategy_match.group(1).strip() if strategy_match else current.get("strategy", "")),
                    confidence=current.get("confidence"),
                    entry=_number(_ENTRY_RE.search(payload)),
                    stop_loss=_number(_SL_RE.search(payload)),
                    take_profit=_number(_TP_RE.search(payload)),
                    reason="Paper-mode authorization recorded",
                )
            p.employee("frequency", "done", "Daily slot reserved")
            p.handoff("boss", "outcomes", "Paper Signal Ticket")
            return

        if message.startswith("No strategy passed"):
            p.system(status="ONLINE", headline="Company analyzed market — no trade", market_open=True)
            p.employees(_ANALYSIS_IDS, "done", "Analysis complete — no qualifying setup")
            p.employee("selector", "done", "No strategy cleared selection gates", direction="HOLD")
            p.boss_decision("HOLD", reason="No strategy passed research + context + risk thresholds")
            p.handoff("selector", "boss", "HOLD Recommendation")


_handler_lock = threading.Lock()
_handler_installed = False


def install_dashboard_logging() -> DashboardLogHandler | None:
    """Attach one fail-open dashboard handler to the runtime logger.

    Set ``XAU_DASHBOARD_ENABLED=0`` to disable telemetry completely.
    """

    global _handler_installed
    if os.getenv("XAU_DASHBOARD_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    with _handler_lock:
        if _handler_installed:
            logger = logging.getLogger("xau-company")
            for handler in logger.handlers:
                if isinstance(handler, DashboardLogHandler):
                    return handler
            return None
        handler = DashboardLogHandler()
        logging.getLogger("xau-company").addHandler(handler)
        _handler_installed = True
        return handler
