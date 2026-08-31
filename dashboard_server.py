from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from xau_company.dashboard import default_state


ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT / "dashboard"


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _read_state(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(state, dict):
            return state
    except (OSError, ValueError, TypeError):
        pass
    return default_state()


def _read_performance(db_path: Path) -> dict[str, Any] | None:
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
        conn.row_factory = sqlite3.Row
        columns = {row[1] for row in conn.execute("PRAGMA table_info(outcomes)").fetchall()}
        if not columns:
            conn.close()
            return None

        signal_count = int(conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0])
        rows = conn.execute(
            """
            SELECT observed_at, direction, entry, sl, tp, calibrated_confidence,
                   strategy, regime, status, delivery_state, telegram_message_id
            FROM outcomes
            ORDER BY observed_at DESC
            LIMIT 8
            """
        ).fetchall()

        resolved = conn.execute(
            """
            SELECT status, calibrated_confidence
            FROM outcomes
            WHERE delivery_state='SENT' AND status IN ('WIN', 'LOSS')
            """
        ).fetchall()
        conn.close()

        wins = sum(1 for row in resolved if row["status"] == "WIN")
        losses = sum(1 for row in resolved if row["status"] == "LOSS")
        total = wins + losses
        brier_values = []
        for row in resolved:
            probability = _float(row["calibrated_confidence"])
            if probability is None:
                continue
            outcome = 1.0 if row["status"] == "WIN" else 0.0
            brier_values.append((probability - outcome) ** 2)

        recent = []
        for row in rows:
            recent.append(
                {
                    "observed_at": row["observed_at"],
                    "direction": row["direction"],
                    "entry": _float(row["entry"]),
                    "stop_loss": _float(row["sl"]),
                    "take_profit": _float(row["tp"]),
                    "confidence": _float(row["calibrated_confidence"]),
                    "strategy": row["strategy"],
                    "regime": row["regime"],
                    "status": row["status"],
                    "delivery_state": row["delivery_state"],
                    "message_id": row["telegram_message_id"],
                }
            )

        return {
            "signals": signal_count,
            "resolved": total,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / total) if total else None,
            "brier": (sum(brier_values) / len(brier_values)) if brier_values else None,
            "recent_signals": recent,
        }
    except (sqlite3.Error, OSError, KeyError):
        return None


def build_state(state_path: Path, db_path: Path) -> dict[str, Any]:
    state = _read_state(state_path)
    performance = _read_performance(db_path)
    if performance is not None:
        state["performance"] = performance
        recent = performance.get("recent_signals") or []
        if recent:
            latest = recent[0]
            delivery = str(latest.get("delivery_state") or "UNKNOWN").upper()
            if str(latest.get("message_id") or "").lower() == "paper":
                delivery = "PAPER"
            state.setdefault("system", {})["telegram"] = delivery
            boss = state.setdefault("boss", {})
            if boss.get("entry") is None and latest.get("entry") is not None:
                boss.update(
                    {
                        "decision": latest.get("direction") or boss.get("decision", "HOLD"),
                        "strategy": latest.get("strategy") or boss.get("strategy", ""),
                        "confidence": latest.get("confidence"),
                        "entry": latest.get("entry"),
                        "stop_loss": latest.get("stop_loss"),
                        "take_profit": latest.get("take_profit"),
                        "regime": latest.get("regime"),
                    }
                )
    state["served_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return state


class DashboardHandler(SimpleHTTPRequestHandler):
    server_version = "XAUCompanyDashboard/1.0"

    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, directory=directory or str(DASHBOARD_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            state = build_state(self.server.state_path, self.server.db_path)  # type: ignore[attr-defined]
            self._json(state)
            return
        if parsed.path == "/api/health":
            self._json({"ok": True, "service": "xau-dashboard"})
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def _json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} - {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the XAU/USD animated company dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--state", default="data/dashboard_state.json")
    parser.add_argument("--db", default="data/xau_outcomes.sqlite3")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.state_path = Path(args.state)  # type: ignore[attr-defined]
    server.db_path = Path(args.db)  # type: ignore[attr-defined]
    print(f"XAU/USD company dashboard: http://{args.host}:{args.port}")
    print(f"State source: {server.state_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
