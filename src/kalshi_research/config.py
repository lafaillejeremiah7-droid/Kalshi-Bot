from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    kalshi_env: str = "production"
    kalshi_series_ticker: str = "KXBTC15M"
    kalshi_rest_base: str = "https://external-api.kalshi.com/trade-api/v2"
    kalshi_ws_url: str = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
    kalshi_api_key_id: str | None = None
    kalshi_private_key_path: Path | None = None
    research_db_path: Path = Path("data/research.sqlite3")
    raw_capture_dir: Path = Path("data/raw")
    report_archive_dir: Path = Path("data/experiments")

    @classmethod
    def from_env(cls) -> "ResearchConfig":
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        return cls(
            kalshi_env=os.getenv("KALSHI_ENV", "production"),
            kalshi_series_ticker=os.getenv("KALSHI_SERIES_TICKER", "KXBTC15M"),
            kalshi_rest_base=os.getenv(
                "KALSHI_REST_BASE", "https://external-api.kalshi.com/trade-api/v2"
            ).rstrip("/"),
            kalshi_ws_url=os.getenv(
                "KALSHI_WS_URL", "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
            ),
            kalshi_api_key_id=os.getenv("KALSHI_API_KEY_ID"),
            kalshi_private_key_path=Path(key_path) if key_path else None,
            research_db_path=Path(os.getenv("RESEARCH_DB_PATH", "data/research.sqlite3")),
            raw_capture_dir=Path(os.getenv("RAW_CAPTURE_DIR", "data/raw")),
            report_archive_dir=Path(
                os.getenv("REPORT_ARCHIVE_DIR", "data/experiments")
            ),
        )

    def ensure_research_dirs(self) -> None:
        self.research_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_capture_dir.mkdir(parents=True, exist_ok=True)
        self.report_archive_dir.mkdir(parents=True, exist_ok=True)
