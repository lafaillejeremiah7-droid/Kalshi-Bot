from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    twelve_data_api_key: str = os.getenv("TWELVE_DATA_API_KEY", "")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    symbol: str = os.getenv("SYMBOL", "XAU/USD")
    interval: str = os.getenv("INTERVAL", "5min")
    output_size: int = int(os.getenv("OUTPUT_SIZE", "3000"))
    poll_seconds: int = int(os.getenv("POLL_SECONDS", "60"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.72"))
    min_consensus: int = int(os.getenv("MIN_CONSENSUS", "3"))
    max_candidates: int = int(os.getenv("MAX_CANDIDATES", "3000"))
    research_every_cycles: int = int(os.getenv("RESEARCH_EVERY_CYCLES", "60"))
    spread_bps: float = float(os.getenv("SPREAD_BPS", "1.5"))
    paper_mode: bool = _bool("PAPER_MODE", True)

    def validate(self) -> None:
        if not self.twelve_data_api_key:
            raise RuntimeError("TWELVE_DATA_API_KEY is required")
        if not 0.5 <= self.min_confidence <= 0.99:
            raise ValueError("MIN_CONFIDENCE must be between 0.50 and 0.99")
        if self.output_size < 300:
            raise ValueError("OUTPUT_SIZE must be at least 300")
