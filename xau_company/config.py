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
    research_interval: str = os.getenv("RESEARCH_INTERVAL", "15min")
    timeframe_csv: str = os.getenv("TIMEFRAMES", "1min,5min,15min,1h,4h")
    output_size: int = int(os.getenv("OUTPUT_SIZE", "3000"))
    context_output_size: int = int(os.getenv("CONTEXT_OUTPUT_SIZE", "500"))
    poll_seconds: int = int(os.getenv("POLL_SECONDS", "60"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.72"))
    min_consensus: int = int(os.getenv("MIN_CONSENSUS", "3"))
    max_candidates: int = int(os.getenv("MAX_CANDIDATES", "20000"))
    research_catalog_size: int = int(os.getenv("RESEARCH_CATALOG_SIZE", "600"))
    walk_forward_folds: int = int(os.getenv("WALK_FORWARD_FOLDS", "4"))
    min_walk_forward_folds: int = int(os.getenv("MIN_WALK_FORWARD_FOLDS", "2"))
    research_every_cycles: int = int(os.getenv("RESEARCH_EVERY_CYCLES", "60"))
    spread_bps: float = float(os.getenv("SPREAD_BPS", "1.5"))
    paper_mode: bool = _bool("PAPER_MODE", True)
    dxy_symbol: str = os.getenv("DXY_SYMBOL", "DXY")
    yield_symbol: str = os.getenv("YIELD_SYMBOL", "US10Y")
    macro_interval: str = os.getenv("MACRO_INTERVAL", "1h")
    high_impact_events_utc: str = os.getenv("HIGH_IMPACT_EVENTS_UTC", "")
    news_block_minutes: int = int(os.getenv("NEWS_BLOCK_MINUTES", "20"))

    @property
    def timeframes(self) -> tuple[str, ...]:
        return tuple(x.strip() for x in self.timeframe_csv.split(",") if x.strip())

    def validate(self) -> None:
        if not self.twelve_data_api_key:
            raise RuntimeError("TWELVE_DATA_API_KEY is required")
        if not 0.5 <= self.min_confidence <= 0.99:
            raise ValueError("MIN_CONFIDENCE must be between 0.50 and 0.99")
        if self.output_size < 300:
            raise ValueError("OUTPUT_SIZE must be at least 300")
        if self.max_candidates < 1000:
            raise ValueError("MAX_CANDIDATES must be at least 1000")
        if self.research_catalog_size < 50:
            raise ValueError("RESEARCH_CATALOG_SIZE must be at least 50")
        if self.walk_forward_folds < 2:
            raise ValueError("WALK_FORWARD_FOLDS must be at least 2")
        allowed = {"1min", "5min", "15min", "30min", "45min", "1h", "2h", "4h", "8h", "1day"}
        invalid = set(self.timeframes) - allowed
        if invalid:
            raise ValueError(f"Unsupported TIMEFRAMES: {sorted(invalid)}")
        if self.research_interval not in allowed:
            raise ValueError("Unsupported RESEARCH_INTERVAL")
