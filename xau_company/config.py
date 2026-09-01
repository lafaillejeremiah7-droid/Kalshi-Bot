from __future__ import annotations

import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
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
    walk_forward_folds: int = int(os.getenv("WALK_FORWARD_FOLDS", "4"))
    min_walk_forward_folds: int = int(os.getenv("MIN_WALK_FORWARD_FOLDS", "2"))
    research_every_cycles: int = int(os.getenv("RESEARCH_EVERY_CYCLES", "60"))

    overfit_min_adjusted_score: float = float(os.getenv("OVERFIT_MIN_ADJUSTED_SCORE", "0.60"))
    overfit_min_profit_factor: float = float(os.getenv("OVERFIT_MIN_PROFIT_FACTOR", "1.15"))
    overfit_min_avg_r: float = float(os.getenv("OVERFIT_MIN_AVG_R", "0.05"))
    overfit_min_trades: int = int(os.getenv("OVERFIT_MIN_TRADES", "40"))
    overfit_max_walk_forward_std: float = float(os.getenv("OVERFIT_MAX_WF_STD", "0.12"))
    overfit_max_train_valid_gap: float = float(os.getenv("OVERFIT_MAX_TRAIN_VALID_GAP", "0.15"))
    overfit_max_drawdown_r: float = float(os.getenv("OVERFIT_MAX_DRAWDOWN_R", "10.0"))
    overfit_max_loss_streak: int = int(os.getenv("OVERFIT_MAX_LOSS_STREAK", "7"))

    spread_bps: float = float(os.getenv("SPREAD_BPS", "1.5"))
    slippage_bps: float = float(os.getenv("SLIPPAGE_BPS", "0.5"))
    backtest_stop_atr: float = float(os.getenv("BACKTEST_STOP_ATR", "1.20"))
    backtest_reward_risk: float = float(os.getenv("BACKTEST_REWARD_RISK", "1.70"))

    outcome_db_path: str = os.getenv("OUTCOME_DB_PATH", "data/xau_outcomes.sqlite3")
    outcome_max_age_hours: int = int(os.getenv("OUTCOME_MAX_AGE_HOURS", "72"))
    calibration_bin_width: float = float(os.getenv("CALIBRATION_BIN_WIDTH", "0.05"))
    calibration_prior_strength: float = float(os.getenv("CALIBRATION_PRIOR_STRENGTH", "20"))

    trade_timezone: str = os.getenv("TRADE_TIMEZONE", "America/Chicago")
    max_trades_per_day: int = int(os.getenv("MAX_TRADES_PER_DAY", "2"))
    max_stale_multiplier: float = float(os.getenv("MAX_STALE_MULTIPLIER", "4.0"))
    max_signal_delay_minutes: int = int(os.getenv("MAX_SIGNAL_DELAY_MINUTES", "5"))
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
        allowed = {"1min", "5min", "15min", "30min", "45min", "1h", "2h", "4h", "8h", "1day"}
        if not self.paper_mode and (not self.telegram_bot_token or not self.telegram_chat_id):
            raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required when PAPER_MODE=false")
        if not 0.5 <= self.min_confidence <= 0.99:
            raise ValueError("MIN_CONFIDENCE must be between 0.50 and 0.99")
        if not 1 <= self.min_consensus <= 6:
            raise ValueError("MIN_CONSENSUS must be between 1 and 6")
        if self.output_size < 300:
            raise ValueError("OUTPUT_SIZE must be at least 300")
        if self.context_output_size < 220:
            raise ValueError("CONTEXT_OUTPUT_SIZE must be at least 220")
        if self.poll_seconds < 10:
            raise ValueError("POLL_SECONDS must be at least 10")
        if self.research_every_cycles < 1:
            raise ValueError("RESEARCH_EVERY_CYCLES must be at least 1")
        if self.walk_forward_folds < 2:
            raise ValueError("WALK_FORWARD_FOLDS must be at least 2")
        if not 2 <= self.min_walk_forward_folds <= self.walk_forward_folds:
            raise ValueError("MIN_WALK_FORWARD_FOLDS must be between 2 and WALK_FORWARD_FOLDS")

        if not 0.0 <= self.overfit_min_adjusted_score <= 1.0:
            raise ValueError("OVERFIT_MIN_ADJUSTED_SCORE must be between 0 and 1")
        if self.overfit_min_profit_factor < 1.0:
            raise ValueError("OVERFIT_MIN_PROFIT_FACTOR must be at least 1.0")
        if self.overfit_min_avg_r < 0:
            raise ValueError("OVERFIT_MIN_AVG_R cannot be negative")
        if self.overfit_min_trades < 20:
            raise ValueError("OVERFIT_MIN_TRADES must be at least 20")
        if not 0.0 <= self.overfit_max_walk_forward_std <= 1.0:
            raise ValueError("OVERFIT_MAX_WF_STD must be between 0 and 1")
        if not 0.0 <= self.overfit_max_train_valid_gap <= 1.0:
            raise ValueError("OVERFIT_MAX_TRAIN_VALID_GAP must be between 0 and 1")
        if self.overfit_max_drawdown_r <= 0:
            raise ValueError("OVERFIT_MAX_DRAWDOWN_R must be positive")
        if self.overfit_max_loss_streak < 1:
            raise ValueError("OVERFIT_MAX_LOSS_STREAK must be at least 1")

        if self.spread_bps < 0 or self.slippage_bps < 0:
            raise ValueError("SPREAD_BPS and SLIPPAGE_BPS cannot be negative")
        if self.backtest_stop_atr <= 0:
            raise ValueError("BACKTEST_STOP_ATR must be positive")
        if self.backtest_reward_risk <= 0:
            raise ValueError("BACKTEST_REWARD_RISK must be positive")
        if not 1 <= self.outcome_max_age_hours <= 72:
            raise ValueError("OUTCOME_MAX_AGE_HOURS must be between 1 and 72")
        if not 0.02 <= self.calibration_bin_width <= 0.20:
            raise ValueError("CALIBRATION_BIN_WIDTH must be between 0.02 and 0.20")
        if self.calibration_prior_strength < 5:
            raise ValueError("CALIBRATION_PRIOR_STRENGTH must be at least 5")
        if not 1 <= self.max_trades_per_day <= 2:
            raise ValueError("MAX_TRADES_PER_DAY must be 1 or 2")
        if self.max_stale_multiplier < 1.5:
            raise ValueError("MAX_STALE_MULTIPLIER must be at least 1.5")
        if not 1 <= self.max_signal_delay_minutes <= 15:
            raise ValueError("MAX_SIGNAL_DELAY_MINUTES must be between 1 and 15")

        try:
            ZoneInfo(self.trade_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown TRADE_TIMEZONE: {self.trade_timezone}") from exc

        invalid = set(self.timeframes) - allowed
        if invalid:
            raise ValueError(f"Unsupported TIMEFRAMES: {sorted(invalid)}")
        if self.research_interval not in allowed or self.research_interval == "1day":
            raise ValueError("RESEARCH_INTERVAL must be an intraday supported interval up to 8h")
        if self.macro_interval not in allowed:
            raise ValueError("Unsupported MACRO_INTERVAL")
        required = {self.research_interval, "1h", "4h"}
        missing = required - set(self.timeframes)
        if missing:
            raise ValueError(f"TIMEFRAMES must include research/1h/4h context: {sorted(missing)}")
        if not {"1min", "5min"}.intersection(self.timeframes):
            raise ValueError("TIMEFRAMES must include 1min or 5min for execution")
