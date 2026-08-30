from __future__ import annotations

import requests

from .models import TradeSignal


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, timeout: int = 15) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout

    def format_signal(self, s: TradeSignal) -> str:
        why = "\n".join(f"• {r}" for r in s.reasons)
        strategy = s.selected_strategy or "not reported"
        stats = s.strategy_stats or {}
        valid = stats.get("valid_hit_rate")
        trades = stats.get("trades")
        validation_line = ""
        if isinstance(valid, (int, float)) and trades is not None:
            validation_line = f"Validation: {valid:.1%} over {trades} historical signals\n"
        return (
            f"XAU COMPANY SIGNAL\n"
            f"Symbol: {s.symbol}\n"
            f"Action: {s.direction.value}\n"
            f"Strategy: {strategy}\n"
            f"{validation_line}"
            f"Entry: {s.entry:.2f}\n"
            f"TP: {s.take_profit:.2f}\n"
            f"SL: {s.stop_loss:.2f}\n"
            f"Selection confidence: {s.confidence:.1%}\n"
            f"Regime: {s.regime}\n"
            f"R:R: {s.risk_reward:.2f}\n\n"
            f"Why this strategy now:\n{why}\n\n"
            "Research signal only; live fills/slippage can differ."
        )

    def send(self, signal: TradeSignal) -> None:
        if not self.bot_token or not self.chat_id:
            raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required to send")
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        response = requests.post(
            url,
            json={"chat_id": self.chat_id, "text": self.format_signal(signal)},
            timeout=self.timeout,
        )
        response.raise_for_status()
