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
        valid = stats.get("walk_forward_hit_rate", stats.get("valid_hit_rate"))
        trades = stats.get("trades")
        folds = stats.get("folds")
        pf = stats.get("profit_factor")
        avg_r = stats.get("avg_r_multiple")
        max_dd = stats.get("max_drawdown_r")
        streak = stats.get("max_loss_streak")

        validation_line = ""
        if isinstance(valid, (int, float)) and trades is not None:
            fold_text = f" / {folds} walk-forward folds" if folds else ""
            validation_line = f"OOS validation: {valid:.1%} over {trades} executed trades{fold_text}\n"
        pf_line = f"Profit factor: {pf:.2f}\n" if isinstance(pf, (int, float)) else ""
        lifecycle_line = ""
        if isinstance(avg_r, (int, float)) and isinstance(max_dd, (int, float)):
            streak_text = f" / worst streak {streak}" if isinstance(streak, int) else ""
            lifecycle_line = f"Avg R: {avg_r:+.2f} / Max DD: {max_dd:.2f}R{streak_text}\n"

        return (
            f"XAU COMPANY SIGNAL\n"
            f"Symbol: {s.symbol}\n"
            f"Action: {s.direction.value}\n"
            f"Strategy: {strategy}\n"
            f"{validation_line}"
            f"{pf_line}"
            f"{lifecycle_line}"
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
