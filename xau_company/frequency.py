from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class FrequencyDecision:
    allowed: bool
    reason: str
    trades_today: int
    remaining_today: int
    local_date: str


class TradeFrequencyGuard:
    """Allow setup-dependent trading on weekdays with a strict daily cap.

    The guard never forces a trade. It only authorizes an otherwise-qualified
    setup when the local trading day is Monday-Friday and the configured daily
    maximum has not already been reached.
    """

    name = "Trade Frequency Guard"

    def __init__(self, timezone_name: str = "America/Chicago", max_trades_per_day: int = 2) -> None:
        self.timezone_name = timezone_name
        self.timezone = ZoneInfo(timezone_name)
        self.max_trades_per_day = max(1, min(2, int(max_trades_per_day)))

    def day_bounds_utc(self, now: datetime) -> tuple[datetime, datetime]:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        local_now = now.astimezone(self.timezone)
        start_local = datetime.combine(local_now.date(), time.min, tzinfo=self.timezone)
        end_local = start_local + timedelta(days=1)
        return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)

    def evaluate(self, now: datetime, trades_today: int) -> FrequencyDecision:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        local_now = now.astimezone(self.timezone)
        count = max(0, int(trades_today))
        remaining = max(0, self.max_trades_per_day - count)

        if local_now.weekday() >= 5:
            return FrequencyDecision(
                allowed=False,
                reason="weekend: trading is limited to Monday-Friday",
                trades_today=count,
                remaining_today=0,
                local_date=local_now.date().isoformat(),
            )

        if count >= self.max_trades_per_day:
            return FrequencyDecision(
                allowed=False,
                reason=f"daily trade cap reached ({self.max_trades_per_day})",
                trades_today=count,
                remaining_today=0,
                local_date=local_now.date().isoformat(),
            )

        return FrequencyDecision(
            allowed=True,
            reason="qualified setup is within weekday daily trade limit",
            trades_today=count,
            remaining_today=remaining,
            local_date=local_now.date().isoformat(),
        )
