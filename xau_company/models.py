from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class AgentVote:
    agent: str
    direction: Direction
    confidence: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeSignal:
    symbol: str
    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float
    confidence: float
    regime: str
    reasons: list[str]
    votes: list[AgentVote]

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry - self.stop_loss)
        reward = abs(self.take_profit - self.entry)
        return reward / risk if risk else 0.0
