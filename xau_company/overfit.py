from __future__ import annotations

from dataclasses import dataclass
from math import log10

from .research import CandidateScore


@dataclass(frozen=True)
class OverfitAuditResult:
    passed: bool
    adjusted_score: float
    multiplicity_penalty: float
    tested_trials: int
    reasons: tuple[str, ...]


class OverfitAuditor:
    """Conservative promotion gate for discovered/evolved strategies."""

    name = "Overfit Auditor"

    def __init__(
        self,
        min_adjusted_score: float = 0.60,
        min_profit_factor: float = 1.15,
        min_avg_r: float = 0.05,
        min_trades: int = 40,
        max_walk_forward_std: float = 0.12,
        max_train_valid_gap: float = 0.15,
        max_drawdown_r: float = 10.0,
        max_loss_streak: int = 7,
        min_folds: int = 3,
        penalty_per_decade: float = 0.025,
        max_multiplicity_penalty: float = 0.18,
    ) -> None:
        self.min_adjusted_score = float(min_adjusted_score)
        self.min_profit_factor = float(min_profit_factor)
        self.min_avg_r = float(min_avg_r)
        self.min_trades = max(1, int(min_trades))
        self.max_walk_forward_std = max(0.0, float(max_walk_forward_std))
        self.max_train_valid_gap = max(0.0, float(max_train_valid_gap))
        self.max_drawdown_r = max(0.0, float(max_drawdown_r))
        self.max_loss_streak = max(1, int(max_loss_streak))
        self.min_folds = max(1, int(min_folds))
        self.penalty_per_decade = max(0.0, float(penalty_per_decade))
        self.max_multiplicity_penalty = max(0.0, float(max_multiplicity_penalty))

    @staticmethod
    def oos_trade_count(score: CandidateScore) -> int:
        regime_total = sum(max(0, int(n)) for n in score.regime_trades.values())
        return regime_total if regime_total > 0 else max(0, int(score.trades))

    def multiplicity_penalty(self, tested_trials: int) -> float:
        trials = max(1, int(tested_trials))
        penalty = self.penalty_per_decade * log10(max(10, trials))
        return min(self.max_multiplicity_penalty, max(0.0, penalty))

    def audit(self, score: CandidateScore, tested_trials: int) -> OverfitAuditResult:
        penalty = self.multiplicity_penalty(tested_trials)
        adjusted = max(0.0, float(score.score) - penalty)
        reasons: list[str] = []

        if adjusted < self.min_adjusted_score:
            reasons.append(
                f"multiplicity-adjusted score {adjusted:.3f} < {self.min_adjusted_score:.3f}"
            )
        if float(score.profit_factor) < self.min_profit_factor:
            reasons.append(
                f"profit factor {score.profit_factor:.2f} < {self.min_profit_factor:.2f}"
            )
        if float(score.avg_r_multiple) < self.min_avg_r:
            reasons.append(f"average R {score.avg_r_multiple:.3f} < {self.min_avg_r:.3f}")
        oos_trades = self.oos_trade_count(score)
        if oos_trades < self.min_trades:
            reasons.append(f"OOS trade sample {oos_trades} < {self.min_trades}")
        if float(score.walk_forward_std) > self.max_walk_forward_std:
            reasons.append(
                f"walk-forward dispersion {score.walk_forward_std:.3f} > {self.max_walk_forward_std:.3f}"
            )
        train_valid_gap = abs(float(score.train_hit_rate) - float(score.valid_hit_rate))
        if train_valid_gap > self.max_train_valid_gap:
            reasons.append(
                f"train/OOS gap {train_valid_gap:.3f} > {self.max_train_valid_gap:.3f}"
            )
        if float(score.max_drawdown_r) > self.max_drawdown_r:
            reasons.append(
                f"max drawdown {score.max_drawdown_r:.2f}R > {self.max_drawdown_r:.2f}R"
            )
        if int(score.max_loss_streak) > self.max_loss_streak:
            reasons.append(f"loss streak {score.max_loss_streak} > {self.max_loss_streak}")
        if int(score.folds) < self.min_folds:
            reasons.append(f"walk-forward folds {score.folds} < {self.min_folds}")

        return OverfitAuditResult(
            passed=not reasons,
            adjusted_score=adjusted,
            multiplicity_penalty=penalty,
            tested_trials=max(1, int(tested_trials)),
            reasons=tuple(reasons),
        )
