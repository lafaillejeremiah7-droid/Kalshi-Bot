"""
Position sizing module implementing multiple sizing methodologies.

Provides fixed fractional, Kelly criterion, volatility-targeted, and
equal-risk position sizing approaches with maximum position limits.
"""

from __future__ import annotations

from dataclasses import dataclass


MAX_POSITION_FRACTION = 0.20  # No single position > 20% of equity


@dataclass
class PositionSizeResult:
    """Result of a position sizing computation.

    Attributes
    ----------
    position_size : float
        Computed position size (fraction of equity or units).
    method : str
        Method used for computation.
    capped : bool
        Whether the position was capped at maximum limit.
    raw_size : float
        Position size before any caps were applied.
    """

    position_size: float
    method: str
    capped: bool = False
    raw_size: float = 0.0


class PositionSizer:
    """Computes position sizes using multiple methodologies.

    Enforces maximum position limits: no single position may exceed 20%
    of account equity.

    Parameters
    ----------
    max_position_fraction : float, optional
        Maximum fraction of equity for any single position.
        Default is 0.20 (20%).

    Examples
    --------
    >>> sizer = PositionSizer()
    >>> size = sizer.fixed_fractional(100000, 0.01, 5.0)
    >>> kelly = sizer.kelly_criterion(0.55, 1.5, 1.0)
    """

    def __init__(
        self, max_position_fraction: float = MAX_POSITION_FRACTION
    ) -> None:
        self.max_position_fraction = max_position_fraction

    def fixed_fractional(
        self,
        account_equity: float,
        risk_per_trade: float,
        stop_distance: float,
    ) -> float:
        """Compute position size using fixed fractional method.

        Position size = (equity * risk_per_trade) / stop_distance
        Result is capped at max_position_fraction * equity.

        Parameters
        ----------
        account_equity : float
            Total account equity.
        risk_per_trade : float
            Fraction of equity to risk per trade (e.g., 0.01 = 1%).
        stop_distance : float
            Distance to stop-loss in price units.

        Returns
        -------
        float
            Position size in equity units (dollar value of position).
        """
        if stop_distance <= 0 or account_equity <= 0:
            return 0.0

        risk_amount = account_equity * risk_per_trade
        raw_size = risk_amount / stop_distance
        max_size = account_equity * self.max_position_fraction

        return min(raw_size, max_size)

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        half_kelly: bool = True,
    ) -> float:
        """Compute optimal fraction using Kelly criterion.

        Kelly fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) /
                         (avg_win * avg_loss)

        Simplified: f = p/a - q/b where p = win_rate, q = loss_rate,
        a = avg_loss, b = avg_win

        Parameters
        ----------
        win_rate : float
            Probability of winning (0 to 1).
        avg_win : float
            Average winning trade return (positive).
        avg_loss : float
            Average losing trade return (positive, represents loss magnitude).
        half_kelly : bool, optional
            If True, return half-Kelly for more conservative sizing.
            Default is True.

        Returns
        -------
        float
            Optimal fraction of capital to allocate (0 to max_position_fraction).
        """
        if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        loss_rate = 1.0 - win_rate
        # Kelly formula: f* = (p * b - q * a) / (a * b)
        # where p = win_rate, q = loss_rate, b = avg_win, a = avg_loss
        kelly_f = (win_rate * avg_win - loss_rate * avg_loss) / (
            avg_win * avg_loss
        )

        if kelly_f <= 0:
            return 0.0

        if half_kelly:
            kelly_f *= 0.5

        return min(kelly_f, self.max_position_fraction)

    def volatility_target(
        self,
        target_vol: float,
        current_vol: float,
        account_equity: float,
    ) -> float:
        """Compute position size to achieve target volatility.

        Position size = (target_vol / current_vol) * account_equity

        Parameters
        ----------
        target_vol : float
            Target annualized volatility (e.g., 0.15 for 15%).
        current_vol : float
            Current annualized volatility of the strategy.
        account_equity : float
            Total account equity.

        Returns
        -------
        float
            Position size in equity units.
        """
        if current_vol <= 0 or account_equity <= 0 or target_vol <= 0:
            return 0.0

        vol_ratio = target_vol / current_vol
        raw_size = vol_ratio * account_equity
        max_size = account_equity * self.max_position_fraction

        return min(raw_size, max_size)

    def equal_risk(
        self,
        n_strategies: int,
        max_portfolio_risk: float,
        strategy_vol: float,
    ) -> float:
        """Compute position size for equal risk contribution.

        Each strategy gets an equal share of the portfolio risk budget.
        Position size = (max_portfolio_risk / n_strategies) / strategy_vol

        Parameters
        ----------
        n_strategies : int
            Total number of strategies in portfolio.
        max_portfolio_risk : float
            Maximum portfolio annualized volatility target.
        strategy_vol : float
            Annualized volatility of this strategy.

        Returns
        -------
        float
            Position size as fraction of equity.
        """
        if n_strategies <= 0 or strategy_vol <= 0 or max_portfolio_risk <= 0:
            return 0.0

        risk_per_strategy = max_portfolio_risk / n_strategies
        raw_fraction = risk_per_strategy / strategy_vol

        return min(raw_fraction, self.max_position_fraction)

    def compute_position_size(
        self, method: str, params: dict
    ) -> float:
        """Compute position size using specified method.

        Parameters
        ----------
        method : str
            One of: 'fixed_fractional', 'kelly', 'volatility_target',
            'equal_risk'.
        params : dict
            Parameters for the chosen method.

        Returns
        -------
        float
            Computed position size.

        Raises
        ------
        ValueError
            If method is not recognized.
        """
        if method == "fixed_fractional":
            return self.fixed_fractional(
                account_equity=params.get("account_equity", 100000),
                risk_per_trade=params.get("risk_per_trade", 0.01),
                stop_distance=params.get("stop_distance", 1.0),
            )
        elif method == "kelly":
            return self.kelly_criterion(
                win_rate=params.get("win_rate", 0.5),
                avg_win=params.get("avg_win", 1.0),
                avg_loss=params.get("avg_loss", 1.0),
                half_kelly=params.get("half_kelly", True),
            )
        elif method == "volatility_target":
            return self.volatility_target(
                target_vol=params.get("target_vol", 0.15),
                current_vol=params.get("current_vol", 0.20),
                account_equity=params.get("account_equity", 100000),
            )
        elif method == "equal_risk":
            return self.equal_risk(
                n_strategies=params.get("n_strategies", 5),
                max_portfolio_risk=params.get("max_portfolio_risk", 0.15),
                strategy_vol=params.get("strategy_vol", 0.20),
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Must be one of: "
                "'fixed_fractional', 'kelly', 'volatility_target', 'equal_risk'"
            )
