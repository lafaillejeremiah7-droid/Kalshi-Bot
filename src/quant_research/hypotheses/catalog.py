"""
Hypothesis catalog data structures for quantitative research.

Defines the Hypothesis dataclass and HypothesisCategory enum used to
represent and organize market behavior hypotheses.

Data Limitations:
    All hypotheses operate on OHLCV-derived features only. Signals that
    attempt to proxy order flow, microstructure, or intraday behavior are
    limited by the granularity of daily bar data and exchange-reported volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import pandas as pd


class HypothesisCategory(Enum):
    """Categories for market behavior hypotheses."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    GAPS = "gaps"
    SESSION_EFFECTS = "session_effects"
    ORDER_FLOW_PROXY = "order_flow_proxy"
    REGIME = "regime"
    CROSS_ASSET = "cross_asset"
    MICROSTRUCTURE_PROXY = "microstructure_proxy"


@dataclass
class Hypothesis:
    """A testable market behavior hypothesis.

    Attributes
    ----------
    id : str
        Unique identifier (e.g., MOM_001, MR_001, VOL_001).
    category : HypothesisCategory
        The category this hypothesis belongs to.
    name : str
        Short descriptive name.
    description : str
        Detailed description of what the hypothesis tests.
    economic_rationale : str
        Explanation of WHY the edge might persist in markets.
    data_requirements : list[str]
        List of columns/features required from the input DataFrame.
    signal_function : Callable[[pd.DataFrame], pd.Series]
        Function that takes a DataFrame with OHLCV+features and returns
        a signal Series. Signal values should be in {-1, 0, 1} for
        discrete signals or [-1, 1] for continuous signals.
    expected_direction : int
        Expected profitable direction: 1 for long bias, -1 for short bias,
        0 for direction-neutral.
    data_limitations : str
        Explicit documentation of when the hypothesis requires data not
        available in OHLCV, or notes about proxy quality.
    """

    id: str
    category: HypothesisCategory
    name: str
    description: str
    economic_rationale: str
    data_requirements: list[str]
    signal_function: Callable[[pd.DataFrame], pd.Series]
    expected_direction: int
    data_limitations: str
