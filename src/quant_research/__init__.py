"""
Quantitative Research Pipeline for NASDAQ Market Analysis.

This package provides a modular pipeline for:
- Data acquisition and feature engineering (OHLCV data)
- Hypothesis generation and statistical testing
- Walk-forward validation and out-of-sample testing
- Strategy design with entries, exits, and position sizing
- Robustness testing including transaction costs and regime analysis
- Comprehensive reporting

Data Limitations:
    This pipeline operates on OHLCV (Open, High, Low, Close, Volume) data only.
    It does NOT have access to:
    - Level II / order book data
    - Bid-ask spread information
    - Trade-level (tick) data
    - Order flow or market microstructure data

    All analyses and features are derived solely from daily OHLCV bars.
"""

__version__ = "0.1.0"
