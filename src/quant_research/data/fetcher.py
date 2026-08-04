"""
Data fetcher module for downloading and caching NASDAQ market data.

This module provides the DataFetcher class for acquiring historical market data
via the yfinance API. Data is cached locally in Parquet format to avoid
repeated API calls.

Data Limitations:
    This module downloads OHLCV (Open, High, Low, Close, Volume) data ONLY.
    It does NOT provide:
    - Level II / order book data
    - Bid-ask spread information
    - Trade-level (tick) data
    - Order flow or market microstructure signals
    - Intraday bars (daily frequency only)

    All downstream analyses must account for these limitations.
    Volume data from exchanges may be incomplete or adjusted.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_TICKERS = ["QQQ", "^IXIC"]
DEFAULT_YEARS = 10
DEFAULT_TRAIN_RATIO = 0.8


class DataFetcher:
    """Fetches and caches NASDAQ market data from yfinance.

    Downloads daily OHLCV data for specified tickers (defaulting to QQQ
    as the primary NASDAQ proxy and ^IXIC as the NASDAQ Composite index).
    Data is cached locally in Parquet format for efficient re-use.

    Data available:
        - Open, High, Low, Close (adjusted prices)
        - Volume (daily trading volume)

    Data NOT available (OHLCV limitations):
        - Order book / Level II depth
        - Bid-ask spreads
        - Individual trade records
        - Order flow imbalance metrics

    Parameters
    ----------
    tickers : list[str], optional
        Tickers to download. Defaults to ["QQQ", "^IXIC"].
    years : int, optional
        Number of years of historical data. Defaults to 10.
    cache_dir : Path or str, optional
        Directory for caching Parquet files. Defaults to "data/cache".

    Examples
    --------
    >>> fetcher = DataFetcher()
    >>> data = fetcher.fetch("QQQ")
    >>> train, test = fetcher.train_test_split(data)
    """

    def __init__(
        self,
        tickers: Optional[list[str]] = None,
        years: int = DEFAULT_YEARS,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
    ) -> None:
        self.tickers = tickers or DEFAULT_TICKERS
        self.years = years
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str) -> Path:
        """Get the cache file path for a given ticker.

        Parameters
        ----------
        ticker : str
            The ticker symbol.

        Returns
        -------
        Path
            Path to the cached Parquet file.
        """
        safe_name = ticker.replace("^", "").replace("/", "_")
        return self.cache_dir / f"{safe_name}.parquet"

    def fetch(
        self,
        ticker: Optional[str] = None,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a ticker, using cache if available.

        Downloads daily OHLCV data from yfinance for the specified ticker.
        If cached data exists and force_download is False, returns cached data.

        Note: This returns OHLCV + Volume data only. No order book,
        bid-ask, or tick-level data is available through this interface.

        Parameters
        ----------
        ticker : str, optional
            Ticker symbol to fetch. Defaults to first ticker in self.tickers.
        force_download : bool, optional
            If True, bypass cache and re-download. Defaults to False.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Open, High, Low, Close, Volume.
            Index is DatetimeIndex.

        Raises
        ------
        ValueError
            If downloaded data is empty or ticker is invalid.
        """
        if ticker is None:
            ticker = self.tickers[0]

        cache_path = self._cache_path(ticker)

        if not force_download and cache_path.exists():
            logger.info(f"Loading cached data for {ticker} from {cache_path}")
            df = pd.read_parquet(cache_path)
            return df

        logger.info(f"Downloading {self.years} years of data for {ticker}")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.years * 365)

        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}'. "
                "Check that the ticker symbol is valid."
            )

        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Ensure standard column names
        expected_cols = ["Open", "High", "Low", "Close", "Volume"]
        data = data[expected_cols]

        # Validate and clean
        data = self._validate_and_clean(data, ticker)

        # Cache to parquet
        data.to_parquet(cache_path)
        logger.info(f"Cached {len(data)} rows for {ticker} at {cache_path}")

        return data

    def fetch_all(self, force_download: bool = False) -> dict[str, pd.DataFrame]:
        """Fetch data for all configured tickers.

        Parameters
        ----------
        force_download : bool, optional
            If True, bypass cache. Defaults to False.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping ticker symbols to their OHLCV DataFrames.
        """
        results = {}
        for ticker in self.tickers:
            results[ticker] = self.fetch(ticker, force_download=force_download)
        return results

    def _validate_and_clean(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Validate data completeness and handle missing values.

        Parameters
        ----------
        data : pd.DataFrame
            Raw OHLCV DataFrame.
        ticker : str
            Ticker symbol for logging purposes.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with missing values handled.
        """
        initial_len = len(data)

        # Report missing values
        missing = data.isnull().sum()
        if missing.any():
            logger.warning(
                f"Missing values detected for {ticker}: "
                f"{missing[missing > 0].to_dict()}"
            )

        # Forward fill small gaps (up to 5 consecutive days, e.g. holidays)
        data = data.ffill(limit=5)

        # Drop any remaining rows with NaN
        data = data.dropna()

        final_len = len(data)
        if final_len < initial_len:
            dropped = initial_len - final_len
            logger.warning(
                f"Dropped {dropped} rows with missing data for {ticker} "
                f"({dropped/initial_len*100:.1f}%)"
            )

        # Validate no negative prices or volumes
        price_cols = ["Open", "High", "Low", "Close"]
        if (data[price_cols] < 0).any().any():
            logger.warning(f"Negative prices detected for {ticker}, filtering out")
            data = data[(data[price_cols] >= 0).all(axis=1)]

        if (data["Volume"] < 0).any():
            logger.warning(f"Negative volume detected for {ticker}, filtering out")
            data = data[data["Volume"] >= 0]

        return data

    def train_test_split(
        self,
        data: pd.DataFrame,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets chronologically.

        Uses a simple time-based split to avoid lookahead bias.
        The first train_ratio fraction of data (by time) is used for training,
        and the remainder for testing.

        Parameters
        ----------
        data : pd.DataFrame
            Full OHLCV DataFrame with DatetimeIndex.
        train_ratio : float, optional
            Fraction of data for training. Defaults to 0.8.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (train_data, test_data) tuple.

        Raises
        ------
        ValueError
            If train_ratio is not between 0 and 1, or if data is empty.
        """
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

        if data.empty:
            raise ValueError("Cannot split empty DataFrame")

        split_idx = int(len(data) * train_ratio)
        train = data.iloc[:split_idx].copy()
        test = data.iloc[split_idx:].copy()

        logger.info(
            f"Train/test split: {len(train)} train rows "
            f"({train.index[0].date()} to {train.index[-1].date()}), "
            f"{len(test)} test rows "
            f"({test.index[0].date()} to {test.index[-1].date()})"
        )

        return train, test
