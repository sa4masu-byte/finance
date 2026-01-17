"""
Stock data fetcher using yfinance for Japanese stocks
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

import pandas as pd
import yfinance as yf
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import DATA_FETCH_CONFIG, CACHE_DIR

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Fetches historical stock data from Yahoo Finance for Japanese stocks
    Supports caching to reduce API calls
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_dir: Path = CACHE_DIR,
        max_retries: int = DATA_FETCH_CONFIG["max_retries"],
    ):
        """
        Initialize data fetcher

        Args:
            cache_enabled: Enable local caching
            cache_dir: Directory for cache files
            max_retries: Maximum number of retry attempts
        """
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 60.0 / DATA_FETCH_CONFIG["requests_per_minute"]

    def fetch_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single stock

        Args:
            symbol: Stock symbol (e.g., "7203.T" for Toyota)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
            Returns None if fetch fails
        """
        # Check cache first
        if self.cache_enabled and not force_refresh:
            cached_data = self._load_from_cache(symbol, start_date, end_date)
            if cached_data is not None:
                logger.info(f"Loaded {symbol} from cache")
                return cached_data

        # Fetch from Yahoo Finance
        logger.info(f"Fetching {symbol} from {start_date} to {end_date}")

        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                self._rate_limit()

                # Fetch data
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return None

                # Validate data
                if not self._validate_data(df):
                    logger.warning(f"Data validation failed for {symbol}")
                    return None

                # Cache the data
                if self.cache_enabled:
                    self._save_to_cache(df, symbol, start_date, end_date)

                logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
                return df

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(DATA_FETCH_CONFIG["retry_delay"] * (attempt + 1))
                else:
                    logger.error(f"Failed to fetch {symbol} after {self.max_retries} attempts")
                    return None

    def fetch_multiple_stocks(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: Bypass cache

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}

        for symbol in symbols:
            df = self.fetch_stock_data(symbol, start_date, end_date, force_refresh)
            if df is not None:
                results[symbol] = df
            else:
                logger.warning(f"Skipping {symbol} due to fetch failure")

        logger.info(f"Successfully fetched {len(results)}/{len(symbols)} stocks")
        return results

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate fetched data

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        # Check required columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
            return False

        # Check for null values in critical columns
        critical_nulls = df[required_columns].isnull().sum().sum()
        if critical_nulls > len(df) * 0.1:  # More than 10% nulls
            logger.error(f"Too many null values: {critical_nulls}/{len(df)}")
            return False

        # Check for negative prices
        if (df["Close"] <= 0).any():
            logger.error("Found non-positive closing prices")
            return False

        return True

    def _get_cache_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        """Generate cache file path"""
        safe_symbol = symbol.replace(".", "_")
        filename = f"{safe_symbol}_{start_date}_{end_date}.parquet"
        return self.cache_dir / filename

    def _save_to_cache(
        self, df: pd.DataFrame, symbol: str, start_date: str, end_date: str
    ):
        """Save DataFrame to cache"""
        try:
            cache_path = self._get_cache_path(symbol, start_date, end_date)
            df.to_parquet(cache_path, compression="gzip")
            logger.debug(f"Saved {symbol} to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {symbol}: {e}")

    def _load_from_cache(
        self, symbol: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache"""
        try:
            cache_path = self._get_cache_path(symbol, start_date, end_date)

            if not cache_path.exists():
                return None

            # Check cache age
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            max_age = timedelta(hours=DATA_FETCH_CONFIG["cache_expiry_hours"])

            if cache_age > max_age:
                logger.debug(f"Cache expired for {symbol} (age: {cache_age})")
                return None

            df = pd.read_parquet(cache_path)
            logger.debug(f"Loaded {symbol} from cache: {cache_path}")
            return df

        except Exception as e:
            logger.warning(f"Failed to load cache for {symbol}: {e}")
            return None

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the most recent closing price for a stock

        Args:
            symbol: Stock symbol

        Returns:
            Latest closing price or None
        """
        try:
            ticker = yf.Ticker(symbol)
            # Get last 5 days to ensure we have recent data
            df = ticker.history(period="5d")
            if not df.empty:
                return float(df["Close"].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {e}")
            return None


# Convenience function
def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Convenience function to fetch stock data

    Args:
        symbol: Stock symbol (e.g., "7203.T")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    fetcher = StockDataFetcher()
    return fetcher.fetch_stock_data(symbol, start_date, end_date)
