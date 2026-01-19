"""
Fetch real historical data for Japanese stocks
Run this on your local machine where network access is available

Usage:
    pip install yfinance
    python scripts/fetch_real_data.py
"""
import sys
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Japanese stock tickers (Tokyo Stock Exchange)
STOCK_LIST = {
    # Auto
    "7203.T": "Toyota",
    "7267.T": "Honda",
    "7201.T": "Nissan",
    # Tech
    "6758.T": "Sony",
    "6861.T": "Keyence",
    "6981.T": "Murata",
    "6501.T": "Hitachi",
    "6702.T": "Fujitsu",
    # Finance
    "8306.T": "MUFG",
    "8316.T": "SMFG",
    "8411.T": "Mizuho",
    # Telecom
    "9432.T": "NTT",
    "9433.T": "KDDI",
    "9984.T": "SoftBank Group",
    # Others
    "4063.T": "Shin-Etsu Chemical",
    "9983.T": "Fast Retailing",
    "7974.T": "Nintendo",
    "4502.T": "Takeda",
    "6098.T": "Recruit",
    "6367.T": "Daikin",
}


def fetch_with_yfinance(symbols: list, start_date: str, end_date: str) -> dict:
    """Fetch data using yfinance"""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return {}

    stock_data = {}

    for symbol in symbols:
        try:
            logger.info(f"Fetching {symbol} ({STOCK_LIST.get(symbol, 'Unknown')})...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue

            # Rename columns to match our format
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume',
            })

            # Keep only OHLCV
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            stock_data[symbol] = df
            logger.info(f"  -> {len(df)} days of data")

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")

    return stock_data


def fetch_with_pandas_datareader(symbols: list, start_date: str, end_date: str) -> dict:
    """Fetch data using pandas-datareader (Stooq)"""
    try:
        import pandas_datareader as pdr
    except ImportError:
        logger.error("pandas-datareader not installed. Run: pip install pandas-datareader")
        return {}

    stock_data = {}

    for symbol in symbols:
        try:
            # Convert .T to .JP for Stooq
            stooq_symbol = symbol.replace('.T', '.JP')
            logger.info(f"Fetching {stooq_symbol} ({STOCK_LIST.get(symbol, 'Unknown')})...")

            df = pdr.get_data_stooq(stooq_symbol, start_date, end_date)

            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue

            # Stooq returns data in reverse order
            df = df.sort_index()

            stock_data[symbol] = df
            logger.info(f"  -> {len(df)} days of data")

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")

    return stock_data


def save_data(stock_data: dict, output_dir: Path):
    """Save fetched data to CSV files"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol, df in stock_data.items():
        filename = output_dir / f"{symbol.replace('.', '_')}.csv"
        df.to_csv(filename)
        logger.info(f"Saved {filename}")

    # Save metadata
    metadata = {
        "fetch_date": datetime.now().isoformat(),
        "symbols": list(stock_data.keys()),
        "date_range": {
            symbol: {
                "start": str(df.index.min()),
                "end": str(df.index.max()),
                "rows": len(df),
            }
            for symbol, df in stock_data.items()
        }
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {output_dir / 'metadata.json'}")


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "historical"

    symbols = list(STOCK_LIST.keys())
    start_date = "2022-01-01"
    end_date = "2024-12-31"

    logger.info(f"Fetching {len(symbols)} stocks from {start_date} to {end_date}")
    logger.info("=" * 60)

    # Try yfinance first
    logger.info("Attempting yfinance...")
    stock_data = fetch_with_yfinance(symbols, start_date, end_date)

    if not stock_data:
        # Fallback to pandas-datareader
        logger.info("Falling back to pandas-datareader (Stooq)...")
        stock_data = fetch_with_pandas_datareader(symbols, start_date, end_date)

    if not stock_data:
        logger.error("Failed to fetch any data. Check your network connection.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"Successfully fetched {len(stock_data)} stocks")

    # Save data
    save_data(stock_data, output_dir)

    logger.info("=" * 60)
    logger.info("Done! Run backtest with:")
    logger.info("  python scripts/validate_with_real_data.py")


if __name__ == "__main__":
    main()
