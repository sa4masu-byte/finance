"""
Backtest validation with simulated realistic stock data
Tests with 3,000,000 JPY capital (margin trading)
"""
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backtesting.backtest_engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_realistic_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_price: float,
    volatility: float = 0.02,
    drift: float = 0.0003,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate realistic stock price data using geometric Brownian motion

    Args:
        symbol: Stock symbol for seed variation
        start_date: Start date
        end_date: End date
        initial_price: Starting price
        volatility: Daily volatility (std of returns)
        drift: Daily drift (mean return)
        seed: Random seed

    Returns:
        DataFrame with OHLCV data
    """
    if seed is None:
        seed = hash(symbol) % 10000

    np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    n_days = len(dates)

    # Generate daily returns using GBM
    daily_returns = np.random.normal(drift, volatility, n_days)

    # Calculate close prices
    close = initial_price * np.cumprod(1 + daily_returns)

    # Generate OHLC with realistic intraday variation
    intraday_vol = volatility * 0.5
    high = close * (1 + np.abs(np.random.normal(0, intraday_vol, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, intraday_vol, n_days)))
    open_price = close * (1 + np.random.normal(0, intraday_vol * 0.5, n_days))

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    # Generate volume with some randomness and trend
    base_volume = np.random.randint(500000, 2000000)
    volume = base_volume * (1 + np.random.normal(0, 0.3, n_days))
    volume = np.maximum(volume, 100000).astype(int)

    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=dates)

    return df


def create_stock_universe() -> dict:
    """Create simulated stock data for Japanese stocks"""

    # Japanese major stocks with realistic initial prices
    stocks = {
        "7203.T": {"name": "Toyota", "price": 2500, "vol": 0.018, "drift": 0.0004},
        "7267.T": {"name": "Honda", "price": 1400, "vol": 0.020, "drift": 0.0003},
        "6758.T": {"name": "Sony", "price": 12000, "vol": 0.022, "drift": 0.0005},
        "6861.T": {"name": "Keyence", "price": 55000, "vol": 0.019, "drift": 0.0004},
        "6981.T": {"name": "Murata", "price": 8000, "vol": 0.021, "drift": 0.0003},
        "8306.T": {"name": "MUFG", "price": 1200, "vol": 0.017, "drift": 0.0003},
        "8316.T": {"name": "SMFG", "price": 6000, "vol": 0.018, "drift": 0.0003},
        "9984.T": {"name": "SoftBank", "price": 6500, "vol": 0.028, "drift": 0.0002},
        "4063.T": {"name": "Shin-Etsu", "price": 4500, "vol": 0.019, "drift": 0.0004},
        "9432.T": {"name": "NTT", "price": 170, "vol": 0.015, "drift": 0.0002},
        "9983.T": {"name": "Fast Retailing", "price": 32000, "vol": 0.023, "drift": 0.0004},
        "7974.T": {"name": "Nintendo", "price": 6000, "vol": 0.022, "drift": 0.0003},
        "4502.T": {"name": "Takeda", "price": 4000, "vol": 0.018, "drift": 0.0002},
        "4188.T": {"name": "Mitsubishi Chem", "price": 800, "vol": 0.020, "drift": 0.0002},
        "6098.T": {"name": "Recruit", "price": 5000, "vol": 0.024, "drift": 0.0004},
    }

    stock_data = {}
    for symbol, info in stocks.items():
        df = generate_realistic_stock_data(
            symbol=symbol,
            start_date="2022-01-01",
            end_date="2024-12-31",
            initial_price=info["price"],
            volatility=info["vol"],
            drift=info["drift"],
        )
        stock_data[symbol] = df
        logger.info(f"Generated {symbol} ({info['name']}): {len(df)} days, "
                   f"Price range: {df['Close'].min():.0f} - {df['Close'].max():.0f}")

    return stock_data


def run_backtest_validation():
    """Run backtest with 3M JPY capital using simulated data"""

    INITIAL_CAPITAL = 3_000_000  # 300万円（信用取引）

    logger.info("=" * 70)
    logger.info("Swing Trading System - Backtest Validation (Simulated Data)")
    logger.info("=" * 70)
    logger.info(f"Initial Capital: {INITIAL_CAPITAL:,} JPY (Margin Trading)")

    # Generate stock data
    logger.info("\n[1/2] Generating simulated stock data...")
    stock_data = create_stock_universe()
    logger.info(f"Generated data for {len(stock_data)} stocks")

    # Run backtest
    logger.info("\n[2/2] Running backtest...")

    test_periods = [
        ("2022-01-01", "2022-12-31", "2022"),
        ("2023-01-01", "2023-12-31", "2023"),
        ("2024-01-01", "2024-12-31", "2024"),
        ("2022-01-01", "2024-12-31", "Full Period (2022-2024)"),
    ]

    all_results = []

    for start_date, end_date, period_name in test_periods:
        logger.info(f"\n{'='*50}")
        logger.info(f"Period: {period_name}")
        logger.info(f"{'='*50}")

        engine = BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            max_positions=5,
            commission_rate=0.001,   # 0.1%
            slippage_rate=0.0005,    # 0.05%
        )

        results = engine.run_backtest(stock_data, start_date, end_date)

        all_results.append({
            "period": period_name,
            "results": results,
        })

        profit_loss = results.final_capital - INITIAL_CAPITAL

        logger.info(f"\nResults:")
        logger.info(f"  Total Trades:     {results.num_trades}")
        logger.info(f"  Winning Trades:   {len(results.winning_trades)}")
        logger.info(f"  Losing Trades:    {len(results.losing_trades)}")
        logger.info(f"  Win Rate:         {results.win_rate:.1%}")
        logger.info(f"  ")
        logger.info(f"  Initial Capital:  {INITIAL_CAPITAL:>12,} JPY")
        logger.info(f"  Final Capital:    {results.final_capital:>12,.0f} JPY")
        logger.info(f"  Profit/Loss:      {profit_loss:>+12,.0f} JPY")
        logger.info(f"  Total Return:     {results.total_return_pct:>+11.2f}%")
        logger.info(f"  ")
        logger.info(f"  Profit Factor:    {results.profit_factor:.2f}")
        logger.info(f"  Sharpe Ratio:     {results.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown:     {results.max_drawdown:.2f}%")
        logger.info(f"  Avg Holding:      {results.avg_holding_days:.1f} days")

        if results.num_trades > 0:
            logger.info(f"  Avg Win:          {results.avg_win:+.2f}%")
            logger.info(f"  Avg Loss:         {results.avg_loss:.2f}%")

    # Final Summary
    full_period = all_results[-1]["results"]
    profit_loss = full_period.final_capital - INITIAL_CAPITAL

    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY - Full Period (2022-2024)")
    logger.info("=" * 70)

    logger.info(f"""
    Initial Capital:     {INITIAL_CAPITAL:>12,} JPY
    Final Capital:       {full_period.final_capital:>12,.0f} JPY
    ─────────────────────────────────────────
    Total P&L:           {profit_loss:>+12,.0f} JPY
    Total Return:        {full_period.total_return_pct:>+11.2f}%
    CAGR (3 years):      {((full_period.final_capital / INITIAL_CAPITAL) ** (1/3) - 1) * 100:>+11.2f}%

    Performance Metrics:
    ─────────────────────────────────────────
    Win Rate:            {full_period.win_rate:>11.1%}
    Profit Factor:       {full_period.profit_factor:>11.2f}
    Sharpe Ratio:        {full_period.sharpe_ratio:>11.2f}
    Max Drawdown:        {full_period.max_drawdown:>11.2f}%
    Total Trades:        {full_period.num_trades:>11}
    Avg Holding Days:    {full_period.avg_holding_days:>11.1f}
    """)

    # Risk-adjusted returns
    if full_period.max_drawdown > 0:
        return_to_drawdown = full_period.total_return_pct / full_period.max_drawdown
        logger.info(f"    Return/Max DD:       {return_to_drawdown:>11.2f}")

    logger.info("=" * 70)

    return all_results


if __name__ == "__main__":
    run_backtest_validation()
