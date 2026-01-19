"""
Backtest validation script for swing trading system
Tests with 3,000,000 JPY capital (margin trading)
"""
import sys
from pathlib import Path
import json
import logging

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import BACKTEST_CONFIG
from src.data.fetcher import StockDataFetcher
from backtesting.backtest_engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_stock_universe():
    """Load stock universe from JSON"""
    universe_file = project_root / "config" / "stock_universe.json"
    with open(universe_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["test_stocks"].keys())


def run_backtest_validation():
    """Run backtest with 3M JPY capital"""

    # Configuration
    INITIAL_CAPITAL = 3_000_000  # 300万円（信用取引）

    logger.info("=" * 70)
    logger.info("Swing Trading System - Backtest Validation")
    logger.info("=" * 70)
    logger.info(f"Initial Capital: {INITIAL_CAPITAL:,} JPY (Margin Trading)")

    # Load stocks
    logger.info("\n[1/3] Loading stock universe...")
    symbols = load_stock_universe()
    logger.info(f"Target stocks: {len(symbols)} stocks")

    # Fetch data
    logger.info("\n[2/3] Fetching historical data...")
    fetcher = StockDataFetcher(cache_enabled=True)

    stock_data = fetcher.fetch_multiple_stocks(
        symbols=symbols,
        start_date="2022-01-01",
        end_date="2025-12-31",
        force_refresh=False,
    )

    logger.info(f"Successfully fetched: {len(stock_data)}/{len(symbols)} stocks")

    if len(stock_data) < 3:
        logger.error("Insufficient data for backtest")
        return None

    # Run backtest
    logger.info("\n[3/3] Running backtest...")

    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        max_positions=5,
        commission_rate=0.001,  # 0.1%
        slippage_rate=0.0005,   # 0.05%
    )

    # Test periods
    test_periods = [
        ("2022-01-01", "2022-12-31", "2022"),
        ("2023-01-01", "2023-12-31", "2023"),
        ("2024-01-01", "2024-12-31", "2024"),
        ("2022-01-01", "2024-12-31", "Full Period (2022-2024)"),
    ]

    all_results = []

    for start_date, end_date, period_name in test_periods:
        logger.info(f"\n--- {period_name} ---")

        # Reset engine
        engine = BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            max_positions=5,
            commission_rate=0.001,
            slippage_rate=0.0005,
        )

        results = engine.run_backtest(stock_data, start_date, end_date)

        all_results.append({
            "period": period_name,
            "results": results,
        })

        # Print results
        logger.info(f"  Trades:          {results.num_trades}")
        logger.info(f"  Win Rate:        {results.win_rate:.1%}")
        logger.info(f"  Total Return:    {results.total_return_pct:+.2f}%")
        logger.info(f"  Final Capital:   {results.final_capital:,.0f} JPY")
        logger.info(f"  Profit/Loss:     {results.final_capital - INITIAL_CAPITAL:+,.0f} JPY")
        logger.info(f"  Profit Factor:   {results.profit_factor:.2f}")
        logger.info(f"  Sharpe Ratio:    {results.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown:    {results.max_drawdown:.2f}%")
        logger.info(f"  Avg Holding:     {results.avg_holding_days:.1f} days")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    full_period = all_results[-1]["results"]

    logger.info(f"\nFull Period Performance (2022-2024):")
    logger.info(f"  Initial Capital:  {INITIAL_CAPITAL:,} JPY")
    logger.info(f"  Final Capital:    {full_period.final_capital:,.0f} JPY")
    logger.info(f"  Total P&L:        {full_period.final_capital - INITIAL_CAPITAL:+,.0f} JPY")
    logger.info(f"  Total Return:     {full_period.total_return_pct:+.2f}%")
    logger.info(f"  CAGR:             {((full_period.final_capital / INITIAL_CAPITAL) ** (1/3) - 1) * 100:.2f}%")
    logger.info(f"  Win Rate:         {full_period.win_rate:.1%}")
    logger.info(f"  Profit Factor:    {full_period.profit_factor:.2f}")
    logger.info(f"  Sharpe Ratio:     {full_period.sharpe_ratio:.2f}")
    logger.info(f"  Max Drawdown:     {full_period.max_drawdown:.2f}%")
    logger.info(f"  Total Trades:     {full_period.num_trades}")

    # Trade breakdown
    if full_period.num_trades > 0:
        logger.info(f"\nTrade Breakdown:")
        logger.info(f"  Winning Trades:   {len(full_period.winning_trades)}")
        logger.info(f"  Losing Trades:    {len(full_period.losing_trades)}")
        logger.info(f"  Avg Win:          {full_period.avg_win:+.2f}%")
        logger.info(f"  Avg Loss:         {full_period.avg_loss:.2f}%")

    logger.info("\n" + "=" * 70)

    return all_results


if __name__ == "__main__":
    run_backtest_validation()
