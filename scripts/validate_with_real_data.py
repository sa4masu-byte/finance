"""
Validate backtest with real historical data

Run this after fetching data with fetch_real_data.py:
    python scripts/fetch_real_data.py
    python scripts/validate_with_real_data.py
"""
import sys
from pathlib import Path
import logging
import json
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backtesting.backtest_engine import BacktestEngine
from backtesting.enhanced_backtest_engine import EnhancedBacktestEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_historical_data(data_dir: Path) -> dict:
    """Load historical data from CSV files"""
    stock_data = {}

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Run 'python scripts/fetch_real_data.py' first")
        return {}

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}")
        return {}

    for csv_file in csv_files:
        if csv_file.name == "metadata.json":
            continue

        symbol = csv_file.stem.replace("_", ".")
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            stock_data[symbol] = df
            logger.info(f"Loaded {symbol}: {len(df)} days")
        except Exception as e:
            logger.error(f"Failed to load {csv_file}: {e}")

    return stock_data


def infer_sector_map(symbols: list) -> dict:
    """Infer sector map from symbols"""
    sector_map = {}
    for symbol in symbols:
        code = symbol.split(".")[0]
        if code.startswith("7"):
            sector_map[symbol] = "auto"
        elif code.startswith("6"):
            sector_map[symbol] = "tech"
        elif code.startswith("8"):
            sector_map[symbol] = "finance"
        elif code.startswith("9"):
            sector_map[symbol] = "telecom"
        elif code.startswith("4"):
            sector_map[symbol] = "pharma"
        else:
            sector_map[symbol] = "other"
    return sector_map


def run_validation():
    """Run validation with real data"""
    INITIAL_CAPITAL = 3_000_000

    data_dir = project_root / "data" / "historical"

    logger.info("=" * 70)
    logger.info("Backtest Validation with REAL Historical Data")
    logger.info("=" * 70)
    logger.info(f"Initial Capital: {INITIAL_CAPITAL:,} JPY")

    # Load data
    logger.info("\n[1/4] Loading historical data...")
    stock_data = load_historical_data(data_dir)

    if not stock_data:
        logger.error("No data available. Exiting.")
        return

    logger.info(f"Loaded {len(stock_data)} stocks")
    sector_map = infer_sector_map(list(stock_data.keys()))

    # Determine date range
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index)
    min_date = min(all_dates)
    max_date = max(all_dates)
    logger.info(f"Date range: {min_date.date()} to {max_date.date()}")

    start_date = str(min_date.date())
    end_date = str(max_date.date())

    # Run original backtest
    logger.info("\n[2/4] Running ORIGINAL backtest...")
    original_engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        max_positions=5,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )
    original_results = original_engine.run_backtest(stock_data, start_date, end_date)

    # Run enhanced backtest
    logger.info("\n[3/4] Running ENHANCED backtest...")
    enhanced_engine = EnhancedBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        max_positions=5,
        commission_rate=0.001,
        slippage_rate=0.0005,
        enable_trailing_stop=True,
        enable_volatility_sizing=True,
        enable_market_regime=True,
        enable_multi_timeframe=True,
        enable_volume_breakout=True,
        enable_swing_low_stop=True,
        enable_additional_filters=True,
        enable_compound=True,
        min_score=55,
        min_confidence=0.60,
    )
    enhanced_results = enhanced_engine.run_backtest(stock_data, start_date, end_date, sector_map=sector_map)

    # Results comparison
    logger.info("\n[4/4] RESULTS COMPARISON (REAL DATA)")
    logger.info("=" * 70)

    years = (max_date - min_date).days / 365.25
    original_cagr = ((original_results.final_capital / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0
    enhanced_cagr = ((enhanced_results.final_capital / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0

    logger.info(f"""
{'='*70}
                        ORIGINAL        ENHANCED        IMPROVEMENT
{'='*70}
Total Return:           {original_results.total_return_pct:+8.2f}%       {enhanced_results.total_return_pct:+8.2f}%       {enhanced_results.total_return_pct - original_results.total_return_pct:+8.2f}%
CAGR ({years:.1f} years):       {original_cagr:+8.2f}%       {enhanced_cagr:+8.2f}%       {enhanced_cagr - original_cagr:+8.2f}%
{'-'*70}
Win Rate:               {original_results.win_rate:8.1%}       {enhanced_results.win_rate:8.1%}       {(enhanced_results.win_rate - original_results.win_rate)*100:+8.1f}pp
Profit Factor:          {original_results.profit_factor:8.2f}        {enhanced_results.profit_factor:8.2f}        {enhanced_results.profit_factor - original_results.profit_factor:+8.2f}
Sharpe Ratio:           {original_results.sharpe_ratio:8.2f}        {enhanced_results.sharpe_ratio:8.2f}        {enhanced_results.sharpe_ratio - original_results.sharpe_ratio:+8.2f}
Max Drawdown:           {original_results.max_drawdown:8.2f}%       {enhanced_results.max_drawdown:8.2f}%       {original_results.max_drawdown - enhanced_results.max_drawdown:+8.2f}%
{'-'*70}
Total Trades:           {original_results.num_trades:8d}        {enhanced_results.num_trades:8d}        {enhanced_results.num_trades - original_results.num_trades:+8d}
Avg Win:                {original_results.avg_win:+8.2f}%       {enhanced_results.avg_win:+8.2f}%
Avg Loss:               {original_results.avg_loss:8.2f}%       {enhanced_results.avg_loss:8.2f}%
{'='*70}
    """)

    # Success criteria
    logger.info("\nSUCCESS CRITERIA CHECK (Enhanced):")
    logger.info("-" * 50)

    criteria = [
        ("Win Rate > 50%", enhanced_results.win_rate > 0.50, f"{enhanced_results.win_rate:.1%}"),
        ("Profit Factor > 1.5", enhanced_results.profit_factor > 1.5, f"{enhanced_results.profit_factor:.2f}"),
        ("Sharpe Ratio > 0.5", enhanced_results.sharpe_ratio > 0.5, f"{enhanced_results.sharpe_ratio:.2f}"),
        ("Max Drawdown < 20%", enhanced_results.max_drawdown < 20, f"{enhanced_results.max_drawdown:.2f}%"),
        ("Positive CAGR", enhanced_cagr > 0, f"{enhanced_cagr:+.2f}%"),
        ("CAGR > 10%", enhanced_cagr > 10, f"{enhanced_cagr:+.2f}%"),
    ]

    passed = 0
    for name, met, value in criteria:
        status = "[PASS]" if met else "[FAIL]"
        if met:
            passed += 1
        logger.info(f"  {status} {name}: {value}")

    logger.info(f"\nResult: {passed}/{len(criteria)} criteria passed")
    logger.info("=" * 70)

    # Recommendation
    if passed >= 4:
        logger.info("\n[RECOMMENDATION] Results look promising for paper trading.")
    elif passed >= 2:
        logger.info("\n[RECOMMENDATION] Needs more optimization. Do not use real money yet.")
    else:
        logger.info("\n[RECOMMENDATION] Strategy needs significant improvement.")


if __name__ == "__main__":
    run_validation()
