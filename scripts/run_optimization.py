"""
Run weight optimization on historical data
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import BACKTEST_CONFIG, OPTIMIZATION_CONFIG, REPORTS_DIR
from src.data.fetcher import StockDataFetcher
from backtesting.optimizer import WeightOptimizer
from backtesting.backtest_engine import BacktestEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_stock_universe():
    """Load stock universe from JSON"""
    universe_file = project_root / "config" / "stock_universe.json"
    with open(universe_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["test_stocks"].keys())


def main():
    """Main optimization workflow"""
    logger.info("="*70)
    logger.info("Japanese Stock Trading Recommender - Weight Optimization")
    logger.info("="*70)

    # 1. Load stock universe
    logger.info("\n[1/5] Loading stock universe...")
    symbols = load_stock_universe()
    logger.info(f"Loaded {len(symbols)} stocks: {symbols[:5]}...")

    # 2. Fetch historical data
    logger.info("\n[2/5] Fetching historical data...")
    logger.info(f"Date range: {BACKTEST_CONFIG['start_date']} to {BACKTEST_CONFIG['end_date']}")

    fetcher = StockDataFetcher(cache_enabled=True)
    stock_data = fetcher.fetch_multiple_stocks(
        symbols=symbols,
        start_date=BACKTEST_CONFIG["start_date"],
        end_date=BACKTEST_CONFIG["end_date"],
        force_refresh=False,
    )

    logger.info(f"Successfully fetched data for {len(stock_data)}/{len(symbols)} stocks")

    if len(stock_data) < 5:
        logger.error("Insufficient data. Need at least 5 stocks.")
        return

    # 3. Define walk-forward periods
    logger.info("\n[3/5] Setting up walk-forward analysis periods...")

    periods = [
        # Period 1: 2022-01 to 2023-06 (train) | 2023-07 to 2023-12 (test)
        ("2022-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),
        # Period 2: 2023-01 to 2024-06 (train) | 2024-07 to 2024-12 (test)
        ("2023-01-01", "2024-06-30", "2024-07-01", "2024-12-31"),
        # Period 3: 2024-01 to 2025-06 (train) | 2025-07 to 2025-12 (test)
        ("2024-01-01", "2025-06-30", "2025-07-01", "2025-12-31"),
    ]

    logger.info(f"Walk-forward periods: {len(periods)}")
    for i, (ts, te, vs, ve) in enumerate(periods):
        logger.info(f"  Period {i+1}: Train={ts}~{te}, Test={vs}~{ve}")

    # 4. Run optimization (use first period for optimization)
    logger.info("\n[4/5] Running weight optimization...")
    train_start, train_end, test_start, test_end = periods[0]

    optimizer = WeightOptimizer(
        stock_data=stock_data,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    # Grid search
    logger.info("\nPhase 1: Grid Search")
    grid_results = optimizer.grid_search()

    logger.info(f"\nTop 5 weight combinations from grid search:")
    for i, (weights, score) in enumerate(grid_results[:5]):
        logger.info(f"  #{i+1} Score={score:.2f}")
        logger.info(f"      Trend={weights['trend']:.2f}, "
                    f"Momentum={weights['momentum']:.2f}, "
                    f"Volume={weights['volume']:.2f}, "
                    f"Volatility={weights['volatility']:.2f}, "
                    f"Pattern={weights['pattern']:.2f}")

    # Bayesian optimization
    logger.info("\nPhase 2: Bayesian Optimization")
    try:
        best_weights, best_score = optimizer.bayesian_optimization(grid_results)
        logger.info(f"\nOptimized weights (score={best_score:.2f}):")
        for key, value in best_weights.items():
            logger.info(f"  {key.capitalize()}: {value:.3f} ({value*100:.1f}%)")
    except ImportError:
        logger.warning("Bayesian optimization requires scikit-optimize. Using grid search result.")
        best_weights, best_score = grid_results[0]

    # 5. Validate on all walk-forward periods
    logger.info("\n[5/5] Walk-forward validation...")

    wf_results = optimizer.walk_forward_analysis(best_weights, periods)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("OPTIMIZATION RESULTS SUMMARY")
    logger.info("="*70)

    logger.info("\nOptimal Weights:")
    for key, value in best_weights.items():
        logger.info(f"  {key.capitalize():12s}: {value:.3f} ({value*100:.1f}%)")

    logger.info("\nWalk-Forward Performance:")
    for i, results in enumerate(wf_results):
        logger.info(f"\n  Period {i+1}:")
        logger.info(f"    Win Rate:        {results.win_rate:.1%}")
        logger.info(f"    Total Return:    {results.total_return_pct:+.2f}%")
        logger.info(f"    Profit Factor:   {results.profit_factor:.2f}")
        logger.info(f"    Sharpe Ratio:    {results.sharpe_ratio:.2f}")
        logger.info(f"    Max Drawdown:    {results.max_drawdown:.2f}%")
        logger.info(f"    Avg Holding:     {results.avg_holding_days:.1f} days")
        logger.info(f"    # Trades:        {results.num_trades}")

    # Average performance
    avg_win_rate = sum(r.win_rate for r in wf_results) / len(wf_results)
    avg_return = sum(r.total_return_pct for r in wf_results) / len(wf_results)
    avg_sharpe = sum(r.sharpe_ratio for r in wf_results) / len(wf_results)
    avg_pf = sum(r.profit_factor for r in wf_results) / len(wf_results)

    logger.info(f"\n  Average Across All Periods:")
    logger.info(f"    Win Rate:        {avg_win_rate:.1%}")
    logger.info(f"    Avg Return:      {avg_return:+.2f}%")
    logger.info(f"    Avg Sharpe:      {avg_sharpe:.2f}")
    logger.info(f"    Avg PF:          {avg_pf:.2f}")

    # Success criteria check
    logger.info("\nSuccess Criteria Check:")
    criteria_met = []
    criteria_met.append(("Win Rate > 60%", avg_win_rate > 0.60, f"{avg_win_rate:.1%}"))
    criteria_met.append(("Profit Factor > 2.0", avg_pf > 2.0, f"{avg_pf:.2f}"))
    criteria_met.append(("Sharpe Ratio > 1.5", avg_sharpe > 1.5, f"{avg_sharpe:.2f}"))

    for criterion, met, value in criteria_met:
        status = "✓" if met else "✗"
        logger.info(f"  {status} {criterion}: {value}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = REPORTS_DIR / f"optimization_results_{timestamp}.json"

    optimizer.save_results(results_file, best_weights, grid_results)

    # Save best weights to a separate file for easy loading
    best_weights_file = REPORTS_DIR / "best_weights.json"
    with open(best_weights_file, "w") as f:
        json.dump({
            "weights": best_weights,
            "score": best_score,
            "timestamp": timestamp,
            "validation_performance": {
                "avg_win_rate": avg_win_rate,
                "avg_return": avg_return,
                "avg_sharpe": avg_sharpe,
                "avg_profit_factor": avg_pf,
            }
        }, f, indent=2)

    logger.info(f"\nResults saved to:")
    logger.info(f"  {results_file}")
    logger.info(f"  {best_weights_file}")

    logger.info("\n" + "="*70)
    logger.info("Optimization complete!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
