"""
Validation script for enhanced backtest with all 10 improvements
Compares original vs enhanced performance
"""
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backtesting.backtest_engine import BacktestEngine
from backtesting.enhanced_backtest_engine import EnhancedBacktestEngine, MarketRegime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_market_regime_data(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_price: float,
    sector: str,
    seed: int = None,
) -> pd.DataFrame:
    """Generate stock data with realistic market characteristics"""
    if seed is None:
        seed = hash(symbol) % 10000
    np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)

    # Define market regimes based on actual 2022-2024 movements
    regime_params = []
    for date in dates:
        year = date.year
        month = date.month

        if year == 2022:
            if month <= 6:
                regime_params.append({"drift": -0.0003, "vol": 0.025, "trend_strength": 0.3})
            else:
                regime_params.append({"drift": 0.0004, "vol": 0.020, "trend_strength": 0.5})
        elif year == 2023:
            if month <= 3:
                regime_params.append({"drift": 0.0006, "vol": 0.018, "trend_strength": 0.6})
            elif month <= 9:
                regime_params.append({"drift": 0.0008, "vol": 0.016, "trend_strength": 0.7})
            else:
                regime_params.append({"drift": 0.0003, "vol": 0.017, "trend_strength": 0.5})
        else:  # 2024
            if month <= 3:
                regime_params.append({"drift": 0.0010, "vol": 0.015, "trend_strength": 0.75})
            elif month <= 8:
                regime_params.append({"drift": 0.0002, "vol": 0.020, "trend_strength": 0.4})
            else:
                regime_params.append({"drift": 0.0005, "vol": 0.018, "trend_strength": 0.55})

    # Sector adjustments
    sector_multipliers = {
        "auto": 1.3, "tech": 1.2, "finance": 1.4, "pharma": 0.9,
        "retail": 1.1, "telecom": 0.85, "materials": 1.15,
    }
    sector_mult = sector_multipliers.get(sector, 1.0)

    # Generate returns
    returns = []
    prev_return = 0
    cumulative_return = 0

    for i, params in enumerate(regime_params):
        drift = params["drift"] * sector_mult
        vol = params["vol"]
        trend_strength = params["trend_strength"]

        momentum_component = prev_return * trend_strength * 0.3

        mean_reversion = 0
        if cumulative_return > 0.15:
            mean_reversion = -0.001
        elif cumulative_return < -0.10:
            mean_reversion = 0.001

        random_component = np.random.normal(0, vol)
        daily_return = drift + momentum_component + mean_reversion + random_component

        if np.random.random() < 0.02:
            gap = np.random.choice([-1, 1]) * np.random.uniform(0.02, 0.05)
            daily_return += gap

        returns.append(daily_return)
        prev_return = daily_return
        cumulative_return += daily_return

    returns = np.array(returns)
    close = initial_price * np.cumprod(1 + returns)

    # Generate OHLC
    open_price = np.zeros(n_days)
    high = np.zeros(n_days)
    low = np.zeros(n_days)

    for i in range(n_days):
        if i == 0:
            open_price[i] = initial_price
        else:
            gap = np.random.normal(0, 0.003)
            open_price[i] = close[i-1] * (1 + gap)

        intraday_range = close[i] * np.random.uniform(0.01, 0.025)

        if close[i] > open_price[i]:
            low[i] = min(open_price[i], close[i]) - intraday_range * np.random.uniform(0.2, 0.5)
            high[i] = max(open_price[i], close[i]) + intraday_range * np.random.uniform(0.1, 0.3)
        else:
            high[i] = max(open_price[i], close[i]) + intraday_range * np.random.uniform(0.2, 0.5)
            low[i] = min(open_price[i], close[i]) - intraday_range * np.random.uniform(0.1, 0.3)

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    low = np.maximum(low, close * 0.9)

    base_volume = np.random.randint(800000, 2000000)
    price_change = np.abs(returns)
    volume_factor = 1 + price_change * 10
    volume = base_volume * volume_factor * (1 + np.random.normal(0, 0.2, n_days))
    volume = np.maximum(volume, 200000).astype(int)

    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=dates)


def create_stock_universe() -> tuple:
    """Create stock data with sector mapping"""
    stocks = {
        "7203.T": {"name": "Toyota", "price": 2200, "sector": "auto"},
        "7267.T": {"name": "Honda", "price": 3400, "sector": "auto"},
        "6758.T": {"name": "Sony", "price": 14000, "sector": "tech"},
        "6861.T": {"name": "Keyence", "price": 62000, "sector": "tech"},
        "6981.T": {"name": "Murata", "price": 9500, "sector": "tech"},
        "8306.T": {"name": "MUFG", "price": 650, "sector": "finance"},
        "8316.T": {"name": "SMFG", "price": 4000, "sector": "finance"},
        "9984.T": {"name": "SoftBank Group", "price": 5500, "sector": "tech"},
        "9432.T": {"name": "NTT", "price": 3200, "sector": "telecom"},
        "4063.T": {"name": "Shin-Etsu Chemical", "price": 19000, "sector": "materials"},
        "9983.T": {"name": "Fast Retailing", "price": 68000, "sector": "retail"},
        "7974.T": {"name": "Nintendo", "price": 57000, "sector": "retail"},
        "4502.T": {"name": "Takeda", "price": 3200, "sector": "pharma"},
        "4188.T": {"name": "Mitsubishi Chemical", "price": 850, "sector": "materials"},
        "6098.T": {"name": "Recruit", "price": 5800, "sector": "tech"},
    }

    stock_data = {}
    sector_map = {}

    for symbol, info in stocks.items():
        df = generate_market_regime_data(
            symbol=symbol,
            start_date="2022-01-01",
            end_date="2024-12-31",
            initial_price=info["price"],
            sector=info["sector"],
        )
        stock_data[symbol] = df
        sector_map[symbol] = info["sector"]

        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        total_return = (end_price / start_price - 1) * 100
        logger.info(f"{symbol} ({info['name']:15s}): {total_return:+6.1f}%")

    return stock_data, sector_map


def run_comparison():
    """Run comparison between original and enhanced backtest"""

    INITIAL_CAPITAL = 3_000_000

    logger.info("=" * 70)
    logger.info("Enhanced Backtest Validation - All 10 Improvements")
    logger.info("=" * 70)
    logger.info(f"Initial Capital: {INITIAL_CAPITAL:,} JPY")

    # Generate data
    logger.info("\n[1/4] Generating market data...")
    stock_data, sector_map = create_stock_universe()

    # Run original backtest
    logger.info("\n[2/4] Running ORIGINAL backtest...")
    original_engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        max_positions=5,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )
    original_results = original_engine.run_backtest(stock_data, "2022-01-01", "2024-12-31")

    # Run enhanced backtest
    logger.info("\n[3/4] Running ENHANCED backtest (all improvements enabled)...")
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
    enhanced_results = enhanced_engine.run_backtest(
        stock_data, "2022-01-01", "2024-12-31", sector_map=sector_map
    )

    # Compare results
    logger.info("\n[4/4] RESULTS COMPARISON")
    logger.info("=" * 70)

    original_pnl = original_results.final_capital - INITIAL_CAPITAL
    enhanced_pnl = enhanced_results.final_capital - INITIAL_CAPITAL
    original_cagr = ((original_results.final_capital / INITIAL_CAPITAL) ** (1/3) - 1) * 100
    enhanced_cagr = ((enhanced_results.final_capital / INITIAL_CAPITAL) ** (1/3) - 1) * 100

    logger.info(f"""
{'='*70}
                        ORIGINAL        ENHANCED        IMPROVEMENT
{'='*70}
Total Return:           {original_results.total_return_pct:+8.2f}%       {enhanced_results.total_return_pct:+8.2f}%       {enhanced_results.total_return_pct - original_results.total_return_pct:+8.2f}%
CAGR (3 years):         {original_cagr:+8.2f}%       {enhanced_cagr:+8.2f}%       {enhanced_cagr - original_cagr:+8.2f}%
Final P&L:              {original_pnl:+12,.0f}    {enhanced_pnl:+12,.0f}    {enhanced_pnl - original_pnl:+12,.0f}
{'-'*70}
Win Rate:               {original_results.win_rate:8.1%}       {enhanced_results.win_rate:8.1%}       {(enhanced_results.win_rate - original_results.win_rate)*100:+8.1f}pp
Profit Factor:          {original_results.profit_factor:8.2f}        {enhanced_results.profit_factor:8.2f}        {enhanced_results.profit_factor - original_results.profit_factor:+8.2f}
Sharpe Ratio:           {original_results.sharpe_ratio:8.2f}        {enhanced_results.sharpe_ratio:8.2f}        {enhanced_results.sharpe_ratio - original_results.sharpe_ratio:+8.2f}
Max Drawdown:           {original_results.max_drawdown:8.2f}%       {enhanced_results.max_drawdown:8.2f}%       {original_results.max_drawdown - enhanced_results.max_drawdown:+8.2f}%
{'-'*70}
Total Trades:           {original_results.num_trades:8d}        {enhanced_results.num_trades:8d}        {enhanced_results.num_trades - original_results.num_trades:+8d}
Avg Win:                {original_results.avg_win:+8.2f}%       {enhanced_results.avg_win:+8.2f}%       {enhanced_results.avg_win - original_results.avg_win:+8.2f}%
Avg Loss:               {original_results.avg_loss:8.2f}%       {enhanced_results.avg_loss:8.2f}%       {original_results.avg_loss - enhanced_results.avg_loss:+8.2f}%
Avg Holding Days:       {original_results.avg_holding_days:8.1f}        {enhanced_results.avg_holding_days:8.1f}        {enhanced_results.avg_holding_days - original_results.avg_holding_days:+8.1f}
{'='*70}
    """)

    # Analyze exit reasons for enhanced backtest
    exit_reasons = {}
    for trade in enhanced_results.trades:
        reason = trade.exit_reason
        if reason not in exit_reasons:
            exit_reasons[reason] = {"count": 0, "returns": []}
        exit_reasons[reason]["count"] += 1
        exit_reasons[reason]["returns"].append(trade.return_pct)

    logger.info("\nENHANCED BACKTEST - EXIT REASON ANALYSIS:")
    logger.info("-" * 50)
    for reason, data in sorted(exit_reasons.items(), key=lambda x: x[1]["count"], reverse=True):
        avg_return = np.mean(data["returns"]) if data["returns"] else 0
        win_rate = sum(1 for r in data["returns"] if r > 0) / len(data["returns"]) if data["returns"] else 0
        logger.info(f"  {reason:25s}: {data['count']:3d} trades | Avg: {avg_return:+6.2f}% | Win: {win_rate:5.1%}")

    # Analyze market regime performance
    if enhanced_results.regime_history:
        logger.info("\nMARKET REGIME DISTRIBUTION:")
        logger.info("-" * 50)
        regime_counts = {}
        for regime in enhanced_results.regime_history:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        total_days = len(enhanced_results.regime_history)
        for regime, count in sorted(regime_counts.items()):
            pct = count / total_days * 100
            logger.info(f"  {regime:20s}: {count:4d} days ({pct:5.1f}%)")

    # Year by year comparison
    logger.info("\nYEAR-BY-YEAR PERFORMANCE:")
    logger.info("-" * 70)
    logger.info(f"{'Year':<10} {'Original':>15} {'Enhanced':>15} {'Improvement':>15}")
    logger.info("-" * 70)

    for year in [2022, 2023, 2024]:
        # Original
        orig_engine = BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            max_positions=5,
            commission_rate=0.001,
            slippage_rate=0.0005,
        )
        orig_res = orig_engine.run_backtest(stock_data, f"{year}-01-01", f"{year}-12-31")

        # Enhanced
        enh_engine = EnhancedBacktestEngine(
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
        enh_res = enh_engine.run_backtest(
            stock_data, f"{year}-01-01", f"{year}-12-31", sector_map=sector_map
        )

        improvement = enh_res.total_return_pct - orig_res.total_return_pct
        logger.info(f"{year:<10} {orig_res.total_return_pct:+14.2f}% {enh_res.total_return_pct:+14.2f}% {improvement:+14.2f}%")

    # Success criteria check
    logger.info("\n" + "=" * 70)
    logger.info("SUCCESS CRITERIA CHECK (Enhanced):")
    logger.info("-" * 50)

    criteria = [
        ("Win Rate > 50%", enhanced_results.win_rate > 0.50, f"{enhanced_results.win_rate:.1%}"),
        ("Profit Factor > 1.5", enhanced_results.profit_factor > 1.5, f"{enhanced_results.profit_factor:.2f}"),
        ("Sharpe Ratio > 0.5", enhanced_results.sharpe_ratio > 0.5, f"{enhanced_results.sharpe_ratio:.2f}"),
        ("Max Drawdown < 20%", enhanced_results.max_drawdown < 20, f"{enhanced_results.max_drawdown:.2f}%"),
        ("Positive CAGR", enhanced_cagr > 0, f"{enhanced_cagr:+.2f}%"),
        ("CAGR > 10%", enhanced_cagr > 10, f"{enhanced_cagr:+.2f}%"),
        ("Total Return > 30%", enhanced_results.total_return_pct > 30, f"{enhanced_results.total_return_pct:+.2f}%"),
    ]

    passed = 0
    for name, met, value in criteria:
        status = "[PASS]" if met else "[FAIL]"
        if met:
            passed += 1
        logger.info(f"  {status} {name}: {value}")

    logger.info(f"\nResult: {passed}/{len(criteria)} criteria passed")
    logger.info("=" * 70)

    # Improvement summary
    logger.info("\n" + "=" * 70)
    logger.info("IMPROVEMENT SUMMARY (10 Enhancements):")
    logger.info("-" * 50)
    improvements = [
        ("1. Trailing Stop", "Locks in profits, improves avg win"),
        ("2. Volatility Position Sizing", "Risk-adjusted position sizes"),
        ("3. Sector Momentum", "Bonus for strong sectors"),
        ("4. Multi-Timeframe", "Weekly trend confirmation"),
        ("5. Volume Breakout", "Identifies institutional buying"),
        ("6. Swing Low Stop", "Smarter stop loss placement"),
        ("7. Market Regime", "Adjusts strategy to market conditions"),
        ("8. Additional Filters", "Reduces false signals"),
        ("9. Compound Interest", "Reinvests profits for growth"),
        ("10. ML Scoring", "Available but disabled by default"),
    ]
    for name, desc in improvements:
        logger.info(f"  {name:30s}: {desc}")

    logger.info("=" * 70)

    return original_results, enhanced_results


def run_ablation_study():
    """Run ablation study to measure impact of each improvement"""

    INITIAL_CAPITAL = 3_000_000

    logger.info("\n" + "=" * 70)
    logger.info("ABLATION STUDY - Impact of Each Improvement")
    logger.info("=" * 70)

    stock_data, sector_map = create_stock_universe()

    # Baseline (all disabled)
    logger.info("\nRunning baseline (all enhancements disabled)...")
    baseline_engine = EnhancedBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        max_positions=5,
        enable_trailing_stop=False,
        enable_volatility_sizing=False,
        enable_market_regime=False,
        enable_multi_timeframe=False,
        enable_volume_breakout=False,
        enable_swing_low_stop=False,
        enable_additional_filters=False,
        enable_compound=False,
        min_score=55,
        min_confidence=0.60,
    )
    baseline = baseline_engine.run_backtest(stock_data, "2022-01-01", "2024-12-31", sector_map=sector_map)
    baseline_return = baseline.total_return_pct

    # Test each improvement individually
    improvements = [
        ("Trailing Stop", {"enable_trailing_stop": True}),
        ("Volatility Sizing", {"enable_volatility_sizing": True}),
        ("Market Regime", {"enable_market_regime": True}),
        ("Multi-Timeframe", {"enable_multi_timeframe": True}),
        ("Volume Breakout", {"enable_volume_breakout": True}),
        ("Swing Low Stop", {"enable_swing_low_stop": True}),
        ("Additional Filters", {"enable_additional_filters": True}),
        ("Compound Interest", {"enable_compound": True}),
    ]

    logger.info(f"\nBaseline Return: {baseline_return:+.2f}%")
    logger.info("-" * 70)
    logger.info(f"{'Improvement':<25} {'Return':>12} {'Delta':>12} {'Win Rate':>12}")
    logger.info("-" * 70)

    for name, params in improvements:
        engine = EnhancedBacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            max_positions=5,
            enable_trailing_stop=False,
            enable_volatility_sizing=False,
            enable_market_regime=False,
            enable_multi_timeframe=False,
            enable_volume_breakout=False,
            enable_swing_low_stop=False,
            enable_additional_filters=False,
            enable_compound=False,
            min_score=55,
            min_confidence=0.60,
            **params,
        )
        results = engine.run_backtest(stock_data, "2022-01-01", "2024-12-31", sector_map=sector_map)
        delta = results.total_return_pct - baseline_return
        logger.info(f"{name:<25} {results.total_return_pct:+11.2f}% {delta:+11.2f}% {results.win_rate:11.1%}")

    # All combined
    logger.info("-" * 70)
    all_engine = EnhancedBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        max_positions=5,
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
    all_results = all_engine.run_backtest(stock_data, "2022-01-01", "2024-12-31", sector_map=sector_map)
    total_delta = all_results.total_return_pct - baseline_return
    logger.info(f"{'ALL COMBINED':<25} {all_results.total_return_pct:+11.2f}% {total_delta:+11.2f}% {all_results.win_rate:11.1%}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_comparison()
    run_ablation_study()
