"""
Backtest validation with realistic market simulation
Includes: Trends, Momentum, Mean Reversion, Sector Correlation
Based on actual Japanese market movements 2022-2024
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


def generate_market_regime_data(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_price: float,
    sector: str,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate stock data with realistic market characteristics:
    - Trend persistence (momentum)
    - Mean reversion at extremes
    - Sector correlation
    - Regime changes (bull/bear/sideways)

    Based on actual Japanese market 2022-2024:
    - 2022 Q1-Q2: Volatile (Ukraine war, Fed hikes)
    - 2022 Q3-Q4: Recovery with weak yen
    - 2023: Strong bull market (Buffett effect, corporate governance reform)
    - 2024: Historic highs, Nikkei 40000+
    """
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
                # Volatile bear (Ukraine, Fed)
                regime_params.append({"drift": -0.0003, "vol": 0.025, "trend_strength": 0.3})
            else:
                # Recovery
                regime_params.append({"drift": 0.0004, "vol": 0.020, "trend_strength": 0.5})
        elif year == 2023:
            if month <= 3:
                # Early bull
                regime_params.append({"drift": 0.0006, "vol": 0.018, "trend_strength": 0.6})
            elif month <= 9:
                # Strong bull (Buffett effect)
                regime_params.append({"drift": 0.0008, "vol": 0.016, "trend_strength": 0.7})
            else:
                # Consolidation
                regime_params.append({"drift": 0.0003, "vol": 0.017, "trend_strength": 0.5})
        else:  # 2024
            if month <= 3:
                # Historic rally (Nikkei 40000)
                regime_params.append({"drift": 0.0010, "vol": 0.015, "trend_strength": 0.75})
            elif month <= 8:
                # Consolidation after highs
                regime_params.append({"drift": 0.0002, "vol": 0.020, "trend_strength": 0.4})
            else:
                # Year-end rally
                regime_params.append({"drift": 0.0005, "vol": 0.018, "trend_strength": 0.55})

    # Sector adjustments (some sectors outperformed)
    sector_multipliers = {
        "auto": 1.3,        # Export beneficiaries (weak yen)
        "tech": 1.2,        # AI boom
        "finance": 1.4,     # Rising rates
        "pharma": 0.9,      # Defensive
        "retail": 1.1,      # Consumption recovery
        "telecom": 0.85,    # Stable but low growth
        "materials": 1.15,  # Commodity prices
    }
    sector_mult = sector_multipliers.get(sector, 1.0)

    # Generate returns with trend persistence and mean reversion
    returns = []
    prev_return = 0
    cumulative_return = 0

    for i, params in enumerate(regime_params):
        drift = params["drift"] * sector_mult
        vol = params["vol"]
        trend_strength = params["trend_strength"]

        # Trend persistence (momentum): previous return influences current
        momentum_component = prev_return * trend_strength * 0.3

        # Mean reversion: extreme moves tend to reverse
        mean_reversion = 0
        if cumulative_return > 0.15:  # Up more than 15%
            mean_reversion = -0.001
        elif cumulative_return < -0.10:  # Down more than 10%
            mean_reversion = 0.001

        # Random component
        random_component = np.random.normal(0, vol)

        # Combined return
        daily_return = drift + momentum_component + mean_reversion + random_component

        # Add occasional gaps (earnings, news)
        if np.random.random() < 0.02:  # 2% chance of gap
            gap = np.random.choice([-1, 1]) * np.random.uniform(0.02, 0.05)
            daily_return += gap

        returns.append(daily_return)
        prev_return = daily_return
        cumulative_return += daily_return

    returns = np.array(returns)

    # Calculate prices
    close = initial_price * np.cumprod(1 + returns)

    # Generate OHLC with realistic intraday patterns
    intraday_vol = 0.008

    # Intraday patterns: gaps, intraday trends
    open_price = np.zeros(n_days)
    high = np.zeros(n_days)
    low = np.zeros(n_days)

    for i in range(n_days):
        if i == 0:
            open_price[i] = initial_price
        else:
            # Gap from previous close
            gap = np.random.normal(0, 0.003)
            open_price[i] = close[i-1] * (1 + gap)

        # Intraday range
        intraday_range = close[i] * np.random.uniform(0.01, 0.025)

        if close[i] > open_price[i]:
            # Bullish day
            low[i] = min(open_price[i], close[i]) - intraday_range * np.random.uniform(0.2, 0.5)
            high[i] = max(open_price[i], close[i]) + intraday_range * np.random.uniform(0.1, 0.3)
        else:
            # Bearish day
            high[i] = max(open_price[i], close[i]) + intraday_range * np.random.uniform(0.2, 0.5)
            low[i] = min(open_price[i], close[i]) - intraday_range * np.random.uniform(0.1, 0.3)

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    low = np.maximum(low, close * 0.9)  # Limit extreme lows

    # Volume with trend correlation (high volume on big moves)
    base_volume = np.random.randint(800000, 2000000)
    price_change = np.abs(returns)
    volume_factor = 1 + price_change * 10  # Higher volume on big moves
    volume = base_volume * volume_factor * (1 + np.random.normal(0, 0.2, n_days))
    volume = np.maximum(volume, 200000).astype(int)

    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=dates)

    return df


def create_realistic_stock_universe() -> dict:
    """Create stock data based on actual Japanese market performance 2022-2024"""

    # Stocks with sector classification and realistic initial prices (Jan 2022)
    stocks = {
        # Auto (benefited from weak yen)
        "7203.T": {"name": "Toyota", "price": 2200, "sector": "auto"},
        "7267.T": {"name": "Honda", "price": 3400, "sector": "auto"},

        # Tech (AI boom beneficiaries)
        "6758.T": {"name": "Sony", "price": 14000, "sector": "tech"},
        "6861.T": {"name": "Keyence", "price": 62000, "sector": "tech"},
        "6981.T": {"name": "Murata", "price": 9500, "sector": "tech"},

        # Finance (rising rates beneficiaries)
        "8306.T": {"name": "MUFG", "price": 650, "sector": "finance"},
        "8316.T": {"name": "SMFG", "price": 4000, "sector": "finance"},

        # Telecom/IT
        "9984.T": {"name": "SoftBank Group", "price": 5500, "sector": "tech"},
        "9432.T": {"name": "NTT", "price": 3200, "sector": "telecom"},

        # Materials
        "4063.T": {"name": "Shin-Etsu Chemical", "price": 19000, "sector": "materials"},

        # Retail
        "9983.T": {"name": "Fast Retailing", "price": 68000, "sector": "retail"},
        "7974.T": {"name": "Nintendo", "price": 57000, "sector": "retail"},

        # Pharma
        "4502.T": {"name": "Takeda", "price": 3200, "sector": "pharma"},

        # Other
        "4188.T": {"name": "Mitsubishi Chemical", "price": 850, "sector": "materials"},
        "6098.T": {"name": "Recruit", "price": 5800, "sector": "tech"},
    }

    stock_data = {}
    for symbol, info in stocks.items():
        df = generate_market_regime_data(
            symbol=symbol,
            start_date="2022-01-01",
            end_date="2024-12-31",
            initial_price=info["price"],
            sector=info["sector"],
        )
        stock_data[symbol] = df

        # Calculate performance
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        total_return = (end_price / start_price - 1) * 100

        logger.info(f"{symbol} ({info['name']:15s}): {start_price:>8,.0f} -> {end_price:>8,.0f} ({total_return:+6.1f}%)")

    return stock_data


def run_realistic_backtest():
    """Run backtest with realistic market data"""

    INITIAL_CAPITAL = 3_000_000  # 300万円（信用取引）

    logger.info("=" * 70)
    logger.info("Swing Trading System - Realistic Market Backtest")
    logger.info("=" * 70)
    logger.info(f"Initial Capital: {INITIAL_CAPITAL:,} JPY (Margin Trading)")
    logger.info("")
    logger.info("Market Assumptions (based on actual 2022-2024):")
    logger.info("  - 2022 H1: Volatile/Bear (Ukraine, Fed rate hikes)")
    logger.info("  - 2022 H2: Recovery with weak yen")
    logger.info("  - 2023: Strong bull market (Buffett effect)")
    logger.info("  - 2024: Historic highs (Nikkei 40000+)")

    # Generate data
    logger.info("\n[1/2] Generating realistic market data...")
    stock_data = create_realistic_stock_universe()
    logger.info(f"\nGenerated data for {len(stock_data)} stocks")

    # Run backtest
    logger.info("\n[2/2] Running backtest...")

    test_periods = [
        ("2022-01-01", "2022-12-31", "2022 (Volatile)"),
        ("2023-01-01", "2023-12-31", "2023 (Bull Market)"),
        ("2024-01-01", "2024-12-31", "2024 (Historic Highs)"),
        ("2022-01-01", "2024-12-31", "Full Period (3 Years)"),
    ]

    all_results = []

    for start_date, end_date, period_name in test_periods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Period: {period_name}")
        logger.info(f"{'='*60}")

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

        logger.info(f"\n  Trades: {results.num_trades} (Win: {len(results.winning_trades)}, Lose: {len(results.losing_trades)})")
        logger.info(f"  Win Rate:        {results.win_rate:.1%}")
        logger.info(f"  Initial:         {INITIAL_CAPITAL:>12,} JPY")
        logger.info(f"  Final:           {results.final_capital:>12,.0f} JPY")
        logger.info(f"  P&L:             {profit_loss:>+12,.0f} JPY")
        logger.info(f"  Return:          {results.total_return_pct:>+11.2f}%")
        logger.info(f"  Profit Factor:   {results.profit_factor:>11.2f}")
        logger.info(f"  Sharpe Ratio:    {results.sharpe_ratio:>11.2f}")
        logger.info(f"  Max Drawdown:    {results.max_drawdown:>11.2f}%")

        if results.num_trades > 0:
            logger.info(f"  Avg Win:         {results.avg_win:>+10.2f}%")
            logger.info(f"  Avg Loss:        {results.avg_loss:>10.2f}%")
            logger.info(f"  Avg Holding:     {results.avg_holding_days:>10.1f} days")

    # Final Summary
    full_period = all_results[-1]["results"]
    profit_loss = full_period.final_capital - INITIAL_CAPITAL
    cagr = ((full_period.final_capital / INITIAL_CAPITAL) ** (1/3) - 1) * 100

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)

    # Year by year comparison
    logger.info("\nYear-by-Year Performance:")
    logger.info("-" * 50)
    for r in all_results[:-1]:
        res = r["results"]
        pnl = res.final_capital - INITIAL_CAPITAL
        logger.info(f"  {r['period']:20s}: {res.total_return_pct:+7.2f}% | Win Rate: {res.win_rate:5.1%} | Trades: {res.num_trades:3d}")

    logger.info("\n" + "-" * 50)
    logger.info(f"  {'TOTAL (3 Years)':20s}: {full_period.total_return_pct:+7.2f}% | Win Rate: {full_period.win_rate:5.1%} | Trades: {full_period.num_trades:3d}")

    logger.info(f"""

{'='*50}
PORTFOLIO SUMMARY
{'='*50}
Initial Capital:     {INITIAL_CAPITAL:>15,} JPY
Final Capital:       {full_period.final_capital:>15,.0f} JPY
────────────────────────────────────────────────────
Total P&L:           {profit_loss:>+15,.0f} JPY
Total Return:        {full_period.total_return_pct:>+14.2f}%
CAGR (3 years):      {cagr:>+14.2f}%

RISK METRICS
────────────────────────────────────────────────────
Win Rate:            {full_period.win_rate:>14.1%}
Profit Factor:       {full_period.profit_factor:>14.2f}
Sharpe Ratio:        {full_period.sharpe_ratio:>14.2f}
Max Drawdown:        {full_period.max_drawdown:>14.2f}%
Total Trades:        {full_period.num_trades:>14}

TRADE STATISTICS
────────────────────────────────────────────────────
Winning Trades:      {len(full_period.winning_trades):>14}
Losing Trades:       {len(full_period.losing_trades):>14}
Avg Win:             {full_period.avg_win:>+13.2f}%
Avg Loss:            {full_period.avg_loss:>13.2f}%
Avg Holding Days:    {full_period.avg_holding_days:>14.1f}
{'='*50}
    """)

    # Success criteria check
    logger.info("SUCCESS CRITERIA CHECK:")
    logger.info("-" * 50)

    criteria = [
        ("Win Rate > 50%", full_period.win_rate > 0.50, f"{full_period.win_rate:.1%}"),
        ("Profit Factor > 1.5", full_period.profit_factor > 1.5, f"{full_period.profit_factor:.2f}"),
        ("Sharpe Ratio > 0.5", full_period.sharpe_ratio > 0.5, f"{full_period.sharpe_ratio:.2f}"),
        ("Max Drawdown < 20%", full_period.max_drawdown < 20, f"{full_period.max_drawdown:.2f}%"),
        ("Positive CAGR", cagr > 0, f"{cagr:+.2f}%"),
    ]

    passed = 0
    for name, met, value in criteria:
        status = "[PASS]" if met else "[FAIL]"
        if met:
            passed += 1
        logger.info(f"  {status} {name}: {value}")

    logger.info(f"\nResult: {passed}/{len(criteria)} criteria passed")
    logger.info("=" * 70)

    return all_results


if __name__ == "__main__":
    run_realistic_backtest()
