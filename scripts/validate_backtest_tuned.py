"""
Backtest with tuned parameters for better performance
Adjusts scoring thresholds and entry criteria
"""
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backtesting.backtest_engine import BacktestEngine
from backtesting.scoring_engine import ScoringEngine
from src.analysis.indicators import TechnicalIndicators

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_trending_market_data(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_price: float,
    trend_bias: float = 0.0005,
    seed: int = None,
) -> pd.DataFrame:
    """Generate data with strong trending characteristics"""
    if seed is None:
        seed = hash(symbol) % 10000
    np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)

    # Generate trending returns with momentum
    returns = []
    trend_state = 0  # -1: downtrend, 0: neutral, 1: uptrend
    trend_duration = 0

    for i in range(n_days):
        # Trend regime changes
        if trend_duration <= 0:
            # New trend
            trend_state = np.random.choice([-1, 0, 1], p=[0.25, 0.30, 0.45])  # Slight bullish bias
            trend_duration = np.random.randint(10, 40)  # 10-40 days trend

        trend_duration -= 1

        # Base return based on trend
        if trend_state == 1:
            base_return = trend_bias + np.random.uniform(0.001, 0.003)
        elif trend_state == -1:
            base_return = -trend_bias + np.random.uniform(-0.003, -0.001)
        else:
            base_return = np.random.uniform(-0.001, 0.001)

        # Add volatility
        volatility = 0.015
        random_component = np.random.normal(0, volatility)

        # Momentum: continuation bias
        if i > 0:
            momentum = returns[-1] * 0.15  # 15% momentum
        else:
            momentum = 0

        daily_return = base_return + random_component + momentum

        # Occasional large moves (earnings, news)
        if np.random.random() < 0.03:
            daily_return += np.random.choice([-1, 1]) * np.random.uniform(0.03, 0.06)

        returns.append(daily_return)

    returns = np.array(returns)
    close = initial_price * np.cumprod(1 + returns)

    # Generate OHLC
    high = close * (1 + np.abs(np.random.normal(0, 0.008, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.008, n_days)))
    open_price = np.roll(close, 1) * (1 + np.random.normal(0, 0.003, n_days))
    open_price[0] = initial_price

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    # Volume correlated with price moves
    base_volume = 1000000
    volume = base_volume * (1 + np.abs(returns) * 5) * (1 + np.random.normal(0, 0.2, n_days))
    volume = np.maximum(volume, 200000).astype(int)

    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=dates)


def create_stock_universe() -> dict:
    """Create diverse stock universe with different characteristics"""
    stocks = {
        # Strong performers (uptrend bias)
        "7203.T": {"name": "Toyota", "price": 2200, "bias": 0.0006},
        "6758.T": {"name": "Sony", "price": 12000, "bias": 0.0005},
        "6861.T": {"name": "Keyence", "price": 55000, "bias": 0.0007},
        "8306.T": {"name": "MUFG", "price": 800, "bias": 0.0008},
        "6098.T": {"name": "Recruit", "price": 5000, "bias": 0.0006},

        # Moderate performers
        "7267.T": {"name": "Honda", "price": 3500, "bias": 0.0003},
        "6981.T": {"name": "Murata", "price": 8000, "bias": 0.0004},
        "8316.T": {"name": "SMFG", "price": 5000, "bias": 0.0005},
        "4063.T": {"name": "Shin-Etsu", "price": 18000, "bias": 0.0004},
        "9432.T": {"name": "NTT", "price": 3000, "bias": 0.0002},

        # Mixed performers
        "9984.T": {"name": "SoftBank", "price": 6000, "bias": 0.0002},
        "9983.T": {"name": "Fast Retailing", "price": 70000, "bias": 0.0003},
        "7974.T": {"name": "Nintendo", "price": 55000, "bias": 0.0003},
        "4502.T": {"name": "Takeda", "price": 3500, "bias": 0.0001},
        "4188.T": {"name": "Mitsubishi Chem", "price": 900, "bias": 0.0002},
    }

    stock_data = {}
    for symbol, info in stocks.items():
        df = generate_trending_market_data(
            symbol=symbol,
            start_date="2022-01-01",
            end_date="2024-12-31",
            initial_price=info["price"],
            trend_bias=info["bias"],
        )
        stock_data[symbol] = df

        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        logger.info(f"{symbol} ({info['name']:15s}): {total_return:+6.1f}%")

    return stock_data


class TunedBacktestEngine(BacktestEngine):
    """Backtest engine with tuned scoring thresholds"""

    def __init__(self, *args, min_score=55, min_confidence=0.60, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_score = min_score
        self.min_confidence = min_confidence

    def _check_entries(self, date, stocks):
        """Look for new entry opportunities with tuned thresholds"""
        candidates = []

        for symbol, df in stocks.items():
            if any(pos.symbol == symbol for pos in self.open_positions):
                continue

            indicators = self._get_indicators_at_date(df, date)
            if not indicators:
                continue

            score_result = self.scoring_engine.calculate_score(indicators)

            # Use tuned thresholds
            if (score_result["total_score"] >= self.min_score and
                score_result["confidence"] >= self.min_confidence):

                current_price = indicators.get("Close")
                atr = indicators.get("ATR")

                if current_price and atr:
                    candidates.append({
                        "symbol": symbol,
                        "score": score_result["total_score"],
                        "price": current_price,
                        "atr": atr,
                    })

        candidates.sort(key=lambda x: x["score"], reverse=True)

        available_slots = self.max_positions - len(self.open_positions)
        for candidate in candidates[:available_slots]:
            self._enter_position(
                symbol=candidate["symbol"],
                date=date,
                price=candidate["price"],
                atr=candidate["atr"],
                score=candidate["score"],
            )


def run_tuned_backtest():
    """Run backtest with various parameter settings"""

    INITIAL_CAPITAL = 3_000_000

    logger.info("=" * 70)
    logger.info("Swing Trading System - Parameter Tuning Backtest")
    logger.info("=" * 70)
    logger.info(f"Initial Capital: {INITIAL_CAPITAL:,} JPY")

    # Generate data
    logger.info("\n[1/3] Generating market data with trending characteristics...")
    stock_data = create_stock_universe()

    # Test different parameter combinations
    parameter_sets = [
        {"name": "Conservative", "min_score": 65, "min_confidence": 0.70},
        {"name": "Moderate", "min_score": 55, "min_confidence": 0.60},
        {"name": "Aggressive", "min_score": 45, "min_confidence": 0.50},
        {"name": "Very Aggressive", "min_score": 40, "min_confidence": 0.40},
    ]

    logger.info("\n[2/3] Testing different parameter settings...")

    best_result = None
    best_params = None
    best_return = -float('inf')

    for params in parameter_sets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {params['name']} (min_score={params['min_score']}, min_confidence={params['min_confidence']})")
        logger.info(f"{'='*60}")

        engine = TunedBacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            max_positions=5,
            commission_rate=0.001,
            slippage_rate=0.0005,
            min_score=params["min_score"],
            min_confidence=params["min_confidence"],
        )

        results = engine.run_backtest(stock_data, "2022-01-01", "2024-12-31")

        profit_loss = results.final_capital - INITIAL_CAPITAL

        logger.info(f"  Trades:        {results.num_trades}")
        logger.info(f"  Win Rate:      {results.win_rate:.1%}")
        logger.info(f"  Return:        {results.total_return_pct:+.2f}%")
        logger.info(f"  P&L:           {profit_loss:+,.0f} JPY")
        logger.info(f"  Profit Factor: {results.profit_factor:.2f}")
        logger.info(f"  Sharpe Ratio:  {results.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown:  {results.max_drawdown:.2f}%")

        if results.total_return_pct > best_return:
            best_return = results.total_return_pct
            best_result = results
            best_params = params

    # Final summary with best parameters
    logger.info("\n" + "=" * 70)
    logger.info("BEST RESULT SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\nBest Parameters: {best_params['name']}")
    logger.info(f"  min_score: {best_params['min_score']}")
    logger.info(f"  min_confidence: {best_params['min_confidence']}")

    profit_loss = best_result.final_capital - INITIAL_CAPITAL
    cagr = ((best_result.final_capital / INITIAL_CAPITAL) ** (1/3) - 1) * 100

    logger.info(f"""
{'='*50}
PORTFOLIO PERFORMANCE (Best Settings)
{'='*50}
Initial Capital:     {INITIAL_CAPITAL:>15,} JPY
Final Capital:       {best_result.final_capital:>15,.0f} JPY
────────────────────────────────────────────────────
Total P&L:           {profit_loss:>+15,.0f} JPY
Total Return:        {best_result.total_return_pct:>+14.2f}%
CAGR (3 years):      {cagr:>+14.2f}%

METRICS
────────────────────────────────────────────────────
Win Rate:            {best_result.win_rate:>14.1%}
Profit Factor:       {best_result.profit_factor:>14.2f}
Sharpe Ratio:        {best_result.sharpe_ratio:>14.2f}
Max Drawdown:        {best_result.max_drawdown:>14.2f}%
Total Trades:        {best_result.num_trades:>14}
Avg Holding Days:    {best_result.avg_holding_days:>14.1f}
{'='*50}
    """)

    # Year by year with best settings
    logger.info("\n[3/3] Year-by-year breakdown with best settings...")

    for year in [2022, 2023, 2024]:
        engine = TunedBacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            max_positions=5,
            commission_rate=0.001,
            slippage_rate=0.0005,
            min_score=best_params["min_score"],
            min_confidence=best_params["min_confidence"],
        )

        results = engine.run_backtest(
            stock_data,
            f"{year}-01-01",
            f"{year}-12-31"
        )

        logger.info(f"  {year}: {results.total_return_pct:+7.2f}% | Win: {results.win_rate:5.1%} | Trades: {results.num_trades:3d} | PF: {results.profit_factor:.2f}")

    logger.info("\n" + "=" * 70)

    return best_result, best_params


if __name__ == "__main__":
    run_tuned_backtest()
