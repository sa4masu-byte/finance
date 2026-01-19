"""
Generate daily report for the web dashboard
Run this script daily (e.g., via cron) to update recommendations

Usage:
    python scripts/generate_daily_report.py
"""
import sys
from pathlib import Path
import json
import logging
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import REPORTS_DIR
from src.analysis.indicators import TechnicalIndicators
from backtesting.scoring_engine import ScoringEngine
from backtesting.enhanced_backtest_engine import MarketRegime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Stock list with names
STOCK_LIST = {
    "7203.T": "Toyota",
    "7267.T": "Honda",
    "6758.T": "Sony",
    "6861.T": "Keyence",
    "6981.T": "Murata",
    "8306.T": "MUFG",
    "8316.T": "SMFG",
    "9984.T": "SoftBank G",
    "9432.T": "NTT",
    "4063.T": "Shin-Etsu",
    "9983.T": "Fast Retail",
    "7974.T": "Nintendo",
    "4502.T": "Takeda",
    "6098.T": "Recruit",
    "6367.T": "Daikin",
}


def fetch_stock_data(symbols: list, period: str = "6mo"):
    """Fetch stock data using yfinance"""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return {}

    stock_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if not df.empty:
                stock_data[symbol] = df
                logger.info(f"Fetched {symbol}: {len(df)} days")
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")

    return stock_data


def detect_market_regime(stock_data: dict) -> str:
    """Detect current market regime from stock data"""
    if not stock_data:
        return "sideways"

    # Calculate average market movement
    recent_returns = []
    for symbol, df in stock_data.items():
        if len(df) >= 20:
            ret_20d = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1)
            recent_returns.append(ret_20d)

    if not recent_returns:
        return "sideways"

    import numpy as np
    avg_return = np.mean(recent_returns)
    volatility = np.std(recent_returns)

    if volatility > 0.15:
        return "high_volatility"
    elif avg_return > 0.03:
        return "bull"
    elif avg_return < -0.03:
        return "bear"
    else:
        return "sideways"


def generate_recommendations(stock_data: dict) -> list:
    """Generate stock recommendations"""
    indicator_engine = TechnicalIndicators()
    scoring_engine = ScoringEngine()

    recommendations = []

    for symbol, df in stock_data.items():
        try:
            # Calculate indicators
            df_with_ind = indicator_engine.calculate_all(df)

            if df_with_ind.empty:
                continue

            # Get latest indicators
            latest = df_with_ind.iloc[-1].to_dict()

            # Calculate score
            result = scoring_engine.calculate_score(latest)
            score = result['total_score']
            confidence = result['confidence']

            # Only recommend if score >= 55 and confidence >= 0.60
            if score >= 55 and confidence >= 0.60:
                recommendations.append({
                    "symbol": symbol,
                    "name": STOCK_LIST.get(symbol, symbol),
                    "score": round(score, 1),
                    "confidence": round(confidence, 2),
                    "price": round(latest.get('Close', 0), 0),
                    "rsi": round(latest.get('RSI_14', 50), 1),
                    "trend": "bullish" if latest.get('SMA_5', 0) > latest.get('SMA_25', 0) else "bearish",
                })

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    # Sort by score descending
    recommendations.sort(key=lambda x: x['score'], reverse=True)

    return recommendations[:5]  # Top 5


def load_positions() -> list:
    """Load current positions from file"""
    positions_file = REPORTS_DIR / "positions.json"

    if positions_file.exists():
        with open(positions_file, 'r') as f:
            return json.load(f)

    return []


def load_performance() -> dict:
    """Load performance metrics from file"""
    performance_file = REPORTS_DIR / "performance.json"

    if performance_file.exists():
        with open(performance_file, 'r') as f:
            return json.load(f)

    return {
        "total_return": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
    }


def generate_report():
    """Generate the daily report"""
    logger.info("=" * 50)
    logger.info("Generating Daily Report")
    logger.info("=" * 50)

    # Fetch latest data
    symbols = list(STOCK_LIST.keys())
    stock_data = fetch_stock_data(symbols)

    if not stock_data:
        logger.error("No stock data available")
        return None

    # Detect market regime
    market_regime = detect_market_regime(stock_data)
    logger.info(f"Market Regime: {market_regime}")

    # Generate recommendations
    recommendations = generate_recommendations(stock_data)
    logger.info(f"Found {len(recommendations)} recommendations")

    # Load positions and performance
    positions = load_positions()
    performance = load_performance()

    # Create report
    report = {
        "generated_at": datetime.now().isoformat(),
        "market_regime": market_regime,
        "recommendations": recommendations,
        "positions": positions,
        "performance": performance,
    }

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "latest_report.json"

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Report saved to {report_file}")

    # Print summary
    print("\n" + "=" * 50)
    print("TODAY'S RECOMMENDATIONS")
    print("=" * 50)

    if recommendations:
        for rec in recommendations:
            print(f"  {rec['symbol']:10s} {rec['name']:15s} Score: {rec['score']:5.1f} ({rec['confidence']:.0%})")
    else:
        print("  No recommendations today")

    print("=" * 50)

    return report


if __name__ == "__main__":
    report = generate_report()
