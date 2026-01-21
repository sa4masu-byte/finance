#!/usr/bin/env python3
"""
Daily Stock Recommender Script

This script provides daily stock recommendations based on:
- Technical indicators and scoring
- Candlestick pattern recognition
- Sector rotation analysis
- Fundamental filters
- ML-enhanced predictions (if model is trained)

Usage:
    python scripts/daily_recommender.py [--top N] [--output FORMAT]

Options:
    --top N        Number of top recommendations (default: 5)
    --output       Output format: cli, json, csv (default: cli)
    --use-ml       Enable ML scoring if model exists
    --verbose      Show detailed analysis
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    SCREENING_CRITERIA,
    SCORING_WEIGHTS,
    RISK_PARAMS,
    MARKET_REGIME_PARAMS,
    ML_SCORING_PARAMS,
)
from src.data.fetcher import StockDataFetcher
from src.analysis.indicators import TechnicalIndicators
from src.analysis.sector_analysis import SectorAnalyzer, get_sector_for_stock
from src.analysis.fundamental_filter import FundamentalFilter
from backtesting.scoring_engine import ScoringEngine


@dataclass
class Recommendation:
    """Stock recommendation with analysis"""
    symbol: str
    name: str
    sector: str
    score: float
    confidence: float
    signal: str  # "Strong Buy", "Buy", "Hold"
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward: float
    pattern_signals: List[str]
    key_indicators: Dict[str, float]
    fundamental_score: float
    sector_score: float
    ml_score: Optional[float] = None


# Stock name mapping
STOCK_NAMES = {
    "7203.T": "Toyota Motor",
    "7267.T": "Honda Motor",
    "6758.T": "Sony Group",
    "6861.T": "Keyence",
    "6981.T": "Murata Manufacturing",
    "8306.T": "MUFG",
    "8316.T": "SMFG",
    "9432.T": "NTT",
    "9984.T": "SoftBank Group",
    "4502.T": "Takeda Pharmaceutical",
    "9983.T": "Fast Retailing",
    "7974.T": "Nintendo",
    "4063.T": "Shin-Etsu Chemical",
    "6098.T": "Recruit Holdings",
}


def fetch_stock_data(symbols: List[str], days: int = 120) -> Dict:
    """Fetch stock data for all symbols"""
    fetcher = StockDataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    stock_data = {}
    for symbol in symbols:
        try:
            df = fetcher.fetch_stock_data(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            if df is not None and len(df) > 20:
                stock_data[symbol] = df
        except Exception as e:
            print(f"Warning: Failed to fetch {symbol}: {e}")

    return stock_data


def analyze_stock(
    symbol: str,
    df,
    indicator_engine: TechnicalIndicators,
    scoring_engine: ScoringEngine,
    sector_analyzer: SectorAnalyzer,
    fundamental_filter: FundamentalFilter,
    stock_data: Dict,
) -> Optional[Recommendation]:
    """Analyze a single stock and generate recommendation"""

    # Calculate all indicators
    df_with_indicators = indicator_engine.calculate_all(df)

    if len(df_with_indicators) < 20:
        return None

    # Get latest indicators
    indicators = indicator_engine.get_latest_indicators(df_with_indicators)

    # Calculate score
    score, confidence = scoring_engine.calculate_score(indicators)

    # Check minimum criteria
    if score < SCREENING_CRITERIA["min_score"]:
        return None
    if confidence < SCREENING_CRITERIA["min_confidence"]:
        return None

    # Get current price and calculate levels
    current_price = indicators["Close"]
    atr = indicators.get("ATR", current_price * 0.02)

    # Stop loss: ATR-based
    stop_loss = current_price - (atr * RISK_PARAMS["stop_loss_atr_multiplier"])

    # Target: Based on risk/reward
    target_price = current_price + (atr * 3)  # 3:1 risk/reward

    risk_reward = (target_price - current_price) / (current_price - stop_loss)

    # Determine signal strength
    if score >= 80 and confidence >= 0.85:
        signal = "Strong Buy"
    elif score >= 70:
        signal = "Buy"
    else:
        signal = "Hold"

    # Get pattern signals
    pattern_signals = []
    if indicators.get("Pattern_Engulfing") == 1:
        pattern_signals.append("Bullish Engulfing")
    if indicators.get("Pattern_Hammer") == 1:
        pattern_signals.append("Hammer")
    if indicators.get("Pattern_ThreeSoldiers") == 1:
        pattern_signals.append("Three White Soldiers")
    if indicators.get("Pattern_MorningStar") == 1:
        pattern_signals.append("Morning Star")
    if indicators.get("Pattern_Score", 0) > 30:
        pattern_signals.append("Strong Bullish Pattern")

    # Key indicators
    key_indicators = {
        "RSI": indicators.get("RSI_14", 50),
        "MACD": indicators.get("MACD_Histogram", 0),
        "Volume_Ratio": indicators.get("Volume_Ratio", 1),
        "BB_Position": indicators.get("BB_Percent", 0.5),
    }

    # Sector score
    sector_metrics = sector_analyzer.calculate_sector_score(
        get_sector_for_stock(symbol)
    )
    sector_score = sector_metrics.sector_score

    # Fundamental score
    fundamental_score = fundamental_filter.get_composite_score(symbol)

    return Recommendation(
        symbol=symbol,
        name=STOCK_NAMES.get(symbol, symbol),
        sector=get_sector_for_stock(symbol),
        score=round(score, 1),
        confidence=round(confidence, 2),
        signal=signal,
        entry_price=round(current_price, 0),
        stop_loss=round(stop_loss, 0),
        target_price=round(target_price, 0),
        risk_reward=round(risk_reward, 2),
        pattern_signals=pattern_signals,
        key_indicators=key_indicators,
        fundamental_score=round(fundamental_score, 1),
        sector_score=round(sector_score, 1),
    )


def get_recommendations(
    top_n: int = 5,
    use_ml: bool = False,
    verbose: bool = False,
) -> List[Recommendation]:
    """Get top stock recommendations"""

    # Initialize components
    indicator_engine = TechnicalIndicators()
    scoring_engine = ScoringEngine()
    sector_analyzer = SectorAnalyzer()
    fundamental_filter = FundamentalFilter()

    # Get list of symbols to analyze
    symbols = list(STOCK_NAMES.keys())

    if verbose:
        print(f"Fetching data for {len(symbols)} stocks...")

    # Fetch data
    stock_data = fetch_stock_data(symbols)

    if verbose:
        print(f"Successfully fetched {len(stock_data)} stocks")

    # Load sector data
    sector_analyzer.load_sector_data(stock_data)

    # Analyze each stock
    recommendations = []
    for symbol, df in stock_data.items():
        rec = analyze_stock(
            symbol,
            df,
            indicator_engine,
            scoring_engine,
            sector_analyzer,
            fundamental_filter,
            stock_data,
        )
        if rec:
            recommendations.append(rec)

    # Sort by score
    recommendations.sort(key=lambda x: x.score, reverse=True)

    # Apply ML scoring if enabled
    if use_ml and ML_SCORING_PARAMS.get("enabled"):
        try:
            from backtesting.ml_scoring import EnsembleScoringEngine
            ensemble = EnsembleScoringEngine()
            if ensemble.load():
                for rec in recommendations:
                    indicators = indicator_engine.get_latest_indicators(
                        indicator_engine.calculate_all(stock_data[rec.symbol])
                    )
                    ml_score, ml_conf = ensemble.predict_score(indicators)
                    rec.ml_score = round(ml_score, 1)

                    # Re-calculate combined score
                    rec.score = round(
                        rec.score * 0.7 + ml_score * 0.3,
                        1
                    )

                # Re-sort after ML adjustment
                recommendations.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            if verbose:
                print(f"ML scoring unavailable: {e}")

    return recommendations[:top_n]


def output_cli(recommendations: List[Recommendation], verbose: bool = False) -> None:
    """Output recommendations to CLI"""
    print("\n" + "=" * 70)
    print(f"  DAILY STOCK RECOMMENDATIONS - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 70)

    if not recommendations:
        print("\nNo strong recommendations today. Market conditions may be unfavorable.")
        return

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{'─' * 70}")
        print(f"  #{i} {rec.name} ({rec.symbol})")
        print(f"{'─' * 70}")

        # Signal and score
        signal_color = {
            "Strong Buy": "\033[92m",  # Green
            "Buy": "\033[93m",          # Yellow
            "Hold": "\033[94m",         # Blue
        }
        reset = "\033[0m"
        print(f"  Signal: {signal_color.get(rec.signal, '')}{rec.signal}{reset}")
        print(f"  Score: {rec.score}/100 (Confidence: {rec.confidence*100:.0f}%)")

        # Price levels
        print(f"\n  Entry Price:  JPY {rec.entry_price:,.0f}")
        print(f"  Stop Loss:    JPY {rec.stop_loss:,.0f} ({(rec.stop_loss/rec.entry_price-1)*100:.1f}%)")
        print(f"  Target:       JPY {rec.target_price:,.0f} ({(rec.target_price/rec.entry_price-1)*100:.1f}%)")
        print(f"  Risk/Reward:  1:{rec.risk_reward:.1f}")

        # Sector and fundamentals
        print(f"\n  Sector: {rec.sector} (Score: {rec.sector_score})")
        print(f"  Fundamental Score: {rec.fundamental_score}")

        if rec.ml_score:
            print(f"  ML Score: {rec.ml_score}")

        # Pattern signals
        if rec.pattern_signals:
            print(f"\n  Pattern Signals:")
            for pattern in rec.pattern_signals:
                print(f"    - {pattern}")

        if verbose:
            print(f"\n  Key Indicators:")
            for key, value in rec.key_indicators.items():
                print(f"    {key}: {value:.2f}")

    print(f"\n{'=' * 70}")
    print("  DISCLAIMER: This is not financial advice. Trade at your own risk.")
    print("=" * 70 + "\n")


def output_json(recommendations: List[Recommendation]) -> None:
    """Output recommendations as JSON"""
    data = {
        "date": datetime.now().isoformat(),
        "recommendations": [asdict(rec) for rec in recommendations],
    }
    print(json.dumps(data, indent=2, default=str))


def output_csv(recommendations: List[Recommendation]) -> None:
    """Output recommendations as CSV"""
    import csv
    import io

    output = io.StringIO()
    if recommendations:
        writer = csv.DictWriter(output, fieldnames=asdict(recommendations[0]).keys())
        writer.writeheader()
        for rec in recommendations:
            writer.writerow(asdict(rec))

    print(output.getvalue())


def main():
    parser = argparse.ArgumentParser(
        description="Daily Stock Recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=5,
        help="Number of top recommendations (default: 5)",
    )
    parser.add_argument(
        "--output", "-o",
        choices=["cli", "json", "csv"],
        default="cli",
        help="Output format (default: cli)",
    )
    parser.add_argument(
        "--use-ml",
        action="store_true",
        help="Enable ML scoring if model exists",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed analysis",
    )

    args = parser.parse_args()

    # Get recommendations
    recommendations = get_recommendations(
        top_n=args.top,
        use_ml=args.use_ml,
        verbose=args.verbose,
    )

    # Output based on format
    if args.output == "json":
        output_json(recommendations)
    elif args.output == "csv":
        output_csv(recommendations)
    else:
        output_cli(recommendations, verbose=args.verbose)


if __name__ == "__main__":
    main()
