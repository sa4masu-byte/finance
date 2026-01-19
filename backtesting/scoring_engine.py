"""
Scoring engine for swing trading recommendations
Calculates composite score from technical indicators
"""
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import SCORING_WEIGHTS, SCORING_PARAMS, SCREENING_CRITERIA

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Calculate composite scores from technical indicators
    Score range: 0-100
    """

    def __init__(self, weights: Optional[Dict] = None, params: Optional[Dict] = None):
        """
        Initialize scoring engine with weights

        Args:
            weights: Dictionary of category weights (trend, momentum, volume, volatility, pattern)
                    If None, uses default weights from settings
            params: Dictionary of scoring parameters. If None, uses default from settings
        """
        self.weights = weights if weights is not None else SCORING_WEIGHTS
        self.params = params if params is not None else SCORING_PARAMS

        # Validate weights
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def calculate_score(self, indicators: Dict) -> Dict:
        """
        Calculate composite score from indicators

        Args:
            indicators: Dictionary of indicator values

        Returns:
            Dictionary with:
            - total_score: Composite score (0-100)
            - trend_score: Trend component score
            - momentum_score: Momentum component score
            - volume_score: Volume component score
            - volatility_score: Volatility component score
            - pattern_score: Pattern component score
            - confidence: Confidence level (0-1)
        """
        # Calculate component scores
        trend_score = self._calculate_trend_score(indicators)
        momentum_score = self._calculate_momentum_score(indicators)
        volume_score = self._calculate_volume_score(indicators)
        volatility_score = self._calculate_volatility_score(indicators)
        pattern_score = self._calculate_pattern_score(indicators)

        # Weighted composite score
        total_score = (
            trend_score * self.weights["trend"]
            + momentum_score * self.weights["momentum"]
            + volume_score * self.weights["volume"]
            + volatility_score * self.weights["volatility"]
            + pattern_score * self.weights["pattern"]
        )

        # Calculate confidence (how many indicators are available)
        confidence = self._calculate_confidence(indicators)

        return {
            "total_score": round(total_score, 2),
            "trend_score": round(trend_score, 2),
            "momentum_score": round(momentum_score, 2),
            "volume_score": round(volume_score, 2),
            "volatility_score": round(volatility_score, 2),
            "pattern_score": round(pattern_score, 2),
            "confidence": round(confidence, 3),
        }

    def _calculate_trend_score(self, ind: Dict) -> float:
        """
        Calculate trend score (0-100)

        Components:
        - Price above SMA_25
        - SMA_5 > SMA_25 (Golden Cross)
        - MACD > Signal
        - MACD Histogram increasing
        - All SMAs aligned bonus
        """
        score = 0
        p = self.params["trend"]
        max_score = p["max_score"]

        try:
            # Price above medium-term SMA
            if ind.get("Close") and ind.get("SMA_25"):
                if ind["Close"] > ind["SMA_25"]:
                    score += p["price_above_sma"]

            # Golden Cross (short SMA above medium SMA)
            if ind.get("SMA_5") and ind.get("SMA_25"):
                if ind["SMA_5"] > ind["SMA_25"]:
                    score += p["golden_cross"]

            # MACD above signal line
            if ind.get("MACD") and ind.get("MACD_Signal"):
                if ind["MACD"] > ind["MACD_Signal"]:
                    score += p["macd_above_signal"]

            # MACD Histogram positive (momentum increasing)
            if ind.get("MACD_Histogram"):
                if ind["MACD_Histogram"] > 0:
                    score += p["macd_histogram_positive"]

            # Bonus for strong trend confirmation
            if ind.get("SMA_5") and ind.get("SMA_25") and ind.get("SMA_75"):
                if ind["SMA_5"] > ind["SMA_25"] > ind["SMA_75"]:
                    score += p["sma_aligned_bonus"]  # All SMAs aligned

        except (KeyError, TypeError) as e:
            logger.warning(f"Error calculating trend score: {e}")

        # Normalize to 0-100
        return (score / max_score) * 100

    def _calculate_momentum_score(self, ind: Dict) -> float:
        """
        Calculate momentum score (0-100)

        Components:
        - RSI in optimal range
        - RSI in oversold recovery zone
        - RSI still has room
        - Stochastic %K > %D
        - Stochastic %K < 80 (not overbought)
        """
        score = 0
        p = self.params["momentum"]
        max_score = p["max_score"]

        try:
            rsi = ind.get("RSI_14")
            if rsi is not None:
                if p["rsi_optimal_min"] <= rsi <= p["rsi_optimal_max"]:
                    # Optimal swing trading zone
                    score += p["rsi_optimal_score"]
                elif p["rsi_oversold_min"] <= rsi < p["rsi_oversold_max"]:
                    # Oversold recovery potential
                    score += p["rsi_oversold_score"]
                elif p["rsi_room_min"] < rsi <= p["rsi_room_max"]:
                    # Still has some room
                    score += p["rsi_room_score"]
                # RSI > 70 or < 30: No points (too extreme)

            # Stochastic
            stoch_k = ind.get("Stoch_K")
            stoch_d = ind.get("Stoch_D")

            if stoch_k is not None and stoch_d is not None:
                # %K above %D (bullish crossover)
                if stoch_k > stoch_d:
                    score += p["stoch_bullish_crossover"]

                # Not overbought
                if stoch_k < p["stoch_overbought_threshold"]:
                    score += p["stoch_not_overbought_score"]

        except (KeyError, TypeError) as e:
            logger.warning(f"Error calculating momentum score: {e}")

        # Normalize to 0-100
        return (score / max_score) * 100

    def _calculate_volume_score(self, ind: Dict) -> float:
        """
        Calculate volume score (0-100)

        Components:
        - Volume > exceptional threshold
        - Volume > high threshold
        - Volume > above average threshold
        - OBV trending up
        """
        score = 0
        p = self.params["volume"]
        max_score = p["max_score"]

        try:
            volume_ratio = ind.get("Volume_Ratio")

            if volume_ratio is not None:
                if volume_ratio >= p["exceptional_threshold"]:
                    # Exceptional volume (likely institutional)
                    score += p["exceptional_score"]
                elif volume_ratio >= p["high_threshold"]:
                    # High volume
                    score += p["high_score"]
                elif volume_ratio >= p["above_avg_threshold"]:
                    # Above average
                    score += p["above_avg_score"]

            # OBV trend
            obv_trend = ind.get("OBV_Trend")
            if obv_trend == 1:  # Uptrend
                score += p["obv_uptrend_score"]

        except (KeyError, TypeError) as e:
            logger.warning(f"Error calculating volume score: {e}")

        # Normalize to 0-100
        return (score / max_score) * 100

    def _calculate_volatility_score(self, ind: Dict) -> float:
        """
        Calculate volatility score (0-100)

        Components:
        - Price near lower Bollinger Band
        - Bollinger Band squeeze (narrow width)
        - Low ATR bonus
        """
        score = 0
        p = self.params["volatility"]
        max_score = p["max_score"]

        try:
            # Bollinger Band position
            bb_percent = ind.get("BB_Percent")
            if bb_percent is not None:
                if bb_percent < p["near_lower_band_threshold"]:
                    # Near lower band (potential bounce)
                    score += p["near_lower_band_score"]
                elif bb_percent < p["below_middle_threshold"]:
                    # Below middle
                    score += p["below_middle_score"]

            # Bollinger Band width (squeeze detection)
            bb_width = ind.get("BB_Width")
            bb_middle = ind.get("BB_Middle")

            if bb_width is not None and bb_middle is not None and bb_middle > 0:
                width_percent = (bb_width / bb_middle) * 100
                if width_percent < p["tight_squeeze_threshold"]:  # Tight squeeze
                    score += p["tight_squeeze_score"]
                elif width_percent < p["moderate_squeeze_threshold"]:
                    score += p["moderate_squeeze_score"]

            # ATR check (bonus for low volatility)
            atr_percent = ind.get("ATR_Percent")
            if atr_percent is not None and atr_percent < p["low_atr_threshold"]:
                score += p["low_atr_score"]

        except (KeyError, TypeError) as e:
            logger.warning(f"Error calculating volatility score: {e}")

        # Normalize to 0-100
        return (score / max_score) * 100

    def _calculate_pattern_score(self, ind: Dict) -> float:
        """
        Calculate pattern score (0-100)

        Components:
        - Price near support (detected from SMAs)
        - Bullish candlestick pattern (placeholder)
        - Volume confirmation (placeholder)

        Note: This is a simplified version. Full implementation would include
        more sophisticated pattern recognition.
        """
        score = 0
        p = self.params["pattern"]
        max_score = p["max_score"]

        try:
            # Simple support detection: price near SMA_25
            close = ind.get("Close")
            sma_25 = ind.get("SMA_25")

            if close and sma_25:
                distance = abs(close - sma_25) / sma_25
                if distance < p["near_support_threshold"]:  # Within threshold of SMA_25
                    score += p["near_support_score"]

            # Volume confirmation (already scored in volume section)
            # Pattern recognition placeholder
            # TODO: Implement candlestick pattern recognition

        except (KeyError, TypeError) as e:
            logger.warning(f"Error calculating pattern score: {e}")

        # Normalize to 0-100
        return (score / max_score) * 100

    def _calculate_confidence(self, indicators: Dict) -> float:
        """
        Calculate confidence level based on availability of indicators

        Args:
            indicators: Dictionary of indicator values

        Returns:
            Confidence level (0-1)
        """
        required_indicators = [
            "Close", "SMA_5", "SMA_25", "RSI_14",
            "MACD", "MACD_Signal", "Volume_Ratio",
            "BB_Upper", "BB_Lower", "ATR"
        ]

        available = sum(1 for ind in required_indicators if indicators.get(ind) is not None)
        confidence = available / len(required_indicators)

        return confidence

    def should_recommend(self, score_result: Dict) -> bool:
        """
        Determine if stock should be recommended based on score

        Args:
            score_result: Result from calculate_score()

        Returns:
            True if stock meets recommendation criteria
        """
        return (
            score_result["total_score"] >= SCREENING_CRITERIA["min_score"]
            and score_result["confidence"] >= SCREENING_CRITERIA["min_confidence"]
        )


# Convenience function
def score_stock(indicators: Dict, weights: Optional[Dict] = None) -> Dict:
    """
    Calculate score for a stock's indicators

    Args:
        indicators: Dictionary of indicator values
        weights: Optional custom weights

    Returns:
        Score result dictionary
    """
    engine = ScoringEngine(weights=weights)
    return engine.calculate_score(indicators)
