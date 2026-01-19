"""
Tests for scoring engine
"""
import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.scoring_engine import ScoringEngine, score_stock


class TestScoringEngine:
    """Test suite for ScoringEngine class"""

    @pytest.fixture
    def engine(self) -> ScoringEngine:
        """Create ScoringEngine instance with default weights"""
        return ScoringEngine()

    @pytest.fixture
    def bullish_indicators(self) -> dict:
        """Create bullish indicator values"""
        return {
            # Trend - bullish
            "Close": 1050,
            "SMA_5": 1040,
            "SMA_25": 1020,
            "SMA_75": 1000,
            "MACD": 5.0,
            "MACD_Signal": 3.0,
            "MACD_Histogram": 2.0,

            # Momentum - bullish
            "RSI_14": 55,  # In optimal zone
            "Stoch_K": 65,
            "Stoch_D": 55,

            # Volume - strong
            "Volume": 1500000,
            "Volume_MA": 1000000,
            "Volume_Ratio": 1.5,
            "OBV": 10000000,
            "OBV_Trend": 1,

            # Volatility
            "BB_Upper": 1100,
            "BB_Middle": 1020,
            "BB_Lower": 940,
            "BB_Width": 160,
            "BB_Percent": 0.4,
            "ATR": 20,
            "ATR_Percent": 1.9,
        }

    @pytest.fixture
    def bearish_indicators(self) -> dict:
        """Create bearish indicator values"""
        return {
            # Trend - bearish
            "Close": 950,
            "SMA_5": 960,
            "SMA_25": 1020,
            "SMA_75": 1050,
            "MACD": -5.0,
            "MACD_Signal": -3.0,
            "MACD_Histogram": -2.0,

            # Momentum - overbought
            "RSI_14": 85,
            "Stoch_K": 90,
            "Stoch_D": 85,

            # Volume - weak
            "Volume": 500000,
            "Volume_MA": 1000000,
            "Volume_Ratio": 0.5,
            "OBV": 8000000,
            "OBV_Trend": -1,

            # Volatility
            "BB_Upper": 1100,
            "BB_Middle": 1000,
            "BB_Lower": 900,
            "BB_Width": 200,
            "BB_Percent": 0.75,
            "ATR": 40,
            "ATR_Percent": 4.0,
        }

    def test_engine_initialization(self):
        """Test engine initializes with default weights"""
        engine = ScoringEngine()
        assert engine.weights is not None
        assert abs(sum(engine.weights.values()) - 1.0) < 0.01

    def test_engine_custom_weights(self):
        """Test engine accepts custom weights"""
        custom_weights = {
            "trend": 0.4,
            "momentum": 0.2,
            "volume": 0.2,
            "volatility": 0.1,
            "pattern": 0.1,
        }
        engine = ScoringEngine(weights=custom_weights)
        assert engine.weights["trend"] == 0.4

    def test_engine_invalid_weights_sum(self):
        """Test engine raises error for invalid weight sum"""
        invalid_weights = {
            "trend": 0.5,
            "momentum": 0.5,
            "volume": 0.5,
            "volatility": 0.1,
            "pattern": 0.1,
        }
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            ScoringEngine(weights=invalid_weights)

    def test_calculate_score_returns_dict(self, engine, bullish_indicators):
        """Test calculate_score returns correct dictionary structure"""
        result = engine.calculate_score(bullish_indicators)

        assert isinstance(result, dict)
        required_keys = [
            'total_score', 'trend_score', 'momentum_score',
            'volume_score', 'volatility_score', 'pattern_score', 'confidence'
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_score_range(self, engine, bullish_indicators):
        """Test all scores are within 0-100 range"""
        result = engine.calculate_score(bullish_indicators)

        for key, value in result.items():
            if key == 'confidence':
                assert 0 <= value <= 1, f"{key} should be 0-1"
            else:
                assert 0 <= value <= 100, f"{key} should be 0-100"

    def test_bullish_higher_than_bearish(self, engine, bullish_indicators, bearish_indicators):
        """Test bullish indicators score higher than bearish"""
        bullish_score = engine.calculate_score(bullish_indicators)
        bearish_score = engine.calculate_score(bearish_indicators)

        assert bullish_score['total_score'] > bearish_score['total_score']

    def test_trend_score_calculation(self, engine, bullish_indicators):
        """Test trend score increases with bullish conditions"""
        # Full bullish trend
        result = engine.calculate_score(bullish_indicators)
        assert result['trend_score'] > 50

        # Reduce trend strength
        weak_trend = bullish_indicators.copy()
        weak_trend['SMA_5'] = 1010  # Below SMA_25
        weak_trend['MACD'] = -1  # Below signal
        weak_result = engine.calculate_score(weak_trend)

        assert result['trend_score'] > weak_result['trend_score']

    def test_momentum_score_rsi_zones(self, engine):
        """Test momentum score for different RSI zones"""
        base = {
            "Close": 1000, "SMA_5": 990, "SMA_25": 980,
            "Stoch_K": 50, "Stoch_D": 45
        }

        # Optimal RSI zone (40-65)
        optimal = {**base, "RSI_14": 55}
        optimal_score = engine.calculate_score(optimal)

        # Oversold recovery zone (30-40)
        oversold = {**base, "RSI_14": 35}
        oversold_score = engine.calculate_score(oversold)

        # Overbought zone (>70)
        overbought = {**base, "RSI_14": 80}
        overbought_score = engine.calculate_score(overbought)

        # Optimal should have highest momentum score
        assert optimal_score['momentum_score'] >= oversold_score['momentum_score']
        assert optimal_score['momentum_score'] > overbought_score['momentum_score']

    def test_volume_score_thresholds(self, engine):
        """Test volume score at different thresholds"""
        base = {"Close": 1000, "OBV_Trend": 0}

        # Exceptional volume (>=2.0)
        exceptional = {**base, "Volume_Ratio": 2.5}
        exceptional_score = engine.calculate_score(exceptional)

        # High volume (1.5-2.0)
        high = {**base, "Volume_Ratio": 1.7}
        high_score = engine.calculate_score(high)

        # Low volume
        low = {**base, "Volume_Ratio": 0.8}
        low_score = engine.calculate_score(low)

        assert exceptional_score['volume_score'] >= high_score['volume_score']
        assert high_score['volume_score'] > low_score['volume_score']

    def test_volatility_score_bb_position(self, engine):
        """Test volatility score based on Bollinger Band position"""
        base = {"BB_Middle": 1000, "BB_Width": 50}

        # Near lower band
        near_lower = {**base, "BB_Percent": 0.2}
        near_lower_score = engine.calculate_score(near_lower)

        # Near upper band
        near_upper = {**base, "BB_Percent": 0.8}
        near_upper_score = engine.calculate_score(near_upper)

        # Near lower band should score higher (potential bounce)
        assert near_lower_score['volatility_score'] > near_upper_score['volatility_score']

    def test_confidence_calculation(self, engine):
        """Test confidence calculation based on available indicators"""
        # All indicators present
        full = {
            "Close": 1000, "SMA_5": 990, "SMA_25": 980, "RSI_14": 50,
            "MACD": 1, "MACD_Signal": 0.5, "Volume_Ratio": 1.2,
            "BB_Upper": 1050, "BB_Lower": 950, "ATR": 20
        }
        full_result = engine.calculate_score(full)

        # Only some indicators
        partial = {"Close": 1000, "SMA_5": 990}
        partial_result = engine.calculate_score(partial)

        assert full_result['confidence'] > partial_result['confidence']
        assert full_result['confidence'] == 1.0

    def test_should_recommend_criteria(self, engine, bullish_indicators):
        """Test should_recommend based on criteria"""
        result = engine.calculate_score(bullish_indicators)

        # High score should be recommended
        if result['total_score'] >= 65 and result['confidence'] >= 0.7:
            assert engine.should_recommend(result)

    def test_empty_indicators(self, engine):
        """Test handling of empty indicators"""
        result = engine.calculate_score({})

        assert result['total_score'] == 0
        assert result['confidence'] == 0

    def test_none_indicator_values(self, engine):
        """Test handling of None values in indicators"""
        indicators = {
            "Close": 1000,
            "SMA_5": None,
            "SMA_25": None,
            "RSI_14": None,
        }
        result = engine.calculate_score(indicators)

        assert result is not None
        assert 0 <= result['total_score'] <= 100


class TestScoreStockFunction:
    """Test the convenience function score_stock"""

    def test_score_stock_basic(self):
        """Test score_stock function"""
        indicators = {
            "Close": 1000,
            "SMA_5": 990,
            "SMA_25": 980,
            "RSI_14": 55,
        }
        result = score_stock(indicators)

        assert isinstance(result, dict)
        assert 'total_score' in result

    def test_score_stock_custom_weights(self):
        """Test score_stock with custom weights"""
        indicators = {"Close": 1000, "RSI_14": 55}
        custom_weights = {
            "trend": 0.5,
            "momentum": 0.2,
            "volume": 0.1,
            "volatility": 0.1,
            "pattern": 0.1,
        }
        result = score_stock(indicators, weights=custom_weights)

        assert isinstance(result, dict)
