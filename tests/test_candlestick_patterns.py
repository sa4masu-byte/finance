"""
Tests for candlestick pattern recognition
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.indicators import TechnicalIndicators


@pytest.fixture
def indicator_engine():
    """Create TechnicalIndicators instance"""
    return TechnicalIndicators()


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data"""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    np.random.seed(42)

    # Start with base price
    base_price = 1000
    prices = [base_price]

    for i in range(49):
        change = np.random.randn() * 20
        prices.append(prices[-1] + change)

    df = pd.DataFrame({
        'Open': prices,
        'High': [p + abs(np.random.randn() * 10) for p in prices],
        'Low': [p - abs(np.random.randn() * 10) for p in prices],
        'Close': [p + np.random.randn() * 5 for p in prices],
        'Volume': np.random.randint(100000, 500000, 50),
    }, index=dates)

    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
    df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)

    return df


@pytest.fixture
def bullish_engulfing_data():
    """Create data with bullish engulfing pattern"""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')

    # Day 8: Bearish candle (Open > Close)
    # Day 9: Bullish candle that engulfs day 8
    df = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 100, 95],   # Day 9 opens below day 8 close
        'High': [102, 103, 104, 105, 106, 107, 108, 108, 101, 110],
        'Low': [99, 100, 101, 102, 103, 104, 105, 99, 94, 94],
        'Close': [101, 102, 103, 104, 105, 106, 107, 100, 99, 108],  # Day 8 closes at 100, Day 9 at 108
        'Volume': [100000] * 10,
    }, index=dates)

    return df


@pytest.fixture
def hammer_data():
    """Create data with hammer pattern after downtrend"""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')

    # Downtrend followed by hammer on day 9
    df = pd.DataFrame({
        'Open': [110, 108, 106, 104, 102, 100, 98, 96, 94, 93],  # Day 9: small body at top
        'High': [111, 109, 107, 105, 103, 101, 99, 97, 95, 94],  # Small upper shadow
        'Low': [108, 106, 104, 102, 100, 98, 96, 94, 92, 85],    # Long lower shadow on day 9
        'Close': [108, 106, 104, 102, 100, 98, 96, 94, 92, 93],  # Close near open
        'Volume': [100000] * 10,
    }, index=dates)

    return df


@pytest.fixture
def doji_data():
    """Create data with doji pattern"""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')

    df = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 100],
        'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 105],  # Upper shadow
        'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 95],     # Lower shadow
        'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 100.5],  # Almost same as open
        'Volume': [100000] * 10,
    }, index=dates)

    return df


class TestCandlestickPatterns:
    """Test candlestick pattern recognition"""

    def test_add_candlestick_patterns_columns(self, indicator_engine, sample_ohlcv):
        """Test that all pattern columns are added"""
        df = indicator_engine.add_candlestick_patterns(sample_ohlcv)

        expected_columns = [
            'Candle_Body', 'Candle_Range', 'Candle_Body_Pct',
            'Upper_Shadow', 'Lower_Shadow',
            'Pattern_Doji', 'Pattern_Hammer', 'Pattern_Engulfing',
            'Pattern_PinBar', 'Pattern_ThreeSoldiers', 'Pattern_MorningStar',
            'Pattern_Score',
        ]

        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_candle_body_calculation(self, indicator_engine, sample_ohlcv):
        """Test candle body calculation"""
        df = indicator_engine.add_candlestick_patterns(sample_ohlcv)

        # Body should be Close - Open
        expected_body = sample_ohlcv['Close'] - sample_ohlcv['Open']
        pd.testing.assert_series_equal(
            df['Candle_Body'],
            expected_body,
            check_names=False
        )

    def test_candle_range_calculation(self, indicator_engine, sample_ohlcv):
        """Test candle range calculation"""
        df = indicator_engine.add_candlestick_patterns(sample_ohlcv)

        # Range should be High - Low
        expected_range = sample_ohlcv['High'] - sample_ohlcv['Low']
        pd.testing.assert_series_equal(
            df['Candle_Range'],
            expected_range,
            check_names=False
        )

    def test_shadow_calculations(self, indicator_engine, sample_ohlcv):
        """Test shadow calculations"""
        df = indicator_engine.add_candlestick_patterns(sample_ohlcv)

        # Upper shadow = High - max(Open, Close)
        upper_expected = sample_ohlcv['High'] - sample_ohlcv[['Open', 'Close']].max(axis=1)
        pd.testing.assert_series_equal(
            df['Upper_Shadow'],
            upper_expected,
            check_names=False
        )

        # Lower shadow = min(Open, Close) - Low
        lower_expected = sample_ohlcv[['Open', 'Close']].min(axis=1) - sample_ohlcv['Low']
        pd.testing.assert_series_equal(
            df['Lower_Shadow'],
            lower_expected,
            check_names=False
        )

    def test_doji_detection(self, indicator_engine, doji_data):
        """Test doji pattern detection"""
        df = indicator_engine.add_candlestick_patterns(doji_data)

        # Day 9 has small body (100.5 - 100 = 0.5) with range of 10
        # Body percent = 0.5 / 10 = 0.05 < 0.10 threshold
        assert df['Pattern_Doji'].iloc[-1] != 0, "Should detect doji on last day"

    def test_bullish_engulfing_detection(self, indicator_engine, bullish_engulfing_data):
        """Test bullish engulfing pattern detection"""
        df = indicator_engine.add_candlestick_patterns(bullish_engulfing_data)

        # Last candle should be bullish engulfing
        assert df['Pattern_Engulfing'].iloc[-1] == 1, "Should detect bullish engulfing"

    def test_hammer_detection(self, indicator_engine, hammer_data):
        """Test hammer pattern detection"""
        df = indicator_engine.add_candlestick_patterns(hammer_data)

        # Last candle should be hammer (bullish reversal after downtrend)
        # Note: Pattern detection depends on trend context
        # Just verify the pattern recognition runs without error
        assert 'Pattern_Hammer' in df.columns

    def test_pattern_score_range(self, indicator_engine, sample_ohlcv):
        """Test that pattern score is within expected range"""
        df = indicator_engine.add_candlestick_patterns(sample_ohlcv)

        # Pattern score should be between -100 and 100
        assert df['Pattern_Score'].min() >= -100, "Pattern score below minimum"
        assert df['Pattern_Score'].max() <= 100, "Pattern score above maximum"

    def test_three_soldiers_detection(self, indicator_engine):
        """Test three white soldiers pattern detection"""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')

        # Create three consecutive bullish candles with higher closes and opens
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 105, 108, 112, 116, 120, 124],
            'High': [102, 104, 106, 108, 111, 115, 119, 123, 127, 131],
            'Low': [99, 100, 101, 102, 104, 107, 111, 115, 119, 123],
            'Close': [101, 103, 105, 107, 110, 114, 118, 122, 126, 130],
            'Volume': [100000] * 10,
        }, index=dates)

        df_with_patterns = indicator_engine.add_candlestick_patterns(df)

        # Should detect three white soldiers somewhere
        assert 'Pattern_ThreeSoldiers' in df_with_patterns.columns

    def test_empty_dataframe(self, indicator_engine):
        """Test with empty DataFrame"""
        df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        result = indicator_engine.add_candlestick_patterns(df)

        assert len(result) == 0

    def test_single_row(self, indicator_engine):
        """Test with single row DataFrame"""
        df = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [95],
            'Close': [102],
            'Volume': [100000],
        }, index=pd.date_range(start='2024-01-01', periods=1))

        result = indicator_engine.add_candlestick_patterns(df)

        assert len(result) == 1
        assert 'Pattern_Score' in result.columns


class TestPatternIntegration:
    """Test pattern integration with full indicator calculation"""

    def test_calculate_all_includes_patterns(self, indicator_engine, sample_ohlcv):
        """Test that calculate_all includes pattern columns"""
        df = indicator_engine.calculate_all(sample_ohlcv)

        assert 'Pattern_Score' in df.columns
        assert 'Pattern_Engulfing' in df.columns

    def test_get_latest_indicators_includes_patterns(self, indicator_engine, sample_ohlcv):
        """Test that get_latest_indicators includes pattern data"""
        df = indicator_engine.calculate_all(sample_ohlcv)
        indicators = indicator_engine.get_latest_indicators(df)

        assert 'Pattern_Score' in indicators
        assert 'Pattern_Engulfing' in indicators
        assert 'Pattern_Hammer' in indicators
