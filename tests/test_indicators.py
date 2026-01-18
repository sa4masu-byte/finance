"""
Tests for technical indicator calculations
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """Test suite for TechnicalIndicators class"""

    @pytest.fixture
    def sample_ohlcv_data(self) -> pd.DataFrame:
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Generate realistic price data
        base_price = 1000
        returns = np.random.randn(100) * 0.02
        close = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'Open': close * (1 + np.random.randn(100) * 0.005),
            'High': close * (1 + np.abs(np.random.randn(100) * 0.01)),
            'Low': close * (1 - np.abs(np.random.randn(100) * 0.01)),
            'Close': close,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)

        return df

    @pytest.fixture
    def indicators(self) -> TechnicalIndicators:
        """Create TechnicalIndicators instance"""
        return TechnicalIndicators()

    def test_calculate_all_returns_dataframe(self, indicators, sample_ohlcv_data):
        """Test that calculate_all returns a DataFrame with all indicators"""
        result = indicators.calculate_all(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)

        # Check that key indicators are present
        expected_columns = [
            'SMA_5', 'SMA_25', 'SMA_75',
            'MACD', 'MACD_Signal', 'MACD_Histogram',
            'RSI_14',
            'Stoch_K', 'Stoch_D',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Percent',
            'ATR', 'ATR_Percent',
            'Volume_MA', 'Volume_Ratio',
            'OBV', 'OBV_MA', 'OBV_Trend'
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_calculate_all_with_empty_dataframe(self, indicators):
        """Test calculate_all handles empty DataFrame"""
        empty_df = pd.DataFrame()
        result = indicators.calculate_all(empty_df)
        assert result is not None

    def test_calculate_all_with_none(self, indicators):
        """Test calculate_all handles None input"""
        result = indicators.calculate_all(None)
        assert result is None

    def test_sma_calculation(self, indicators, sample_ohlcv_data):
        """Test SMA calculation correctness"""
        result = indicators.add_moving_averages(sample_ohlcv_data)

        # Manually calculate SMA_5 for comparison
        expected_sma_5 = sample_ohlcv_data['Close'].rolling(window=5).mean()

        pd.testing.assert_series_equal(
            result['SMA_5'],
            expected_sma_5,
            check_names=False
        )

    def test_rsi_bounds(self, indicators, sample_ohlcv_data):
        """Test RSI values are within 0-100 range"""
        result = indicators.add_rsi(sample_ohlcv_data)

        # Skip NaN values
        rsi_values = result['RSI_14'].dropna()

        assert (rsi_values >= 0).all(), "RSI should be >= 0"
        assert (rsi_values <= 100).all(), "RSI should be <= 100"

    def test_rsi_zero_loss_handling(self, indicators):
        """Test RSI calculation when there are no losses (all gains)"""
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        # Create consistently rising prices
        prices = list(range(100, 120))

        df = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': prices,
            'Volume': [100000] * 20
        }, index=dates)

        result = indicators.add_rsi(df)
        rsi_values = result['RSI_14'].dropna()

        # When all prices go up, RSI should be 100 (or very close)
        assert (rsi_values >= 99).all(), "RSI should be ~100 when all gains"

    def test_stochastic_bounds(self, indicators, sample_ohlcv_data):
        """Test Stochastic values are within 0-100 range"""
        result = indicators.add_stochastic(sample_ohlcv_data)

        stoch_k = result['Stoch_K'].dropna()
        stoch_d = result['Stoch_D'].dropna()

        assert (stoch_k >= 0).all() and (stoch_k <= 100).all()
        assert (stoch_d >= 0).all() and (stoch_d <= 100).all()

    def test_stochastic_zero_range_handling(self, indicators):
        """Test Stochastic calculation when high == low (zero range)"""
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        # Create flat prices
        flat_price = 1000

        df = pd.DataFrame({
            'Open': [flat_price] * 20,
            'High': [flat_price] * 20,
            'Low': [flat_price] * 20,
            'Close': [flat_price] * 20,
            'Volume': [100000] * 20
        }, index=dates)

        result = indicators.add_stochastic(df)
        stoch_k = result['Stoch_K'].dropna()

        # When range is zero, Stoch_K should be neutral (50)
        assert (stoch_k == 50.0).all(), "Stoch_K should be 50 when high == low"

    def test_bollinger_bands_order(self, indicators, sample_ohlcv_data):
        """Test that BB_Upper > BB_Middle > BB_Lower"""
        result = indicators.add_bollinger_bands(sample_ohlcv_data)

        # Skip NaN values
        valid_rows = result.dropna(subset=['BB_Upper', 'BB_Middle', 'BB_Lower'])

        assert (valid_rows['BB_Upper'] >= valid_rows['BB_Middle']).all()
        assert (valid_rows['BB_Middle'] >= valid_rows['BB_Lower']).all()

    def test_bollinger_percent_zero_width_handling(self, indicators):
        """Test BB_Percent calculation when bands have zero width"""
        dates = pd.date_range(start='2024-01-01', periods=25, freq='D')
        # Create constant prices (std = 0)
        flat_price = 1000

        df = pd.DataFrame({
            'Open': [flat_price] * 25,
            'High': [flat_price] * 25,
            'Low': [flat_price] * 25,
            'Close': [flat_price] * 25,
            'Volume': [100000] * 25
        }, index=dates)

        result = indicators.add_bollinger_bands(df)
        bb_percent = result['BB_Percent'].dropna()

        # When bands have zero width, BB_Percent should be neutral (0.5)
        assert (bb_percent == 0.5).all(), "BB_Percent should be 0.5 when width is zero"

    def test_atr_positive(self, indicators, sample_ohlcv_data):
        """Test ATR is always positive"""
        result = indicators.add_atr(sample_ohlcv_data)

        atr_values = result['ATR'].dropna()
        assert (atr_values >= 0).all(), "ATR should be >= 0"

    def test_volume_ratio_calculation(self, indicators, sample_ohlcv_data):
        """Test volume ratio calculation"""
        result = indicators.add_volume_indicators(sample_ohlcv_data)

        # Volume ratio should be Volume / Volume_MA
        valid_rows = result.dropna(subset=['Volume_MA', 'Volume_Ratio'])
        expected_ratio = valid_rows['Volume'] / valid_rows['Volume_MA']

        pd.testing.assert_series_equal(
            valid_rows['Volume_Ratio'],
            expected_ratio,
            check_names=False
        )

    def test_obv_trend_values(self, indicators, sample_ohlcv_data):
        """Test OBV_Trend contains only valid values (-1, 0, 1)"""
        result = indicators.add_obv(sample_ohlcv_data)

        obv_trend = result['OBV_Trend'].dropna()
        valid_values = {-1, 0, 1}

        assert set(obv_trend.unique()).issubset(valid_values)

    def test_get_latest_indicators(self, indicators, sample_ohlcv_data):
        """Test get_latest_indicators returns correct dictionary"""
        df_with_indicators = indicators.calculate_all(sample_ohlcv_data)
        result = indicators.get_latest_indicators(df_with_indicators)

        assert isinstance(result, dict)
        assert 'Close' in result
        assert 'RSI_14' in result
        assert result['Close'] is not None

    def test_get_latest_indicators_empty_df(self, indicators):
        """Test get_latest_indicators with empty DataFrame"""
        result = indicators.get_latest_indicators(pd.DataFrame())
        assert result == {}


class TestIndicatorEdgeCases:
    """Test edge cases and error handling"""

    def test_missing_columns(self):
        """Test handling of missing required columns"""
        indicators = TechnicalIndicators()

        # DataFrame missing 'Close' column
        df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Volume': [1000, 1100, 1200]
        })

        with pytest.raises(KeyError):
            indicators.add_moving_averages(df)

    def test_custom_parameters(self):
        """Test initialization with custom parameters"""
        custom_params = {
            "sma_short": 10,
            "sma_medium": 50,
            "sma_long": 200,
            "macd_fast": 8,
            "macd_slow": 17,
            "macd_signal": 9,
            "rsi_period": 10,
            "stoch_k": 10,
            "stoch_d": 3,
            "bb_period": 15,
            "bb_std": 2.5,
            "atr_period": 10,
            "volume_ma_period": 15,
            "obv_ma_period": 5,
        }

        indicators = TechnicalIndicators(params=custom_params)
        assert indicators.params["sma_short"] == 10
        assert indicators.params["bb_std"] == 2.5
