"""
Tests for sector analysis module
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.sector_analysis import (
    SectorAnalyzer,
    SectorMetrics,
    SectorPhase,
    SECTOR_MAPPING,
    get_sector_for_stock,
    is_defensive_sector,
    is_cyclical_sector,
)


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for multiple symbols"""
    dates = pd.date_range(start='2024-01-01', periods=120, freq='D')
    np.random.seed(42)

    stock_data = {}

    # Technology stocks
    for symbol in ['6758.T', '6861.T']:  # Sony, Keyence
        base_price = 5000 + np.random.randint(0, 2000)
        returns = np.random.randn(120) * 0.02
        prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(120) * 0.005),
            'High': prices * (1 + abs(np.random.randn(120)) * 0.01),
            'Low': prices * (1 - abs(np.random.randn(120)) * 0.01),
            'Close': prices,
            'Volume': np.random.randint(100000, 500000, 120),
        }, index=dates)

        stock_data[symbol] = df

    # Financial stocks
    for symbol in ['8306.T', '8316.T']:  # MUFG, SMFG
        base_price = 1000 + np.random.randint(0, 500)
        returns = np.random.randn(120) * 0.015
        prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(120) * 0.005),
            'High': prices * (1 + abs(np.random.randn(120)) * 0.01),
            'Low': prices * (1 - abs(np.random.randn(120)) * 0.01),
            'Close': prices,
            'Volume': np.random.randint(500000, 2000000, 120),
        }, index=dates)

        stock_data[symbol] = df

    return stock_data


@pytest.fixture
def market_data():
    """Create sample market index data"""
    dates = pd.date_range(start='2024-01-01', periods=120, freq='D')
    np.random.seed(42)

    base_price = 30000
    returns = np.random.randn(120) * 0.01
    prices = base_price * np.cumprod(1 + returns)

    return pd.DataFrame({
        'Open': prices * (1 + np.random.randn(120) * 0.003),
        'High': prices * (1 + abs(np.random.randn(120)) * 0.005),
        'Low': prices * (1 - abs(np.random.randn(120)) * 0.005),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 120),
    }, index=dates)


@pytest.fixture
def sector_analyzer():
    """Create SectorAnalyzer instance"""
    return SectorAnalyzer()


class TestSectorMapping:
    """Test sector mapping functionality"""

    def test_get_sector_for_known_stock(self):
        """Test getting sector for known stocks"""
        assert get_sector_for_stock('7203.T') == 'Automotive'
        assert get_sector_for_stock('6758.T') == 'Technology'
        assert get_sector_for_stock('8306.T') == 'Financial'

    def test_get_sector_for_unknown_stock(self):
        """Test getting sector for unknown stock"""
        assert get_sector_for_stock('9999.T') == 'Unknown'

    def test_is_defensive_sector(self):
        """Test defensive sector identification"""
        assert is_defensive_sector('Healthcare')
        assert is_defensive_sector('Telecom')
        assert not is_defensive_sector('Technology')

    def test_is_cyclical_sector(self):
        """Test cyclical sector identification"""
        assert is_cyclical_sector('Technology')
        assert is_cyclical_sector('Automotive')
        assert not is_cyclical_sector('Healthcare')


class TestSectorAnalyzer:
    """Test SectorAnalyzer class"""

    def test_load_sector_data(self, sector_analyzer, sample_stock_data):
        """Test loading sector data"""
        sector_analyzer.load_sector_data(sample_stock_data)

        # Should have aggregated data for Technology and Financial sectors
        assert len(sector_analyzer._sector_data) > 0

    def test_calculate_sector_momentum(self, sector_analyzer, sample_stock_data):
        """Test sector momentum calculation"""
        sector_analyzer.load_sector_data(sample_stock_data)

        momentum = sector_analyzer.calculate_sector_momentum('Technology')

        assert 'momentum_1m' in momentum
        assert 'momentum_3m' in momentum
        assert 'momentum_6m' in momentum

    def test_calculate_sector_momentum_unknown_sector(self, sector_analyzer):
        """Test momentum calculation for unknown sector"""
        momentum = sector_analyzer.calculate_sector_momentum('NonExistent')

        assert momentum['momentum_1m'] == 0.0
        assert momentum['momentum_3m'] == 0.0
        assert momentum['momentum_6m'] == 0.0

    def test_calculate_relative_strength(self, sector_analyzer, sample_stock_data, market_data):
        """Test relative strength calculation"""
        sector_analyzer.load_sector_data(sample_stock_data)

        rs = sector_analyzer.calculate_relative_strength('Technology', market_data)

        # Should be a positive number
        assert rs > 0

    def test_calculate_sector_volatility(self, sector_analyzer, sample_stock_data):
        """Test sector volatility calculation"""
        sector_analyzer.load_sector_data(sample_stock_data)

        volatility = sector_analyzer.calculate_sector_volatility('Technology')

        # Should be positive (annualized volatility)
        assert volatility >= 0

    def test_calculate_correlation_matrix(self, sector_analyzer, sample_stock_data):
        """Test correlation matrix calculation"""
        corr_matrix = sector_analyzer.calculate_correlation_matrix(sample_stock_data)

        # Should have correlations for all symbols
        assert len(corr_matrix) > 0

        # Diagonal should be 1.0
        for symbol in corr_matrix.index:
            assert abs(corr_matrix.loc[symbol, symbol] - 1.0) < 0.001

    def test_detect_market_phase(self, sector_analyzer, market_data):
        """Test market phase detection"""
        phase = sector_analyzer.detect_market_phase(market_data)

        assert isinstance(phase, SectorPhase)

    def test_detect_market_phase_insufficient_data(self, sector_analyzer):
        """Test market phase detection with insufficient data"""
        short_data = pd.DataFrame({
            'Close': [100, 101, 102],
        }, index=pd.date_range(start='2024-01-01', periods=3))

        phase = sector_analyzer.detect_market_phase(short_data)

        # Should return default phase
        assert phase == SectorPhase.LATE_EXPANSION

    def test_get_recommended_sectors(self, sector_analyzer):
        """Test recommended sectors by phase"""
        sectors = sector_analyzer.get_recommended_sectors(SectorPhase.EARLY_EXPANSION)

        assert 'Technology' in sectors
        assert len(sectors) > 0

    def test_calculate_sector_score(self, sector_analyzer, sample_stock_data, market_data):
        """Test sector score calculation"""
        sector_analyzer.load_sector_data(sample_stock_data)

        metrics = sector_analyzer.calculate_sector_score('Technology', market_data)

        assert isinstance(metrics, SectorMetrics)
        assert metrics.sector == 'Technology'
        assert 0 <= metrics.sector_score <= 100

    def test_get_stock_sector_adjustment(self, sector_analyzer, sample_stock_data, market_data):
        """Test position adjustment by sector"""
        adjustment = sector_analyzer.get_stock_sector_adjustment(
            '6758.T',  # Sony (Technology)
            sample_stock_data,
            market_data,
        )

        # Adjustment should be between 0.5 and 1.5
        assert 0.5 <= adjustment <= 1.5

    def test_get_stock_sector_adjustment_unknown(self, sector_analyzer, sample_stock_data):
        """Test adjustment for unknown stock"""
        adjustment = sector_analyzer.get_stock_sector_adjustment(
            '9999.T',  # Unknown
            sample_stock_data,
        )

        assert adjustment == 1.0

    def test_rank_sectors(self, sector_analyzer, sample_stock_data, market_data):
        """Test sector ranking"""
        ranked = sector_analyzer.rank_sectors(sample_stock_data, market_data)

        assert len(ranked) > 0
        # Should be sorted by score descending
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)


class TestSectorMetrics:
    """Test SectorMetrics dataclass"""

    def test_sector_metrics_creation(self):
        """Test creating SectorMetrics"""
        metrics = SectorMetrics(
            sector='Technology',
            momentum_1m=5.0,
            momentum_3m=10.0,
            momentum_6m=15.0,
            relative_strength=1.1,
            volatility=25.0,
            correlation_to_market=0.8,
            sector_score=75.0,
        )

        assert metrics.sector == 'Technology'
        assert metrics.momentum_1m == 5.0
        assert metrics.sector_score == 75.0
