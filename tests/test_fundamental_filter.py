"""
Tests for fundamental filter module
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.fundamental_filter import (
    FundamentalData,
    FundamentalFilter,
    DEFAULT_FUNDAMENTALS,
)


@pytest.fixture
def fundamental_filter():
    """Create FundamentalFilter instance"""
    return FundamentalFilter(use_api=False)


class TestFundamentalData:
    """Test FundamentalData dataclass"""

    def test_quality_score_calculation(self):
        """Test quality score calculation"""
        # Good fundamentals
        data = FundamentalData(
            symbol='TEST',
            pe_ratio=15.0,
            pbr=1.5,
            dividend_yield=3.0,
            roe=15.0,
            debt_to_equity=0.3,
        )

        score = data.quality_score

        # Should be above average (50)
        assert score > 50

    def test_quality_score_poor_fundamentals(self):
        """Test quality score with poor fundamentals"""
        data = FundamentalData(
            symbol='TEST',
            pe_ratio=50.0,  # Expensive
            pbr=5.0,        # Overvalued
            roe=-5.0,       # Negative ROE
            debt_to_equity=3.0,  # High debt
        )

        score = data.quality_score

        # Should be below average
        assert score < 50

    def test_quality_score_missing_data(self):
        """Test quality score with missing data"""
        data = FundamentalData(symbol='TEST')

        score = data.quality_score

        # Should return base score
        assert score == 50

    def test_is_valid(self):
        """Test data validity check"""
        data = FundamentalData(symbol='TEST')

        # Should be valid (just created)
        assert data.is_valid


class TestFundamentalFilter:
    """Test FundamentalFilter class"""

    def test_load_default_data(self, fundamental_filter):
        """Test default data loading"""
        # Should have cached data for known stocks
        data = fundamental_filter.get_fundamentals('7203.T')

        assert data.symbol == '7203.T'
        assert data.pe_ratio is not None

    def test_get_fundamentals_unknown_stock(self, fundamental_filter):
        """Test getting fundamentals for unknown stock"""
        data = fundamental_filter.get_fundamentals('9999.T')

        assert data.symbol == '9999.T'
        # Should return empty data
        assert data.pe_ratio is None

    def test_filter_by_pe(self, fundamental_filter):
        """Test P/E ratio filtering"""
        symbols = list(DEFAULT_FUNDAMENTALS.keys())

        # Filter for reasonable P/E (0-30)
        filtered = fundamental_filter.filter_by_pe(symbols, min_pe=0, max_pe=30)

        # Should exclude high P/E stocks
        assert '6861.T' not in filtered  # Keyence has P/E ~45

    def test_filter_by_dividend(self, fundamental_filter):
        """Test dividend yield filtering"""
        symbols = list(DEFAULT_FUNDAMENTALS.keys())

        # Filter for dividend yield >= 3%
        filtered = fundamental_filter.filter_by_dividend(symbols, min_yield=3.0)

        # Should include high dividend stocks
        assert '8316.T' in filtered  # SMFG ~3.8%

    def test_filter_by_quality(self, fundamental_filter):
        """Test quality score filtering"""
        symbols = list(DEFAULT_FUNDAMENTALS.keys())

        filtered = fundamental_filter.filter_by_quality(symbols, min_score=60)

        # Should filter out low quality stocks
        assert len(filtered) <= len(symbols)

    def test_get_quality_adjustment(self, fundamental_filter):
        """Test position size adjustment"""
        adjustment = fundamental_filter.get_quality_adjustment('7203.T')

        # Should be between 0.5 and 1.5
        assert 0.5 <= adjustment <= 1.5

    def test_rank_by_value(self, fundamental_filter):
        """Test value ranking"""
        symbols = list(DEFAULT_FUNDAMENTALS.keys())[:5]

        ranked = fundamental_filter.rank_by_value(symbols)

        assert len(ranked) == len(symbols)
        # Should be sorted by score descending
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_by_growth(self, fundamental_filter):
        """Test growth ranking"""
        symbols = list(DEFAULT_FUNDAMENTALS.keys())[:5]

        ranked = fundamental_filter.rank_by_growth(symbols)

        assert len(ranked) == len(symbols)

    def test_get_composite_score(self, fundamental_filter):
        """Test composite score calculation"""
        score = fundamental_filter.get_composite_score('7203.T')

        # Should be between 0 and 100
        assert 0 <= score <= 100

    def test_get_all_scores(self, fundamental_filter):
        """Test getting all scores for multiple symbols"""
        symbols = ['7203.T', '6758.T', '8306.T']

        scores = fundamental_filter.get_all_scores(symbols)

        assert len(scores) == 3
        assert '7203.T' in scores
        assert 'quality_score' in scores['7203.T']
        assert 'composite_score' in scores['7203.T']


class TestDefaultFundamentals:
    """Test default fundamental data"""

    def test_all_stocks_have_data(self):
        """Test that all mapped stocks have fundamental data"""
        for symbol in DEFAULT_FUNDAMENTALS:
            data = DEFAULT_FUNDAMENTALS[symbol]
            # Should have at least some data
            assert 'market_cap' in data or 'pe_ratio' in data

    def test_market_cap_reasonable(self):
        """Test that market caps are reasonable"""
        for symbol, data in DEFAULT_FUNDAMENTALS.items():
            if 'market_cap' in data and data['market_cap']:
                # Should be > 1 trillion yen for major stocks
                assert data['market_cap'] > 1_000_000_000_000
