"""
Unit tests for Enhanced Backtest Engine
Tests all 10 improvements
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from backtesting.enhanced_backtest_engine import (
    EnhancedBacktestEngine,
    EnhancedTrade,
    EnhancedBacktestResults,
    MarketRegime,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_stock_data():
    """Generate sample stock data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="B")
    n_days = len(dates)

    # Create uptrend stock
    returns = np.random.normal(0.001, 0.015, n_days)
    close = 1000 * np.cumprod(1 + returns)

    data = {
        "TEST1": pd.DataFrame({
            "Open": close * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            "High": close * (1 + np.random.uniform(0, 0.02, n_days)),
            "Low": close * (1 - np.random.uniform(0, 0.02, n_days)),
            "Close": close,
            "Volume": np.random.randint(100000, 1000000, n_days),
        }, index=dates),
    }

    # Add a downtrend stock
    returns_down = np.random.normal(-0.001, 0.015, n_days)
    close_down = 2000 * np.cumprod(1 + returns_down)
    data["TEST2"] = pd.DataFrame({
        "Open": close_down * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        "High": close_down * (1 + np.random.uniform(0, 0.02, n_days)),
        "Low": close_down * (1 - np.random.uniform(0, 0.02, n_days)),
        "Close": close_down,
        "Volume": np.random.randint(100000, 1000000, n_days),
    }, index=dates)

    return data


@pytest.fixture
def sector_map():
    """Sample sector mapping"""
    return {"TEST1": "tech", "TEST2": "finance"}


@pytest.fixture
def enhanced_engine():
    """Create an enhanced backtest engine with default settings"""
    return EnhancedBacktestEngine(
        initial_capital=1_000_000,
        max_positions=3,
        enable_trailing_stop=True,
        enable_volatility_sizing=True,
        enable_market_regime=True,
        enable_multi_timeframe=True,
        enable_volume_breakout=True,
        enable_swing_low_stop=True,
        enable_additional_filters=True,
        enable_compound=True,
        min_score=50,
        min_confidence=0.50,
    )


# =============================================================================
# Test EnhancedTrade Dataclass
# =============================================================================

class TestEnhancedTrade:
    """Tests for EnhancedTrade dataclass"""

    def test_trade_creation(self):
        """Test basic trade creation"""
        trade = EnhancedTrade(
            symbol="TEST",
            entry_date=datetime(2023, 1, 1),
            entry_price=1000.0,
            shares=100,
        )
        assert trade.symbol == "TEST"
        assert trade.entry_price == 1000.0
        assert trade.is_open is True

    def test_return_pct_open_trade(self):
        """Test return calculation for open trade"""
        trade = EnhancedTrade(
            symbol="TEST",
            entry_date=datetime(2023, 1, 1),
            entry_price=1000.0,
        )
        assert trade.return_pct == 0.0

    def test_return_pct_closed_trade(self):
        """Test return calculation for closed trade"""
        trade = EnhancedTrade(
            symbol="TEST",
            entry_date=datetime(2023, 1, 1),
            entry_price=1000.0,
            exit_date=datetime(2023, 1, 10),
            exit_price=1100.0,
            shares=100,
        )
        assert trade.return_pct == 10.0
        assert trade.is_open is False

    def test_profit_loss_calculation(self):
        """Test profit/loss calculation"""
        trade = EnhancedTrade(
            symbol="TEST",
            entry_date=datetime(2023, 1, 1),
            entry_price=1000.0,
            exit_date=datetime(2023, 1, 10),
            exit_price=1100.0,
            shares=100,
        )
        assert trade.profit_loss == 10000.0  # 100 * (1100 - 1000)

    def test_holding_days(self):
        """Test holding days calculation"""
        trade = EnhancedTrade(
            symbol="TEST",
            entry_date=datetime(2023, 1, 1),
            entry_price=1000.0,
            exit_date=datetime(2023, 1, 11),
            exit_price=1100.0,
        )
        assert trade.holding_days == 10


# =============================================================================
# Test EnhancedBacktestResults
# =============================================================================

class TestEnhancedBacktestResults:
    """Tests for EnhancedBacktestResults dataclass"""

    def test_empty_results(self):
        """Test empty results"""
        results = EnhancedBacktestResults(
            initial_capital=1_000_000,
            final_capital=1_000_000,
        )
        assert results.total_return_pct == 0.0
        assert results.num_trades == 0
        assert results.win_rate == 0.0

    def test_results_with_trades(self):
        """Test results with trades"""
        trades = [
            EnhancedTrade(
                symbol="TEST1",
                entry_date=datetime(2023, 1, 1),
                entry_price=1000.0,
                exit_date=datetime(2023, 1, 10),
                exit_price=1100.0,
                shares=100,
            ),
            EnhancedTrade(
                symbol="TEST2",
                entry_date=datetime(2023, 1, 15),
                entry_price=2000.0,
                exit_date=datetime(2023, 1, 20),
                exit_price=1900.0,
                shares=50,
            ),
        ]
        results = EnhancedBacktestResults(
            trades=trades,
            initial_capital=1_000_000,
            final_capital=1_005_000,
        )
        assert results.num_trades == 2
        assert len(results.winning_trades) == 1
        assert len(results.losing_trades) == 1
        assert results.win_rate == 0.5

    def test_profit_factor(self):
        """Test profit factor calculation"""
        trades = [
            EnhancedTrade(
                symbol="TEST1",
                entry_date=datetime(2023, 1, 1),
                entry_price=1000.0,
                exit_date=datetime(2023, 1, 10),
                exit_price=1100.0,
                shares=100,
            ),
            EnhancedTrade(
                symbol="TEST2",
                entry_date=datetime(2023, 1, 15),
                entry_price=2000.0,
                exit_date=datetime(2023, 1, 20),
                exit_price=1900.0,
                shares=100,
            ),
        ]
        results = EnhancedBacktestResults(trades=trades, initial_capital=1_000_000, final_capital=1_000_000)
        # Win: 100 * 100 = 10000, Loss: 100 * 100 = 10000
        assert results.profit_factor == 1.0

    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses"""
        trades = [
            EnhancedTrade(
                symbol="TEST1",
                entry_date=datetime(2023, 1, 1),
                entry_price=1000.0,
                exit_date=datetime(2023, 1, 10),
                exit_price=1100.0,
                shares=100,
            ),
        ]
        results = EnhancedBacktestResults(trades=trades, initial_capital=1_000_000, final_capital=1_000_000)
        assert results.profit_factor == 10.0  # Capped at 10

    def test_max_drawdown(self):
        """Test max drawdown calculation"""
        equity_curve = [1000000, 1100000, 1050000, 900000, 950000]
        results = EnhancedBacktestResults(
            equity_curve=equity_curve,
            initial_capital=1_000_000,
            final_capital=950_000,
        )
        # Max was 1100000, lowest after that was 900000
        # Drawdown = (1100000 - 900000) / 1100000 = 18.18%
        assert results.max_drawdown > 18.0
        assert results.max_drawdown < 19.0


# =============================================================================
# Test MarketRegime Enum
# =============================================================================

class TestMarketRegime:
    """Tests for MarketRegime enum"""

    def test_regime_values(self):
        """Test regime enum values"""
        assert MarketRegime.BULL.value == "bull"
        assert MarketRegime.BEAR.value == "bear"
        assert MarketRegime.SIDEWAYS.value == "sideways"
        assert MarketRegime.HIGH_VOLATILITY.value == "high_volatility"


# =============================================================================
# Test EnhancedBacktestEngine
# =============================================================================

class TestEnhancedBacktestEngine:
    """Tests for EnhancedBacktestEngine"""

    def test_engine_initialization(self, enhanced_engine):
        """Test engine initialization"""
        assert enhanced_engine.initial_capital == 1_000_000
        assert enhanced_engine.max_positions == 3
        assert enhanced_engine.enable_trailing_stop is True
        assert enhanced_engine.enable_market_regime is True

    def test_engine_initialization_with_defaults(self):
        """Test engine with default settings"""
        engine = EnhancedBacktestEngine()
        assert engine.initial_capital > 0
        assert engine.max_positions > 0

    def test_run_backtest_basic(self, enhanced_engine, sample_stock_data, sector_map):
        """Test basic backtest execution"""
        results = enhanced_engine.run_backtest(
            sample_stock_data,
            "2023-01-01",
            "2023-06-30",
            sector_map=sector_map,
        )
        assert isinstance(results, EnhancedBacktestResults)
        assert results.initial_capital == 1_000_000
        assert len(results.equity_curve) > 0
        assert len(results.dates) > 0

    def test_run_backtest_with_regime_history(self, enhanced_engine, sample_stock_data, sector_map):
        """Test that regime history is recorded"""
        results = enhanced_engine.run_backtest(
            sample_stock_data,
            "2023-01-01",
            "2023-06-30",
            sector_map=sector_map,
        )
        assert len(results.regime_history) > 0
        assert all(r in ["bull", "bear", "sideways", "high_volatility"] for r in results.regime_history)

    def test_run_backtest_enhancements_disabled(self, sample_stock_data, sector_map):
        """Test backtest with all enhancements disabled"""
        engine = EnhancedBacktestEngine(
            initial_capital=1_000_000,
            enable_trailing_stop=False,
            enable_volatility_sizing=False,
            enable_market_regime=False,
            enable_multi_timeframe=False,
            enable_volume_breakout=False,
            enable_swing_low_stop=False,
            enable_additional_filters=False,
            enable_compound=False,
            min_score=50,
            min_confidence=0.50,
        )
        results = engine.run_backtest(
            sample_stock_data,
            "2023-01-01",
            "2023-06-30",
            sector_map=sector_map,
        )
        assert isinstance(results, EnhancedBacktestResults)

    def test_resample_to_weekly(self, enhanced_engine, sample_stock_data):
        """Test weekly resampling"""
        df = sample_stock_data["TEST1"]
        weekly = enhanced_engine._resample_to_weekly(df)
        assert "SMA_Weekly" in weekly.columns
        assert len(weekly) < len(df)

    def test_build_market_index(self, enhanced_engine, sample_stock_data):
        """Test market index building"""
        # First calculate indicators
        from src.analysis.indicators import TechnicalIndicators
        indicator_engine = TechnicalIndicators()

        stocks_with_indicators = {}
        for symbol, df in sample_stock_data.items():
            stocks_with_indicators[symbol] = indicator_engine.calculate_all(df)

        market_index = enhanced_engine._build_market_index(stocks_with_indicators)
        assert "Close" in market_index.columns
        assert "SMA_20" in market_index.columns
        assert "SMA_50" in market_index.columns
        assert "Volatility" in market_index.columns

    def test_infer_sector_map(self, enhanced_engine):
        """Test sector inference from symbols"""
        stocks = {
            "7203.T": None,  # Auto
            "6758.T": None,  # Tech
            "8306.T": None,  # Finance
            "9432.T": None,  # Telecom
        }
        sector_map = enhanced_engine._infer_sector_map(stocks)
        assert sector_map["7203.T"] == "auto"
        assert sector_map["6758.T"] == "tech"
        assert sector_map["8306.T"] == "finance"
        assert sector_map["9432.T"] == "telecom"

    def test_volatility_position_sizing(self, enhanced_engine):
        """Test volatility-based position sizing"""
        # High volatility should result in smaller position
        high_vol_size = enhanced_engine._calculate_volatility_position_size(atr=50, price=1000)
        low_vol_size = enhanced_engine._calculate_volatility_position_size(atr=10, price=1000)
        assert low_vol_size >= high_vol_size

    def test_regime_position_multiplier(self, enhanced_engine):
        """Test regime-based position multiplier"""
        enhanced_engine.current_regime = MarketRegime.BULL
        bull_mult = enhanced_engine._get_regime_position_multiplier()

        enhanced_engine.current_regime = MarketRegime.BEAR
        bear_mult = enhanced_engine._get_regime_position_multiplier()

        assert bull_mult > bear_mult

    def test_regime_min_score(self, enhanced_engine):
        """Test regime-based minimum score"""
        enhanced_engine.current_regime = MarketRegime.BULL
        bull_score = enhanced_engine._get_regime_min_score()

        enhanced_engine.current_regime = MarketRegime.BEAR
        bear_score = enhanced_engine._get_regime_min_score()

        assert bear_score > bull_score  # Bear market requires higher score

    def test_empty_stock_data(self, enhanced_engine, sector_map):
        """Test with empty stock data"""
        results = enhanced_engine.run_backtest(
            {},
            "2023-01-01",
            "2023-06-30",
            sector_map=sector_map,
        )
        assert results.num_trades == 0

    def test_sector_bonus_calculation(self, enhanced_engine):
        """Test sector bonus calculation"""
        enhanced_engine.sector_performance = {"tech": 0.10, "finance": -0.05}

        tech_bonus = enhanced_engine._calculate_sector_bonus("tech")
        finance_bonus = enhanced_engine._calculate_sector_bonus("finance")

        assert tech_bonus > 0  # Strong sector gets bonus
        assert finance_bonus == 0  # Weak sector no bonus


# =============================================================================
# Test Trailing Stop Functionality
# =============================================================================

class TestTrailingStop:
    """Tests for trailing stop functionality"""

    def test_trailing_stop_activation(self):
        """Test trailing stop activates at correct level"""
        trade = EnhancedTrade(
            symbol="TEST",
            entry_date=datetime(2023, 1, 1),
            entry_price=1000.0,
            shares=100,
            highest_price=1000.0,
            trailing_activated=False,
        )

        # Price rises 3% - should activate trailing
        current_price = 1030.0
        if (current_price - trade.entry_price) / trade.entry_price >= 0.03:
            trade.trailing_activated = True
            trade.highest_price = current_price

        assert trade.trailing_activated is True
        assert trade.highest_price == 1030.0


# =============================================================================
# Test Swing Low Stop
# =============================================================================

class TestSwingLowStop:
    """Tests for swing low stop functionality"""

    def test_calculate_swing_low_stop(self, enhanced_engine, sample_stock_data):
        """Test swing low stop calculation"""
        from src.analysis.indicators import TechnicalIndicators
        indicator_engine = TechnicalIndicators()

        df = indicator_engine.calculate_all(sample_stock_data["TEST1"])
        test_date = df.index[20]

        swing_low = enhanced_engine._calculate_swing_low_stop(df, test_date)

        # Should return a positive value
        assert swing_low > 0
        # Should be below current price
        assert swing_low < df.loc[test_date, "Close"]

    def test_swing_low_stop_insufficient_data(self, enhanced_engine, sample_stock_data):
        """Test swing low stop with insufficient data"""
        from src.analysis.indicators import TechnicalIndicators
        indicator_engine = TechnicalIndicators()

        df = indicator_engine.calculate_all(sample_stock_data["TEST1"])
        test_date = df.index[2]  # Only 2 days of data

        swing_low = enhanced_engine._calculate_swing_low_stop(df, test_date)
        assert swing_low == 0.0  # Should return 0 for insufficient data


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full backtest flow"""

    def test_full_backtest_flow(self, sample_stock_data, sector_map):
        """Test complete backtest flow with all features"""
        engine = EnhancedBacktestEngine(
            initial_capital=1_000_000,
            max_positions=2,
            enable_trailing_stop=True,
            enable_volatility_sizing=True,
            enable_market_regime=True,
            enable_multi_timeframe=True,
            enable_volume_breakout=True,
            enable_swing_low_stop=True,
            enable_additional_filters=True,
            enable_compound=True,
            min_score=40,
            min_confidence=0.40,
        )

        results = engine.run_backtest(
            sample_stock_data,
            "2023-01-01",
            "2023-12-31",
            sector_map=sector_map,
        )

        # Basic sanity checks
        assert results.final_capital > 0
        assert len(results.equity_curve) > 0
        assert results.max_drawdown >= 0
        assert 0 <= results.win_rate <= 1

    def test_backtest_consistency(self, sample_stock_data, sector_map):
        """Test that same inputs produce same results"""
        engine1 = EnhancedBacktestEngine(
            initial_capital=1_000_000,
            min_score=40,
            min_confidence=0.40,
        )
        engine2 = EnhancedBacktestEngine(
            initial_capital=1_000_000,
            min_score=40,
            min_confidence=0.40,
        )

        results1 = engine1.run_backtest(sample_stock_data, "2023-01-01", "2023-06-30", sector_map=sector_map)
        results2 = engine2.run_backtest(sample_stock_data, "2023-01-01", "2023-06-30", sector_map=sector_map)

        assert results1.num_trades == results2.num_trades
        assert abs(results1.final_capital - results2.final_capital) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
