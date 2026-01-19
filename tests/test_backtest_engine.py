"""
Tests for backtesting engine
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.backtest_engine import BacktestEngine, Trade, BacktestResults


class TestTrade:
    """Test suite for Trade dataclass"""

    def test_trade_creation(self):
        """Test Trade object creation"""
        trade = Trade(
            symbol="7203.T",
            entry_date=datetime(2024, 1, 1),
            entry_price=1000.0,
            shares=100,
            stop_loss=950.0,
            target_price=1150.0,
            score_at_entry=75.0
        )

        assert trade.symbol == "7203.T"
        assert trade.entry_price == 1000.0
        assert trade.is_open

    def test_trade_is_open(self):
        """Test is_open property"""
        trade = Trade(
            symbol="7203.T",
            entry_date=datetime(2024, 1, 1),
            entry_price=1000.0,
        )
        assert trade.is_open

        trade.exit_date = datetime(2024, 1, 10)
        trade.exit_price = 1100.0
        assert not trade.is_open

    def test_trade_return_pct(self):
        """Test return percentage calculation"""
        trade = Trade(
            symbol="7203.T",
            entry_date=datetime(2024, 1, 1),
            entry_price=1000.0,
            exit_date=datetime(2024, 1, 10),
            exit_price=1100.0,
        )

        assert trade.return_pct == pytest.approx(10.0, rel=0.01)

    def test_trade_return_pct_negative(self):
        """Test negative return percentage"""
        trade = Trade(
            symbol="7203.T",
            entry_date=datetime(2024, 1, 1),
            entry_price=1000.0,
            exit_date=datetime(2024, 1, 10),
            exit_price=900.0,
        )

        assert trade.return_pct == pytest.approx(-10.0, rel=0.01)

    def test_trade_profit_loss(self):
        """Test profit/loss calculation"""
        trade = Trade(
            symbol="7203.T",
            entry_date=datetime(2024, 1, 1),
            entry_price=1000.0,
            shares=100,
            exit_date=datetime(2024, 1, 10),
            exit_price=1100.0,
        )

        assert trade.profit_loss == pytest.approx(10000.0, rel=0.01)

    def test_trade_holding_days(self):
        """Test holding days calculation"""
        trade = Trade(
            symbol="7203.T",
            entry_date=datetime(2024, 1, 1),
            entry_price=1000.0,
            exit_date=datetime(2024, 1, 11),
            exit_price=1100.0,
        )

        assert trade.holding_days == 10


class TestBacktestResults:
    """Test suite for BacktestResults dataclass"""

    @pytest.fixture
    def sample_trades(self) -> list:
        """Create sample trades for testing"""
        return [
            Trade(
                symbol="7203.T", entry_date=datetime(2024, 1, 1),
                entry_price=1000.0, shares=100,
                exit_date=datetime(2024, 1, 10), exit_price=1100.0,
                exit_reason="profit_target"
            ),
            Trade(
                symbol="6758.T", entry_date=datetime(2024, 1, 5),
                entry_price=500.0, shares=200,
                exit_date=datetime(2024, 1, 15), exit_price=450.0,
                exit_reason="stop_loss"
            ),
            Trade(
                symbol="7267.T", entry_date=datetime(2024, 1, 8),
                entry_price=2000.0, shares=50,
                exit_date=datetime(2024, 1, 18), exit_price=2200.0,
                exit_reason="profit_target"
            ),
        ]

    def test_results_creation(self, sample_trades):
        """Test BacktestResults creation"""
        results = BacktestResults(
            trades=sample_trades,
            initial_capital=1000000.0,
            final_capital=1050000.0,
            equity_curve=[1000000, 1010000, 1050000],
            dates=[datetime(2024, 1, 1), datetime(2024, 1, 10), datetime(2024, 1, 20)]
        )

        assert results.num_trades == 3
        assert len(results.winning_trades) == 2
        assert len(results.losing_trades) == 1

    def test_total_return_pct(self, sample_trades):
        """Test total return percentage"""
        results = BacktestResults(
            trades=sample_trades,
            initial_capital=1000000.0,
            final_capital=1100000.0,
        )

        assert results.total_return_pct == pytest.approx(10.0, rel=0.01)

    def test_win_rate(self, sample_trades):
        """Test win rate calculation"""
        results = BacktestResults(trades=sample_trades)

        # 2 winners out of 3 trades = 66.67%
        assert results.win_rate == pytest.approx(2/3, rel=0.01)

    def test_avg_win_loss(self, sample_trades):
        """Test average win and loss calculations"""
        results = BacktestResults(trades=sample_trades)

        assert results.avg_win > 0
        assert results.avg_loss < 0

    def test_profit_factor(self, sample_trades):
        """Test profit factor calculation"""
        results = BacktestResults(trades=sample_trades)

        # Total wins: 10000 + 10000 = 20000
        # Total losses: 10000
        # PF = 20000 / 10000 = 2.0
        assert results.profit_factor == pytest.approx(2.0, rel=0.1)

    def test_profit_factor_no_losses(self):
        """Test profit factor when there are no losses"""
        trades = [
            Trade(
                symbol="7203.T", entry_date=datetime(2024, 1, 1),
                entry_price=1000.0, shares=100,
                exit_date=datetime(2024, 1, 10), exit_price=1100.0,
            ),
        ]
        results = BacktestResults(trades=trades)

        # Should be capped at 10.0 instead of inf
        assert results.profit_factor == 10.0

    def test_profit_factor_no_wins(self):
        """Test profit factor when there are no wins"""
        trades = [
            Trade(
                symbol="7203.T", entry_date=datetime(2024, 1, 1),
                entry_price=1000.0, shares=100,
                exit_date=datetime(2024, 1, 10), exit_price=900.0,
            ),
        ]
        results = BacktestResults(trades=trades)

        assert results.profit_factor == 0.0

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        results = BacktestResults(
            equity_curve=[100, 101, 102, 103, 104, 105]
        )

        assert results.sharpe_ratio > 0

    def test_max_drawdown(self):
        """Test maximum drawdown calculation"""
        results = BacktestResults(
            equity_curve=[100, 110, 105, 95, 100, 90, 95]
        )

        # Max was 110, min after that was 90
        # Drawdown = (110 - 90) / 110 = 18.18%
        assert results.max_drawdown > 0
        assert results.max_drawdown <= 100

    def test_avg_holding_days(self, sample_trades):
        """Test average holding days calculation"""
        results = BacktestResults(trades=sample_trades)

        assert results.avg_holding_days > 0


class TestBacktestEngine:
    """Test suite for BacktestEngine class"""

    @pytest.fixture
    def sample_stock_data(self) -> dict:
        """Create sample stock data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)

        def create_stock_df():
            base_price = 1000
            returns = np.random.randn(100) * 0.02
            close = base_price * np.cumprod(1 + returns)

            return pd.DataFrame({
                'Open': close * (1 + np.random.randn(100) * 0.005),
                'High': close * (1 + np.abs(np.random.randn(100) * 0.01)),
                'Low': close * (1 - np.abs(np.random.randn(100) * 0.01)),
                'Close': close,
                'Volume': np.random.randint(100000, 1000000, 100)
            }, index=dates)

        return {
            "7203.T": create_stock_df(),
            "6758.T": create_stock_df(),
            "7267.T": create_stock_df(),
        }

    def test_engine_initialization(self):
        """Test BacktestEngine initialization"""
        engine = BacktestEngine()

        assert engine.initial_capital > 0
        assert engine.max_positions > 0
        assert engine.commission_rate >= 0
        assert engine.slippage_rate >= 0

    def test_engine_custom_parameters(self):
        """Test BacktestEngine with custom parameters"""
        engine = BacktestEngine(
            initial_capital=2000000,
            max_positions=10,
            commission_rate=0.002,
            slippage_rate=0.001
        )

        assert engine.initial_capital == 2000000
        assert engine.max_positions == 10

    def test_run_backtest_returns_results(self, sample_stock_data):
        """Test run_backtest returns BacktestResults"""
        engine = BacktestEngine()

        results = engine.run_backtest(
            stock_data=sample_stock_data,
            start_date='2024-02-01',
            end_date='2024-03-31'
        )

        assert isinstance(results, BacktestResults)
        assert results.initial_capital == engine.initial_capital
        assert len(results.equity_curve) > 0

    def test_backtest_respects_max_positions(self, sample_stock_data):
        """Test backtest respects max positions limit"""
        engine = BacktestEngine(max_positions=2)

        engine.run_backtest(
            stock_data=sample_stock_data,
            start_date='2024-02-01',
            end_date='2024-03-31'
        )

        # At any point, should not have more than max_positions
        assert len(engine.open_positions) <= engine.max_positions

    def test_backtest_applies_costs(self, sample_stock_data):
        """Test that commission and slippage are applied"""
        engine = BacktestEngine(
            commission_rate=0.001,
            slippage_rate=0.0005
        )

        results = engine.run_backtest(
            stock_data=sample_stock_data,
            start_date='2024-02-01',
            end_date='2024-03-31'
        )

        # If any trades were made, final capital should reflect costs
        if results.num_trades > 0:
            # Costs should have been deducted
            assert results.final_capital != results.initial_capital

    def test_backtest_closes_positions_at_end(self, sample_stock_data):
        """Test all positions are closed at backtest end"""
        engine = BacktestEngine()

        engine.run_backtest(
            stock_data=sample_stock_data,
            start_date='2024-02-01',
            end_date='2024-03-31'
        )

        assert len(engine.open_positions) == 0

    def test_empty_stock_data(self):
        """Test handling of empty stock data"""
        engine = BacktestEngine()

        results = engine.run_backtest(
            stock_data={},
            start_date='2024-02-01',
            end_date='2024-03-31'
        )

        assert results.num_trades == 0
