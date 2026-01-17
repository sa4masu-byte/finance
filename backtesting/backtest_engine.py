"""
Backtesting engine for swing trading strategy
Simulates trading based on technical indicator scores
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import BACKTEST_CONFIG, RISK_PARAMS
from src.analysis.indicators import TechnicalIndicators
from backtesting.scoring_engine import ScoringEngine

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: int = 0
    stop_loss: float = 0.0
    target_price: float = 0.0
    exit_reason: str = ""
    score_at_entry: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def return_pct(self) -> float:
        if self.exit_price is None:
            return 0.0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100

    @property
    def profit_loss(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def holding_days(self) -> int:
        if self.exit_date is None:
            return 0
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResults:
    """Results from a backtest run"""
    trades: List[Trade] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    dates: List[datetime] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return ((self.final_capital - self.initial_capital) / self.initial_capital) * 100

    @property
    def num_trades(self) -> int:
        return len([t for t in self.trades if not t.is_open])

    @property
    def winning_trades(self) -> List[Trade]:
        return [t for t in self.trades if not t.is_open and t.return_pct > 0]

    @property
    def losing_trades(self) -> List[Trade]:
        return [t for t in self.trades if not t.is_open and t.return_pct < 0]

    @property
    def win_rate(self) -> float:
        if self.num_trades == 0:
            return 0.0
        return len(self.winning_trades) / self.num_trades

    @property
    def avg_win(self) -> float:
        if not self.winning_trades:
            return 0.0
        return np.mean([t.return_pct for t in self.winning_trades])

    @property
    def avg_loss(self) -> float:
        if not self.losing_trades:
            return 0.0
        return np.mean([t.return_pct for t in self.losing_trades])

    @property
    def profit_factor(self) -> float:
        total_wins = sum(t.profit_loss for t in self.winning_trades)
        total_losses = abs(sum(t.profit_loss for t in self.losing_trades))

        if total_losses == 0:
            return float('inf') if total_wins > 0 else 0.0
        return total_wins / total_losses

    @property
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)"""
        if len(self.equity_curve) < 2:
            return 0.0

        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if returns.std() == 0:
            return 0.0

        return (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized

    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        if not self.equity_curve:
            return 0.0

        equity = pd.Series(self.equity_curve)
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        return abs(drawdown.min()) * 100

    @property
    def avg_holding_days(self) -> float:
        closed_trades = [t for t in self.trades if not t.is_open]
        if not closed_trades:
            return 0.0
        return np.mean([t.holding_days for t in closed_trades])


class BacktestEngine:
    """
    Backtesting engine for swing trading strategy
    """

    def __init__(
        self,
        initial_capital: float = BACKTEST_CONFIG["initial_capital"],
        max_positions: int = BACKTEST_CONFIG["max_positions"],
        commission_rate: float = BACKTEST_CONFIG["commission_rate"],
        slippage_rate: float = BACKTEST_CONFIG["slippage_rate"],
        scoring_weights: Optional[Dict] = None,
    ):
        """
        Initialize backtest engine

        Args:
            initial_capital: Starting capital
            max_positions: Maximum simultaneous positions
            commission_rate: Commission per trade (e.g., 0.001 = 0.1%)
            slippage_rate: Slippage per trade
            scoring_weights: Custom scoring weights for optimization
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # Initialize engines
        self.indicator_engine = TechnicalIndicators()
        self.scoring_engine = ScoringEngine(weights=scoring_weights)

        # Trading state
        self.open_positions: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve = []
        self.dates = []

    def run_backtest(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> BacktestResults:
        """
        Run backtest on historical data

        Args:
            stock_data: Dictionary mapping symbol to OHLCV DataFrame
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            BacktestResults object with performance metrics
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        logger.info(f"Testing {len(stock_data)} stocks")

        # Reset state
        self.capital = self.initial_capital
        self.open_positions = []
        self.closed_trades = []
        self.equity_curve = [self.initial_capital]
        self.dates = []

        # Calculate indicators for all stocks
        stocks_with_indicators = {}
        for symbol, df in stock_data.items():
            df_with_ind = self.indicator_engine.calculate_all(df)
            stocks_with_indicators[symbol] = df_with_ind

        # Get date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Get all unique dates
        all_dates = set()
        for df in stocks_with_indicators.values():
            all_dates.update(df.index)
        all_dates = sorted([d for d in all_dates if start_dt <= d <= end_dt])

        # Simulate trading day by day
        for current_date in all_dates:
            self.dates.append(current_date)
            self._process_trading_day(current_date, stocks_with_indicators)

            # Record equity
            total_equity = self.capital + sum(
                pos.shares * self._get_price(stocks_with_indicators[pos.symbol], current_date, "Close")
                for pos in self.open_positions
                if self._get_price(stocks_with_indicators[pos.symbol], current_date, "Close") is not None
            )
            self.equity_curve.append(total_equity)

        # Close any remaining positions at end date
        self._close_all_positions(end_dt, stocks_with_indicators, "backtest_end")

        # Create results
        results = BacktestResults(
            trades=self.closed_trades,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            equity_curve=self.equity_curve,
            dates=self.dates,
        )

        logger.info(f"Backtest complete: {results.num_trades} trades, "
                    f"{results.win_rate:.1%} win rate, "
                    f"{results.total_return_pct:.2f}% return")

        return results

    def _process_trading_day(self, date: datetime, stocks: Dict[str, pd.DataFrame]):
        """Process a single trading day"""
        # 1. Check exit conditions for open positions
        self._check_exits(date, stocks)

        # 2. Look for new entry opportunities
        if len(self.open_positions) < self.max_positions:
            self._check_entries(date, stocks)

    def _check_exits(self, date: datetime, stocks: Dict[str, pd.DataFrame]):
        """Check if any open positions should be closed"""
        for position in self.open_positions[:]:  # Copy list to allow modification
            symbol = position.symbol
            df = stocks[symbol]

            current_price = self._get_price(df, date, "Close")
            if current_price is None:
                continue

            should_exit, reason = self._should_exit(position, date, current_price, df)

            if should_exit:
                self._close_position(position, date, current_price, reason)

    def _should_exit(
        self,
        position: Trade,
        date: datetime,
        current_price: float,
        df: pd.DataFrame
    ) -> Tuple[bool, str]:
        """Determine if position should be exited"""
        # 1. Stop loss
        if current_price <= position.stop_loss:
            return True, "stop_loss"

        # 2. Profit target
        if current_price >= position.target_price:
            return True, "profit_target"

        # 3. Maximum holding period
        holding_days = (date - position.entry_date).days
        if holding_days >= RISK_PARAMS["max_holding_days"]:
            return True, "max_holding_period"

        # 4. Re-evaluation: score drops significantly
        indicators = self._get_indicators_at_date(df, date)
        if indicators:
            score_result = self.scoring_engine.calculate_score(indicators)
            if score_result["total_score"] < RISK_PARAMS["reeval_score_threshold"]:
                return True, "score_deterioration"

        return False, ""

    def _check_entries(self, date: datetime, stocks: Dict[str, pd.DataFrame]):
        """Look for new entry opportunities"""
        # Get candidates
        candidates = []

        for symbol, df in stocks.items():
            # Skip if already in position
            if any(pos.symbol == symbol for pos in self.open_positions):
                continue

            indicators = self._get_indicators_at_date(df, date)
            if not indicators:
                continue

            # Calculate score
            score_result = self.scoring_engine.calculate_score(indicators)

            # Check if meets criteria
            if self.scoring_engine.should_recommend(score_result):
                current_price = indicators.get("Close")
                atr = indicators.get("ATR")

                if current_price and atr:
                    candidates.append({
                        "symbol": symbol,
                        "score": score_result["total_score"],
                        "price": current_price,
                        "atr": atr,
                    })

        # Sort by score (descending)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Enter positions (up to max_positions)
        available_slots = self.max_positions - len(self.open_positions)
        for candidate in candidates[:available_slots]:
            self._enter_position(
                symbol=candidate["symbol"],
                date=date,
                price=candidate["price"],
                atr=candidate["atr"],
                score=candidate["score"],
            )

    def _enter_position(
        self,
        symbol: str,
        date: datetime,
        price: float,
        atr: float,
        score: float,
    ):
        """Enter a new position"""
        # Calculate position size
        position_value = self.capital * RISK_PARAMS["position_size_pct"]

        # Apply costs
        effective_price = price * (1 + self.slippage_rate + self.commission_rate)

        shares = int(position_value / effective_price)
        if shares == 0:
            return

        cost = shares * effective_price

        # Check if we have enough capital
        if cost > self.capital:
            return

        # Calculate stop loss and target
        stop_loss = price - (atr * RISK_PARAMS["stop_loss_atr_multiplier"])
        target_price = price * (1 + RISK_PARAMS["profit_target_max"])

        # Create trade
        trade = Trade(
            symbol=symbol,
            entry_date=date,
            entry_price=effective_price,
            shares=shares,
            stop_loss=stop_loss,
            target_price=target_price,
            score_at_entry=score,
        )

        self.open_positions.append(trade)
        self.capital -= cost

        logger.debug(f"Entered {symbol} at ¥{effective_price:.2f} "
                     f"({shares} shares, score={score:.1f})")

    def _close_position(self, position: Trade, date: datetime, price: float, reason: str):
        """Close an open position"""
        # Apply costs
        effective_price = price * (1 - self.slippage_rate - self.commission_rate)

        # Update trade
        position.exit_date = date
        position.exit_price = effective_price
        position.exit_reason = reason

        # Return capital
        proceeds = position.shares * effective_price
        self.capital += proceeds

        # Move to closed trades
        self.open_positions.remove(position)
        self.closed_trades.append(position)

        logger.debug(f"Closed {position.symbol} at ¥{effective_price:.2f} "
                     f"({position.return_pct:+.2f}%, reason={reason})")

    def _close_all_positions(
        self,
        date: datetime,
        stocks: Dict[str, pd.DataFrame],
        reason: str
    ):
        """Close all open positions"""
        for position in self.open_positions[:]:
            symbol = position.symbol
            df = stocks[symbol]
            current_price = self._get_price(df, date, "Close")

            if current_price:
                self._close_position(position, date, current_price, reason)

    def _get_indicators_at_date(self, df: pd.DataFrame, date: datetime) -> Optional[Dict]:
        """Get indicator values at a specific date"""
        try:
            if date not in df.index:
                return None

            row = df.loc[date]
            return row.to_dict()
        except Exception as e:
            logger.warning(f"Error getting indicators at {date}: {e}")
            return None

    def _get_price(self, df: pd.DataFrame, date: datetime, column: str) -> Optional[float]:
        """Get price at a specific date"""
        try:
            if date not in df.index:
                return None
            return float(df.loc[date, column])
        except Exception:
            return None
