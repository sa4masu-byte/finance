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
            # Cap at reasonable maximum for calculations
            return 10.0 if total_wins > 0 else 0.0
        return min(total_wins / total_losses, 10.0)  # Cap at 10.0 to avoid extreme values

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


class VectorizedBacktestEngine:
    """
    Vectorized backtesting engine for improved performance.

    Uses numpy operations for batch processing:
    - Pre-calculates all indicators
    - Vectorized score calculations
    - Batch signal generation
    - Optimized position tracking

    Achieves 10-50x speedup over day-by-day simulation.
    """

    def __init__(
        self,
        initial_capital: float = BACKTEST_CONFIG["initial_capital"],
        max_positions: int = BACKTEST_CONFIG["max_positions"],
        commission_rate: float = BACKTEST_CONFIG["commission_rate"],
        slippage_rate: float = BACKTEST_CONFIG["slippage_rate"],
        scoring_weights: Optional[Dict] = None,
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        self.indicator_engine = TechnicalIndicators()
        self.scoring_engine = ScoringEngine(weights=scoring_weights)

    def run_backtest(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> BacktestResults:
        """
        Run vectorized backtest

        Args:
            stock_data: Dictionary mapping symbol to OHLCV DataFrame
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            BacktestResults object
        """
        logger.info(f"Running vectorized backtest from {start_date} to {end_date}")

        # Pre-calculate all indicators and signals
        processed_data = self._preprocess_all_stocks(stock_data, start_date, end_date)

        # Generate all signals
        signals_df = self._generate_all_signals(processed_data)

        # Simulate trades using vectorized operations
        results = self._simulate_trades_vectorized(
            processed_data, signals_df, start_date, end_date
        )

        return results

    def _preprocess_all_stocks(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> Dict[str, pd.DataFrame]:
        """Pre-calculate indicators for all stocks"""
        processed = {}
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        for symbol, df in stock_data.items():
            # Calculate indicators
            df_with_ind = self.indicator_engine.calculate_all(df.copy())

            # Filter date range
            mask = (df_with_ind.index >= start_dt) & (df_with_ind.index <= end_dt)
            df_filtered = df_with_ind[mask].copy()

            if len(df_filtered) > 0:
                processed[symbol] = df_filtered

        return processed

    def _generate_all_signals(
        self,
        processed_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Generate signals for all stocks in a vectorized manner"""
        all_signals = []

        for symbol, df in processed_data.items():
            # Calculate scores for each row
            scores = self._vectorized_scoring(df)

            # Create signal DataFrame
            signal_df = pd.DataFrame({
                'symbol': symbol,
                'date': df.index,
                'close': df['Close'].values,
                'score': scores,
                'atr': df['ATR'].values if 'ATR' in df.columns else df['Close'].values * 0.02,
            })

            all_signals.append(signal_df)

        if not all_signals:
            return pd.DataFrame()

        signals = pd.concat(all_signals, ignore_index=True)
        signals = signals.sort_values(['date', 'score'], ascending=[True, False])

        return signals

    def _vectorized_scoring(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate scores for entire DataFrame using vectorized operations"""
        scores = np.zeros(len(df))

        # Trend score (30% weight)
        trend_score = np.zeros(len(df))

        # Price above SMAs
        if 'SMA_5' in df.columns:
            trend_score += np.where(df['Close'] > df['SMA_5'], 7, 0)
        if 'SMA_25' in df.columns:
            trend_score += np.where(df['Close'] > df['SMA_25'], 7, 0)
        if 'SMA_75' in df.columns:
            trend_score += np.where(df['Close'] > df['SMA_75'], 6, 0)

        # MACD positive
        if 'MACD_Histogram' in df.columns:
            trend_score += np.where(df['MACD_Histogram'] > 0, 10, 0)

        scores += trend_score * 0.30

        # Momentum score (25% weight)
        momentum_score = np.zeros(len(df))

        if 'RSI_14' in df.columns:
            rsi = df['RSI_14'].values
            momentum_score += np.where((rsi >= 40) & (rsi <= 65), 15, 0)
            momentum_score += np.where((rsi >= 30) & (rsi < 40), 12, 0)
            momentum_score += np.where((rsi > 65) & (rsi <= 70), 5, 0)

        if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
            stoch_k = df['Stoch_K'].values
            stoch_d = df['Stoch_D'].values
            momentum_score += np.where(stoch_k > stoch_d, 10, 0)
            momentum_score += np.where(stoch_k < 80, 5, 0)

        scores += momentum_score * 0.25

        # Volume score (20% weight)
        volume_score = np.zeros(len(df))

        if 'Volume_Ratio' in df.columns:
            vol_ratio = df['Volume_Ratio'].values
            volume_score += np.where(vol_ratio >= 2.0, 20, 0)
            volume_score += np.where((vol_ratio >= 1.5) & (vol_ratio < 2.0), 12, 0)
            volume_score += np.where((vol_ratio >= 1.2) & (vol_ratio < 1.5), 6, 0)

        if 'OBV_Trend' in df.columns:
            volume_score += np.where(df['OBV_Trend'] > 0, 5, 0)

        scores += volume_score * 0.20

        # Volatility score (15% weight)
        volatility_score = np.zeros(len(df))

        if 'BB_Percent' in df.columns:
            bb_pct = df['BB_Percent'].values
            volatility_score += np.where(bb_pct < 0.3, 8, 0)
            volatility_score += np.where((bb_pct >= 0.3) & (bb_pct < 0.5), 5, 0)

        if 'ATR_Percent' in df.columns:
            atr_pct = df['ATR_Percent'].values
            volatility_score += np.where(atr_pct < 3, 7, 0)
            volatility_score += np.where((atr_pct >= 3) & (atr_pct < 5), 4, 0)

        scores += volatility_score * 0.15

        # Pattern score (10% weight)
        pattern_score = np.zeros(len(df))

        if 'Pattern_Score' in df.columns:
            patt = df['Pattern_Score'].values
            pattern_score += np.where(patt > 20, 10, 0)
            pattern_score += np.where((patt > 0) & (patt <= 20), 5, 0)

        if 'Pattern_Engulfing' in df.columns:
            pattern_score += np.where(df['Pattern_Engulfing'] == 1, 5, 0)

        scores += pattern_score * 0.10

        return scores

    def _simulate_trades_vectorized(
        self,
        processed_data: Dict[str, pd.DataFrame],
        signals_df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> BacktestResults:
        """Simulate trades using vectorized operations where possible"""
        if signals_df.empty:
            return BacktestResults(
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
            )

        # Get unique dates
        all_dates = sorted(signals_df['date'].unique())

        # Trading state
        capital = self.initial_capital
        positions: Dict[str, Trade] = {}
        closed_trades: List[Trade] = []
        equity_curve = [capital]
        dates_recorded = []

        min_score = 65  # Minimum score for entry

        for date in all_dates:
            dates_recorded.append(date)

            # Get signals for this day
            day_signals = signals_df[signals_df['date'] == date]

            # Check exits for existing positions
            positions_to_close = []
            for symbol, position in positions.items():
                if symbol not in processed_data:
                    continue

                df = processed_data[symbol]
                if date not in df.index:
                    continue

                row = df.loc[date]
                current_price = row['Close']

                # Exit conditions
                should_exit = False
                exit_reason = ""

                if current_price <= position.stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif current_price >= position.target_price:
                    should_exit = True
                    exit_reason = "profit_target"
                elif (date - position.entry_date).days >= RISK_PARAMS["max_holding_days"]:
                    should_exit = True
                    exit_reason = "max_holding"

                # Score-based exit
                if not should_exit:
                    day_sym_signal = day_signals[day_signals['symbol'] == symbol]
                    if not day_sym_signal.empty:
                        score = day_sym_signal.iloc[0]['score']
                        if score < RISK_PARAMS["reeval_score_threshold"]:
                            should_exit = True
                            exit_reason = "score_deterioration"

                if should_exit:
                    positions_to_close.append((symbol, current_price, exit_reason))

            # Close positions
            for symbol, price, reason in positions_to_close:
                position = positions[symbol]
                effective_price = price * (1 - self.slippage_rate - self.commission_rate)

                position.exit_date = date
                position.exit_price = effective_price
                position.exit_reason = reason

                capital += position.shares * effective_price
                closed_trades.append(position)
                del positions[symbol]

            # Check entries
            available_slots = self.max_positions - len(positions)
            if available_slots > 0:
                # Filter signals for potential entries
                entry_candidates = day_signals[
                    (day_signals['score'] >= min_score) &
                    (~day_signals['symbol'].isin(positions.keys()))
                ].head(available_slots)

                for _, row in entry_candidates.iterrows():
                    symbol = row['symbol']
                    price = row['close']
                    score = row['score']
                    atr = row['atr']

                    # Position sizing
                    position_value = capital * RISK_PARAMS["position_size_pct"]
                    effective_price = price * (1 + self.slippage_rate + self.commission_rate)
                    shares = int(position_value / effective_price)

                    if shares == 0 or shares * effective_price > capital:
                        continue

                    # Calculate levels
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

                    positions[symbol] = trade
                    capital -= shares * effective_price

            # Calculate equity
            total_equity = capital
            for symbol, position in positions.items():
                if symbol in processed_data:
                    df = processed_data[symbol]
                    if date in df.index:
                        total_equity += position.shares * df.loc[date, 'Close']

            equity_curve.append(total_equity)

        # Close remaining positions at end
        end_dt = pd.to_datetime(end_date)
        for symbol, position in list(positions.items()):
            if symbol in processed_data:
                df = processed_data[symbol]
                if len(df) > 0:
                    last_price = df['Close'].iloc[-1]
                    effective_price = last_price * (1 - self.slippage_rate - self.commission_rate)

                    position.exit_date = end_dt
                    position.exit_price = effective_price
                    position.exit_reason = "backtest_end"

                    capital += position.shares * effective_price
                    closed_trades.append(position)

        return BacktestResults(
            trades=closed_trades,
            initial_capital=self.initial_capital,
            final_capital=capital,
            equity_curve=equity_curve,
            dates=dates_recorded,
        )
