"""
Enhanced Backtesting engine with all 10 improvements
1. Trailing Stop
2. Volatility-based Position Sizing
3. Sector Momentum
4. Multi-timeframe Analysis
5. Volume Breakout
6. Improved Stop Loss (Swing Low)
7. Market Regime Detection
8. Additional Filters
9. Compound Interest
10. ML Scoring (optional)
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    BACKTEST_CONFIG, RISK_PARAMS, ENHANCED_RISK_PARAMS,
    MARKET_REGIME_PARAMS, MULTI_TIMEFRAME_PARAMS,
    VOLUME_BREAKOUT_PARAMS, ADDITIONAL_FILTERS, ML_SCORING_PARAMS
)
from src.analysis.indicators import TechnicalIndicators
from backtesting.scoring_engine import ScoringEngine

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class EnhancedTrade:
    """Represents a single trade with enhanced tracking"""
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
    sector: str = ""

    # Enhanced fields
    trailing_stop: float = 0.0
    highest_price: float = 0.0
    trailing_activated: bool = False
    swing_low_stop: float = 0.0
    atr_at_entry: float = 0.0
    market_regime: str = ""

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
class EnhancedBacktestResults:
    """Enhanced results from a backtest run"""
    trades: List[EnhancedTrade] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    dates: List[datetime] = field(default_factory=list)
    regime_history: List[str] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return ((self.final_capital - self.initial_capital) / self.initial_capital) * 100

    @property
    def num_trades(self) -> int:
        return len([t for t in self.trades if not t.is_open])

    @property
    def winning_trades(self) -> List[EnhancedTrade]:
        return [t for t in self.trades if not t.is_open and t.return_pct > 0]

    @property
    def losing_trades(self) -> List[EnhancedTrade]:
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
            return 10.0 if total_wins > 0 else 0.0
        return min(total_wins / total_losses, 10.0)

    @property
    def sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    @property
    def max_drawdown(self) -> float:
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


class EnhancedBacktestEngine:
    """
    Enhanced Backtesting engine with all 10 improvements
    """

    def __init__(
        self,
        initial_capital: float = BACKTEST_CONFIG["initial_capital"],
        max_positions: int = BACKTEST_CONFIG["max_positions"],
        commission_rate: float = BACKTEST_CONFIG["commission_rate"],
        slippage_rate: float = BACKTEST_CONFIG["slippage_rate"],
        scoring_weights: Optional[Dict] = None,
        # Enhanced parameters
        enable_trailing_stop: bool = True,
        enable_volatility_sizing: bool = True,
        enable_market_regime: bool = True,
        enable_multi_timeframe: bool = True,
        enable_volume_breakout: bool = True,
        enable_swing_low_stop: bool = True,
        enable_additional_filters: bool = True,
        enable_compound: bool = True,
        min_score: float = 55,
        min_confidence: float = 0.60,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_capital = initial_capital  # Track for compound interest
        self.max_positions = max_positions
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # Enhancement flags
        self.enable_trailing_stop = enable_trailing_stop
        self.enable_volatility_sizing = enable_volatility_sizing
        self.enable_market_regime = enable_market_regime
        self.enable_multi_timeframe = enable_multi_timeframe
        self.enable_volume_breakout = enable_volume_breakout
        self.enable_swing_low_stop = enable_swing_low_stop
        self.enable_additional_filters = enable_additional_filters
        self.enable_compound = enable_compound
        self.min_score = min_score
        self.min_confidence = min_confidence

        # Initialize engines
        self.indicator_engine = TechnicalIndicators()
        self.scoring_engine = ScoringEngine(weights=scoring_weights)

        # Trading state
        self.open_positions: List[EnhancedTrade] = []
        self.closed_trades: List[EnhancedTrade] = []
        self.equity_curve = []
        self.dates = []
        self.regime_history = []

        # Market regime state
        self.current_regime = MarketRegime.SIDEWAYS
        self.market_index_data: Optional[pd.DataFrame] = None

        # Sector tracking
        self.sector_performance: Dict[str, float] = {}

    def run_backtest(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> EnhancedBacktestResults:
        """
        Run enhanced backtest on historical data
        """
        logger.info(f"Running enhanced backtest from {start_date} to {end_date}")
        logger.info(f"Testing {len(stock_data)} stocks with all enhancements")

        # Reset state
        self.capital = self.initial_capital
        self.max_capital = self.initial_capital
        self.open_positions = []
        self.closed_trades = []
        self.equity_curve = [self.initial_capital]
        self.dates = []
        self.regime_history = []
        self.sector_performance = {}

        # Default sector map if not provided
        if sector_map is None:
            sector_map = self._infer_sector_map(stock_data)

        # Calculate indicators for all stocks
        stocks_with_indicators = {}
        weekly_data = {}

        for symbol, df in stock_data.items():
            df_with_ind = self.indicator_engine.calculate_all(df)
            stocks_with_indicators[symbol] = df_with_ind

            # Generate weekly data for multi-timeframe analysis
            if self.enable_multi_timeframe:
                weekly_data[symbol] = self._resample_to_weekly(df_with_ind)

        # Get date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Get all unique dates
        all_dates = set()
        for df in stocks_with_indicators.values():
            all_dates.update(df.index)
        all_dates = sorted([d for d in all_dates if start_dt <= d <= end_dt])

        # Build market index for regime detection
        if self.enable_market_regime:
            self.market_index_data = self._build_market_index(stocks_with_indicators)

        # Simulate trading day by day
        for current_date in all_dates:
            self.dates.append(current_date)

            # Update market regime
            if self.enable_market_regime:
                self._update_market_regime(current_date)
            self.regime_history.append(self.current_regime.value)

            # Update sector performance
            self._update_sector_performance(current_date, stocks_with_indicators, sector_map)

            # Process trading day
            self._process_trading_day(
                current_date,
                stocks_with_indicators,
                weekly_data,
                sector_map
            )

            # Record equity
            total_equity = self.capital + sum(
                pos.shares * self._get_price(stocks_with_indicators[pos.symbol], current_date, "Close")
                for pos in self.open_positions
                if self._get_price(stocks_with_indicators[pos.symbol], current_date, "Close") is not None
            )
            self.equity_curve.append(total_equity)

            # Update max capital for compound interest
            if self.enable_compound and total_equity > self.max_capital:
                profit = total_equity - self.max_capital
                if profit / self.initial_capital >= ENHANCED_RISK_PARAMS["compound_threshold"]:
                    reinvest = profit * ENHANCED_RISK_PARAMS["compound_reinvest_pct"]
                    self.max_capital = self.initial_capital + reinvest
                    logger.debug(f"Compound reinvestment: max_capital updated to {self.max_capital:.0f}")

        # Close remaining positions
        self._close_all_positions(end_dt, stocks_with_indicators, "backtest_end")

        # Create results
        results = EnhancedBacktestResults(
            trades=self.closed_trades,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            equity_curve=self.equity_curve,
            dates=self.dates,
            regime_history=self.regime_history,
        )

        logger.info(f"Enhanced backtest complete: {results.num_trades} trades, "
                    f"{results.win_rate:.1%} win rate, "
                    f"{results.total_return_pct:.2f}% return")

        return results

    def _resample_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample daily data to weekly for multi-timeframe analysis"""
        weekly = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        # Calculate weekly SMA
        weekly['SMA_Weekly'] = weekly['Close'].rolling(
            window=MULTI_TIMEFRAME_PARAMS['weekly_sma_period']
        ).mean()

        return weekly

    def _build_market_index(self, stocks: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build market index from average of all stocks"""
        all_closes = []
        for symbol, df in stocks.items():
            normalized = df['Close'] / df['Close'].iloc[0] * 100
            all_closes.append(normalized)

        if not all_closes:
            return pd.DataFrame()

        market_df = pd.concat(all_closes, axis=1).mean(axis=1).to_frame('Close')

        # Calculate SMAs for regime detection
        market_df['SMA_20'] = market_df['Close'].rolling(
            window=MARKET_REGIME_PARAMS['sma_short']
        ).mean()
        market_df['SMA_50'] = market_df['Close'].rolling(
            window=MARKET_REGIME_PARAMS['sma_long']
        ).mean()

        # Calculate volatility
        market_df['Volatility'] = market_df['Close'].pct_change().rolling(
            window=MARKET_REGIME_PARAMS['lookback_days']
        ).std()

        return market_df

    def _update_market_regime(self, date: datetime):
        """Detect current market regime (改善7)"""
        if self.market_index_data is None or date not in self.market_index_data.index:
            return

        row = self.market_index_data.loc[date]
        sma_20 = row.get('SMA_20')
        sma_50 = row.get('SMA_50')
        volatility = row.get('Volatility')

        if pd.isna(sma_20) or pd.isna(sma_50):
            return

        # Calculate SMA ratio
        sma_ratio = (sma_20 - sma_50) / sma_50 if sma_50 != 0 else 0

        # Check volatility first
        if volatility and volatility > MARKET_REGIME_PARAMS['volatility_threshold']:
            self.current_regime = MarketRegime.HIGH_VOLATILITY
        elif sma_ratio > MARKET_REGIME_PARAMS['bull_threshold']:
            self.current_regime = MarketRegime.BULL
        elif sma_ratio < MARKET_REGIME_PARAMS['bear_threshold']:
            self.current_regime = MarketRegime.BEAR
        else:
            self.current_regime = MarketRegime.SIDEWAYS

    def _update_sector_performance(
        self,
        date: datetime,
        stocks: Dict[str, pd.DataFrame],
        sector_map: Dict[str, str]
    ):
        """Update sector performance for sector momentum (改善3)"""
        sector_returns = {}

        for symbol, df in stocks.items():
            if date not in df.index:
                continue

            sector = sector_map.get(symbol, "unknown")
            if sector not in sector_returns:
                sector_returns[sector] = []

            # Calculate 20-day return
            idx = df.index.get_loc(date)
            if idx >= 20:
                ret = (df['Close'].iloc[idx] / df['Close'].iloc[idx - 20] - 1)
                sector_returns[sector].append(ret)

        # Average sector returns
        for sector, returns in sector_returns.items():
            if returns:
                self.sector_performance[sector] = np.mean(returns)

    def _infer_sector_map(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Infer sector map from stock symbols"""
        sector_map = {}
        for symbol in stock_data.keys():
            # Simple sector inference based on symbol prefix
            if symbol.startswith("7"):
                sector_map[symbol] = "auto"
            elif symbol.startswith("6"):
                sector_map[symbol] = "tech"
            elif symbol.startswith("8"):
                sector_map[symbol] = "finance"
            elif symbol.startswith("9"):
                sector_map[symbol] = "telecom"
            elif symbol.startswith("4"):
                sector_map[symbol] = "pharma"
            else:
                sector_map[symbol] = "other"
        return sector_map

    def _process_trading_day(
        self,
        date: datetime,
        stocks: Dict[str, pd.DataFrame],
        weekly_data: Dict[str, pd.DataFrame],
        sector_map: Dict[str, str]
    ):
        """Process a single trading day with all enhancements"""
        # 1. Update trailing stops and check exits
        self._check_exits(date, stocks)

        # 2. Look for new entry opportunities
        if len(self.open_positions) < self.max_positions:
            self._check_entries(date, stocks, weekly_data, sector_map)

    def _check_exits(self, date: datetime, stocks: Dict[str, pd.DataFrame]):
        """Check exit conditions with trailing stop (改善1) and swing low stop (改善6)"""
        for position in self.open_positions[:]:
            symbol = position.symbol
            df = stocks[symbol]

            current_price = self._get_price(df, date, "Close")
            if current_price is None:
                continue

            # Update highest price and trailing stop (改善1)
            if self.enable_trailing_stop:
                if current_price > position.highest_price:
                    position.highest_price = current_price

                    # Check if trailing stop should be activated
                    gain_pct = (current_price - position.entry_price) / position.entry_price
                    if gain_pct >= ENHANCED_RISK_PARAMS["trailing_stop_activation_pct"]:
                        position.trailing_activated = True
                        # Update trailing stop
                        new_trailing_stop = current_price * (
                            1 - ENHANCED_RISK_PARAMS["trailing_stop_pct"]
                        )
                        # Also use ATR-based trailing
                        atr_stop = current_price - (
                            position.atr_at_entry *
                            ENHANCED_RISK_PARAMS["trailing_stop_atr_multiplier"]
                        )
                        position.trailing_stop = max(new_trailing_stop, atr_stop, position.trailing_stop)

            should_exit, reason = self._should_exit(position, date, current_price, df)

            if should_exit:
                self._close_position(position, date, current_price, reason)

    def _should_exit(
        self,
        position: EnhancedTrade,
        date: datetime,
        current_price: float,
        df: pd.DataFrame
    ) -> Tuple[bool, str]:
        """Enhanced exit conditions"""
        # 1. Trailing stop (改善1)
        if self.enable_trailing_stop and position.trailing_activated:
            if current_price <= position.trailing_stop:
                return True, "trailing_stop"

        # 2. Swing low stop (改善6)
        if self.enable_swing_low_stop and position.swing_low_stop > 0:
            if current_price <= position.swing_low_stop:
                return True, "swing_low_stop"

        # 3. Regular stop loss
        if current_price <= position.stop_loss:
            return True, "stop_loss"

        # 4. Profit target
        if current_price >= position.target_price:
            return True, "profit_target"

        # 5. Maximum holding period
        holding_days = (date - position.entry_date).days
        if holding_days >= RISK_PARAMS["max_holding_days"]:
            return True, "max_holding_period"

        # 6. Score deterioration (only check after minimum holding period)
        # Also require score to drop significantly from entry
        if holding_days >= RISK_PARAMS["min_holding_days"]:
            indicators = self._get_indicators_at_date(df, date)
            if indicators:
                score_result = self.scoring_engine.calculate_score(indicators)
                current_score = score_result["total_score"]
                # Exit if score drops below threshold AND is significantly lower than entry
                score_drop = position.score_at_entry - current_score
                if current_score < RISK_PARAMS["reeval_score_threshold"] and score_drop > 15:
                    return True, "score_deterioration"

        return False, ""

    def _check_entries(
        self,
        date: datetime,
        stocks: Dict[str, pd.DataFrame],
        weekly_data: Dict[str, pd.DataFrame],
        sector_map: Dict[str, str]
    ):
        """Enhanced entry check with all filters"""
        candidates = []

        # Get regime-based minimum score
        min_score = self._get_regime_min_score()

        for symbol, df in stocks.items():
            # Skip if already in position
            if any(pos.symbol == symbol for pos in self.open_positions):
                continue

            indicators = self._get_indicators_at_date(df, date)
            if not indicators:
                continue

            # Additional filters (改善8)
            if self.enable_additional_filters:
                if not self._passes_additional_filters(df, date, indicators):
                    continue

            # Calculate base score
            score_result = self.scoring_engine.calculate_score(indicators)
            total_score = score_result["total_score"]
            confidence = score_result["confidence"]

            # Multi-timeframe analysis bonus (改善4)
            if self.enable_multi_timeframe and symbol in weekly_data:
                weekly_bonus = self._calculate_weekly_alignment_bonus(
                    weekly_data[symbol], date
                )
                total_score += weekly_bonus

            # Volume breakout bonus (改善5)
            if self.enable_volume_breakout:
                breakout_bonus = self._calculate_volume_breakout_bonus(df, date)
                total_score += breakout_bonus

            # Sector momentum bonus (改善3)
            sector = sector_map.get(symbol, "unknown")
            sector_bonus = self._calculate_sector_bonus(sector)
            total_score += sector_bonus

            # Check if meets criteria with regime adjustment
            if total_score >= min_score and confidence >= self.min_confidence:
                current_price = indicators.get("Close")
                atr = indicators.get("ATR")

                if current_price and atr:
                    candidates.append({
                        "symbol": symbol,
                        "score": total_score,
                        "price": current_price,
                        "atr": atr,
                        "sector": sector,
                    })

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Enter positions
        available_slots = self.max_positions - len(self.open_positions)
        for candidate in candidates[:available_slots]:
            self._enter_position(
                symbol=candidate["symbol"],
                date=date,
                price=candidate["price"],
                atr=candidate["atr"],
                score=candidate["score"],
                sector=candidate["sector"],
                df=stocks[candidate["symbol"]],
            )

    def _get_regime_min_score(self) -> float:
        """Get minimum score based on market regime (改善7)"""
        if not self.enable_market_regime:
            return self.min_score

        if self.current_regime == MarketRegime.BULL:
            return MARKET_REGIME_PARAMS["bull_min_score"]
        elif self.current_regime == MarketRegime.BEAR:
            return MARKET_REGIME_PARAMS["bear_min_score"]
        else:
            return MARKET_REGIME_PARAMS["sideways_min_score"]

    def _passes_additional_filters(
        self,
        df: pd.DataFrame,
        date: datetime,
        indicators: Dict
    ) -> bool:
        """Check additional filters (改善8)"""
        # Recent performance filter
        if ADDITIONAL_FILTERS["recent_performance_enabled"]:
            idx = df.index.get_loc(date) if date in df.index else -1
            if idx >= ADDITIONAL_FILTERS["recent_performance_days"]:
                recent_return = (
                    df['Close'].iloc[idx] /
                    df['Close'].iloc[idx - ADDITIONAL_FILTERS["recent_performance_days"]] - 1
                )
                if recent_return < ADDITIONAL_FILTERS["recent_drawdown_max"]:
                    return False

        return True

    def _calculate_weekly_alignment_bonus(
        self,
        weekly_df: pd.DataFrame,
        date: datetime
    ) -> float:
        """Calculate bonus for weekly trend alignment (改善4)"""
        if weekly_df.empty:
            return 0

        # Find the most recent weekly data point
        weekly_dates = weekly_df.index[weekly_df.index <= date]
        if len(weekly_dates) == 0:
            return 0

        latest_weekly = weekly_df.loc[weekly_dates[-1]]

        weekly_sma = latest_weekly.get('SMA_Weekly')
        weekly_close = latest_weekly.get('Close')

        if pd.isna(weekly_sma) or pd.isna(weekly_close):
            return 0

        # Check if price is above weekly SMA (bullish alignment)
        if weekly_close > weekly_sma:
            return MULTI_TIMEFRAME_PARAMS["daily_weekly_alignment_bonus"]

        return 0

    def _calculate_volume_breakout_bonus(self, df: pd.DataFrame, date: datetime) -> float:
        """Calculate bonus for volume breakout (改善5)"""
        if date not in df.index:
            return 0

        idx = df.index.get_loc(date)
        if idx < 2:
            return 0

        volume_ratio = df.loc[date].get('Volume_Ratio', 1)
        if volume_ratio is None:
            return 0

        # Check for volume spike with price breakout
        if volume_ratio >= VOLUME_BREAKOUT_PARAMS["volume_spike_threshold"]:
            # Check price breakout
            current_close = df['Close'].iloc[idx]
            prev_close = df['Close'].iloc[idx - 1]
            price_change = (current_close - prev_close) / prev_close

            if price_change >= VOLUME_BREAKOUT_PARAMS["price_breakout_pct"]:
                return VOLUME_BREAKOUT_PARAMS["breakout_score_bonus"]

        return 0

    def _calculate_sector_bonus(self, sector: str) -> float:
        """Calculate bonus for strong sector momentum (改善3)"""
        if not ADDITIONAL_FILTERS["sector_strength_enabled"]:
            return 0

        sector_return = self.sector_performance.get(sector, 0)

        # Bonus for top-performing sectors
        if sector_return > 0.05:  # 5% 20-day return
            return ADDITIONAL_FILTERS["sector_strength_bonus"]

        return 0

    def _enter_position(
        self,
        symbol: str,
        date: datetime,
        price: float,
        atr: float,
        score: float,
        sector: str,
        df: pd.DataFrame,
    ):
        """Enter position with volatility-based sizing (改善2)"""
        # Calculate position size based on volatility (改善2)
        if self.enable_volatility_sizing:
            position_size = self._calculate_volatility_position_size(atr, price)
        else:
            position_size = RISK_PARAMS["position_size_pct"]

        # Adjust for market regime (改善7)
        if self.enable_market_regime:
            regime_multiplier = self._get_regime_position_multiplier()
            position_size *= regime_multiplier

        # Cap position size
        position_size = min(position_size, ENHANCED_RISK_PARAMS["max_position_pct"])
        position_size = max(position_size, ENHANCED_RISK_PARAMS["min_position_pct"])

        # Calculate position value
        available_capital = self.max_capital if self.enable_compound else self.capital
        position_value = available_capital * position_size

        # Apply costs
        effective_price = price * (1 + self.slippage_rate + self.commission_rate)

        shares = int(position_value / effective_price)
        if shares == 0:
            return

        cost = shares * effective_price
        if cost > self.capital:
            return

        # Calculate stop losses
        stop_loss = price - (atr * RISK_PARAMS["stop_loss_atr_multiplier"])

        # Swing low stop (改善6)
        swing_low_stop = 0.0
        if self.enable_swing_low_stop:
            swing_low_stop = self._calculate_swing_low_stop(df, date)

        target_price = price * (1 + RISK_PARAMS["profit_target_max"])

        # Create trade
        trade = EnhancedTrade(
            symbol=symbol,
            entry_date=date,
            entry_price=effective_price,
            shares=shares,
            stop_loss=stop_loss,
            target_price=target_price,
            score_at_entry=score,
            sector=sector,
            highest_price=effective_price,
            swing_low_stop=swing_low_stop,
            atr_at_entry=atr,
            market_regime=self.current_regime.value,
        )

        self.open_positions.append(trade)
        self.capital -= cost

        logger.debug(f"Entered {symbol} at {effective_price:.2f} "
                     f"({shares} shares, score={score:.1f}, regime={self.current_regime.value})")

    def _calculate_volatility_position_size(self, atr: float, price: float) -> float:
        """Calculate position size based on volatility (改善2)"""
        # ATR as percentage of price
        atr_pct = atr / price if price > 0 else 0.02

        # Base risk divided by volatility
        base_risk = ENHANCED_RISK_PARAMS["base_risk_per_trade"]
        stop_distance = atr * RISK_PARAMS["stop_loss_atr_multiplier"]
        stop_pct = stop_distance / price if price > 0 else 0.02

        # Position size = risk / stop distance
        if stop_pct > 0:
            position_size = base_risk / stop_pct
        else:
            position_size = RISK_PARAMS["position_size_pct"]

        # Adjust by volatility factor
        volatility_adjustment = ENHANCED_RISK_PARAMS["volatility_adjustment_factor"]
        avg_atr_pct = 0.02  # Assume 2% average ATR
        position_size *= (avg_atr_pct / max(atr_pct, 0.01)) ** 0.5

        return min(max(position_size, 0.05), 0.40)

    def _get_regime_position_multiplier(self) -> float:
        """Get position size multiplier based on market regime (改善7)"""
        if self.current_regime == MarketRegime.BULL:
            return MARKET_REGIME_PARAMS["bull_position_multiplier"]
        elif self.current_regime == MarketRegime.BEAR:
            return MARKET_REGIME_PARAMS["bear_position_multiplier"]
        elif self.current_regime == MarketRegime.HIGH_VOLATILITY:
            return MARKET_REGIME_PARAMS["high_vol_position_multiplier"]
        return 1.0

    def _calculate_swing_low_stop(self, df: pd.DataFrame, date: datetime) -> float:
        """Calculate swing low based stop loss (改善6)"""
        if date not in df.index:
            return 0.0

        idx = df.index.get_loc(date)
        lookback = ENHANCED_RISK_PARAMS["swing_low_lookback"]

        if idx < lookback:
            return 0.0

        # Find lowest low in lookback period
        swing_low = df['Low'].iloc[idx - lookback:idx].min()

        # Apply buffer
        buffer = ENHANCED_RISK_PARAMS["swing_low_buffer_pct"]
        return swing_low * (1 - buffer)

    def _close_position(self, position: EnhancedTrade, date: datetime, price: float, reason: str):
        """Close an open position"""
        effective_price = price * (1 - self.slippage_rate - self.commission_rate)

        position.exit_date = date
        position.exit_price = effective_price
        position.exit_reason = reason

        proceeds = position.shares * effective_price
        self.capital += proceeds

        self.open_positions.remove(position)
        self.closed_trades.append(position)

        logger.debug(f"Closed {position.symbol} at {effective_price:.2f} "
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
        except Exception:
            return None

    def _get_price(self, df: pd.DataFrame, date: datetime, column: str) -> Optional[float]:
        """Get price at a specific date"""
        try:
            if date not in df.index:
                return None
            return float(df.loc[date, column])
        except Exception:
            return None
