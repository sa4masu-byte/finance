"""
シンプルで勝てる戦略

原則:
1. トレンドフォロー（順張り）
2. 厳選エントリー（取引回数を絞る）
3. 損小利大（リスクリワード2:1以上）
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: int
    stop_loss: float
    target_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""

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


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    initial_capital: float = 1_000_000
    final_capital: float = 1_000_000
    equity_curve: List[float] = field(default_factory=list)

    @property
    def num_trades(self) -> int:
        return len([t for t in self.trades if not t.is_open])

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if not t.is_open]
        if not closed:
            return 0.0
        winners = [t for t in closed if t.return_pct > 0]
        return len(winners) / len(closed)

    @property
    def avg_win(self) -> float:
        winners = [t for t in self.trades if not t.is_open and t.return_pct > 0]
        if not winners:
            return 0.0
        return np.mean([t.return_pct for t in winners])

    @property
    def avg_loss(self) -> float:
        losers = [t for t in self.trades if not t.is_open and t.return_pct < 0]
        if not losers:
            return 0.0
        return np.mean([t.return_pct for t in losers])

    @property
    def profit_factor(self) -> float:
        winners = [t for t in self.trades if not t.is_open and t.return_pct > 0]
        losers = [t for t in self.trades if not t.is_open and t.return_pct < 0]
        total_win = sum(t.profit_loss for t in winners)
        total_loss = abs(sum(t.profit_loss for t in losers))
        if total_loss == 0:
            return 10.0 if total_win > 0 else 0.0
        return min(total_win / total_loss, 10.0)

    @property
    def total_return_pct(self) -> float:
        return ((self.final_capital - self.initial_capital) / self.initial_capital) * 100

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        equity = pd.Series(self.equity_curve)
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        return abs(drawdown.min()) * 100

    @property
    def sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    @property
    def avg_holding_days(self) -> float:
        closed = [t for t in self.trades if not t.is_open]
        if not closed:
            return 0.0
        days = [(t.exit_date - t.entry_date).days for t in closed]
        return np.mean(days)


class SimpleWinningStrategy:
    """
    シンプルで勝てる戦略

    エントリー条件（全て満たす）:
    1. 25日SMAが上向き（トレンド確認）
    2. 価格が25日SMAの上にある
    3. RSI 40-65（過熱していない）
    4. 出来高が平均以上
    5. 直近5日で-5%以上下落していない

    エグジット条件:
    - ストップロス: エントリー価格 - ATR×2
    - 利確: エントリー価格 + ATR×4（リスクリワード2:1）
    - 最大保有: 10日
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        max_positions: int = 3,
        position_size_pct: float = 0.25,
        commission_rate: float = 0.001,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.commission_rate = commission_rate

        self.open_positions: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[float] = []

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """必要な指標を計算"""
        df = df.copy()

        # SMA
        df['SMA_25'] = df['Close'].rolling(25).mean()
        df['SMA_25_Prev'] = df['SMA_25'].shift(1)
        df['SMA_Rising'] = df['SMA_25'] > df['SMA_25_Prev']

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.0001)
        df['RSI'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        # Volume ratio
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        # Recent performance
        df['Return_5d'] = df['Close'].pct_change(5)

        return df

    def check_entry_signal(self, row: pd.Series) -> bool:
        """エントリーシグナルをチェック"""
        # 必要なデータがあるか
        required = ['SMA_25', 'SMA_Rising', 'RSI', 'ATR', 'Volume_Ratio', 'Return_5d', 'Close']
        if any(pd.isna(row.get(col)) for col in required):
            return False

        # 条件1: 25日SMAが上向き
        if not row['SMA_Rising']:
            return False

        # 条件2: 価格がSMAの上
        if row['Close'] <= row['SMA_25']:
            return False

        # 条件3: RSI 40-65
        if not (40 <= row['RSI'] <= 65):
            return False

        # 条件4: 出来高が平均以上
        if row['Volume_Ratio'] < 1.0:
            return False

        # 条件5: 直近5日で-5%以上下落していない
        if row['Return_5d'] < -0.05:
            return False

        return True

    def run_backtest(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """バックテスト実行"""
        self.capital = self.initial_capital
        self.open_positions = []
        self.closed_trades = []
        self.equity_curve = [self.initial_capital]

        # 指標計算
        stocks_with_indicators = {}
        for symbol, df in stock_data.items():
            stocks_with_indicators[symbol] = self.calculate_indicators(df)

        # 日付範囲
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        all_dates = set()
        for df in stocks_with_indicators.values():
            all_dates.update(df.index)
        all_dates = sorted([d for d in all_dates if start_dt <= d <= end_dt])

        # 日次ループ
        for date in all_dates:
            # 1. エグジット判定
            self._check_exits(date, stocks_with_indicators)

            # 2. エントリー判定
            if len(self.open_positions) < self.max_positions:
                self._check_entries(date, stocks_with_indicators)

            # 3. エクイティ記録
            total_equity = self.capital + sum(
                pos.shares * self._get_price(stocks_with_indicators[pos.symbol], date)
                for pos in self.open_positions
                if self._get_price(stocks_with_indicators[pos.symbol], date)
            )
            self.equity_curve.append(total_equity)

        # 残ポジションをクローズ
        for pos in self.open_positions[:]:
            last_date = all_dates[-1]
            price = self._get_price(stocks_with_indicators[pos.symbol], last_date)
            if price:
                self._close_position(pos, last_date, price, "end_of_backtest")

        return BacktestResult(
            trades=self.closed_trades,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            equity_curve=self.equity_curve,
        )

    def _check_exits(self, date: datetime, stocks: Dict[str, pd.DataFrame]):
        """エグジット判定"""
        for pos in self.open_positions[:]:
            df = stocks[pos.symbol]
            price = self._get_price(df, date)
            if price is None:
                continue

            # ストップロス
            if price <= pos.stop_loss:
                self._close_position(pos, date, price, "stop_loss")
                continue

            # 利確
            if price >= pos.target_price:
                self._close_position(pos, date, price, "take_profit")
                continue

            # 最大保有期間
            holding_days = (date - pos.entry_date).days
            if holding_days >= 10:
                self._close_position(pos, date, price, "max_holding")

    def _check_entries(self, date: datetime, stocks: Dict[str, pd.DataFrame]):
        """エントリー判定"""
        candidates = []

        for symbol, df in stocks.items():
            # 既にポジションあり
            if any(p.symbol == symbol for p in self.open_positions):
                continue

            # データ取得
            if date not in df.index:
                continue
            row = df.loc[date]

            # シグナル判定
            if self.check_entry_signal(row):
                candidates.append({
                    'symbol': symbol,
                    'price': row['Close'],
                    'atr': row['ATR'],
                    'rsi': row['RSI'],
                })

        # RSIが低い順（上昇余地が大きい）でソート
        candidates.sort(key=lambda x: x['rsi'])

        # エントリー
        available_slots = self.max_positions - len(self.open_positions)
        for c in candidates[:available_slots]:
            self._enter_position(c['symbol'], date, c['price'], c['atr'])

    def _enter_position(self, symbol: str, date: datetime, price: float, atr: float):
        """ポジションエントリー"""
        # ポジションサイズ計算
        position_value = self.capital * self.position_size_pct
        shares = int(position_value / price)
        if shares == 0:
            return

        cost = shares * price * (1 + self.commission_rate)
        if cost > self.capital:
            return

        # ストップロス・利確設定（リスクリワード 2:1）
        stop_loss = price - (atr * 2)
        target_price = price + (atr * 4)

        trade = Trade(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
            shares=shares,
            stop_loss=stop_loss,
            target_price=target_price,
        )

        self.open_positions.append(trade)
        self.capital -= cost

    def _close_position(self, pos: Trade, date: datetime, price: float, reason: str):
        """ポジションクローズ"""
        pos.exit_date = date
        pos.exit_price = price
        pos.exit_reason = reason

        proceeds = pos.shares * price * (1 - self.commission_rate)
        self.capital += proceeds

        self.open_positions.remove(pos)
        self.closed_trades.append(pos)

    def _get_price(self, df: pd.DataFrame, date: datetime) -> Optional[float]:
        """価格取得"""
        try:
            if date in df.index:
                return float(df.loc[date, 'Close'])
        except:
            pass
        return None


def run_comparison(stock_data: Dict[str, pd.DataFrame], start_date: str, end_date: str):
    """戦略比較"""
    import sys
    sys.path.insert(0, '.')
    from backtesting.backtest_engine import BacktestEngine
    from backtesting.enhanced_backtest_engine import EnhancedBacktestEngine

    print("=" * 70)
    print("🔬 戦略比較")
    print("=" * 70)

    # 1. 基本版
    basic = BacktestEngine(initial_capital=1_000_000, max_positions=3)
    basic_result = basic.run_backtest(stock_data, start_date, end_date)

    # 2. 改善版（従来）
    enhanced = EnhancedBacktestEngine(
        initial_capital=1_000_000, max_positions=3,
        min_score=65, min_confidence=0.65
    )
    enhanced_result = enhanced.run_backtest(stock_data, start_date, end_date, {})

    # 3. 新シンプル戦略
    simple = SimpleWinningStrategy(initial_capital=1_000_000, max_positions=3)
    simple_result = simple.run_backtest(stock_data, start_date, end_date)

    # 結果表示
    results = [
        ("基本版", basic_result),
        ("改善版", enhanced_result),
        ("新戦略", simple_result),
    ]

    print(f"\n{'戦略':<10} {'取引':>6} {'勝率':>8} {'PF':>7} {'DD':>7} {'Return':>10}")
    print("-" * 55)
    for name, r in results:
        print(f"{name:<10} {r.num_trades:>6} {r.win_rate*100:>7.1f}% {r.profit_factor:>7.2f} {r.max_drawdown:>6.1f}% {r.total_return_pct:>+9.1f}%")

    return simple_result
