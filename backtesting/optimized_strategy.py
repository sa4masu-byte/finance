"""
最適化済み本番戦略

最適パラメータ (グリッドサーチ結果):
- RSI: 32-68
- SL: 2.2 ATR
- TP: 3.5 ATR
- Trail: 3.5%
- Max Days: 10
- Return: +32.0%
- Sharpe: 2.82
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: int
    stop_loss: float
    target_price: float
    trailing_stop: float = 0.0
    highest_price: float = 0.0
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""

    @property
    def return_pct(self) -> float:
        if not self.exit_price:
            return 0.0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100

    @property
    def profit_loss(self) -> float:
        if not self.exit_price:
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
        return len([t for t in self.trades if t.exit_date])

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if t.exit_date]
        if not closed:
            return 0.0
        return len([t for t in closed if t.return_pct > 0]) / len(closed)

    @property
    def avg_win(self) -> float:
        winners = [t for t in self.trades if t.exit_date and t.return_pct > 0]
        return np.mean([t.return_pct for t in winners]) if winners else 0.0

    @property
    def avg_loss(self) -> float:
        losers = [t for t in self.trades if t.exit_date and t.return_pct < 0]
        return np.mean([t.return_pct for t in losers]) if losers else 0.0

    @property
    def profit_factor(self) -> float:
        wins = sum(t.profit_loss for t in self.trades if t.exit_date and t.return_pct > 0)
        losses = abs(sum(t.profit_loss for t in self.trades if t.exit_date and t.return_pct < 0))
        if losses == 0:
            return 10.0 if wins > 0 else 0.0
        return min(wins / losses, 10.0)

    @property
    def total_return_pct(self) -> float:
        return ((self.final_capital - self.initial_capital) / self.initial_capital) * 100

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        eq = pd.Series(self.equity_curve)
        dd = (eq - eq.cummax()) / eq.cummax()
        return abs(dd.min()) * 100

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
        closed = [t for t in self.trades if t.exit_date]
        if not closed:
            return 0.0
        return np.mean([(t.exit_date - t.entry_date).days for t in closed])


class OptimizedStrategy:
    """
    最適化済み本番戦略

    エントリー条件:
    1. 25日SMA上向き
    2. 価格 > 25日SMA
    3. 5日SMA > 25日SMA（短期も上向き）
    4. RSI 32-68
    5. 出来高 >= 20日平均
    6. 直近5日で-5%以上下落していない

    エグジット:
    - ストップロス: ATR × 2.2
    - 利確: ATR × 3.5
    - トレーリング: 3.5%利益で発動、2%トレール
    - 最大保有: 10日
    """

    # 最適化済みパラメータ
    RSI_MIN = 32
    RSI_MAX = 68
    STOP_LOSS_ATR = 2.2
    TAKE_PROFIT_ATR = 3.5
    TRAILING_ACTIVATION = 0.035  # 3.5%
    TRAILING_STOP_PCT = 0.02     # 2%
    MAX_HOLDING_DAYS = 10

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        max_positions: int = 3,
        position_size_pct: float = 0.30,
        commission_rate: float = 0.001,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.commission_rate = commission_rate

        self.positions: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[float] = []

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標を計算"""
        df = df.copy()

        # 移動平均
        df['SMA_5'] = df['Close'].rolling(5).mean()
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

        # 出来高
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        # 直近リターン
        df['Return_5d'] = df['Close'].pct_change(5)

        return df

    def check_entry_signal(self, row: pd.Series) -> bool:
        """エントリーシグナル判定"""
        required = ['SMA_25', 'SMA_Rising', 'RSI', 'ATR', 'Volume_Ratio', 'Return_5d', 'Close', 'SMA_5']
        if any(pd.isna(row.get(col)) for col in required):
            return False

        # 1. トレンド確認
        if not row['SMA_Rising']:
            return False

        # 2. 価格 > SMA25
        if row['Close'] <= row['SMA_25']:
            return False

        # 3. SMA5 > SMA25
        if row['SMA_5'] <= row['SMA_25']:
            return False

        # 4. RSI範囲
        if not (self.RSI_MIN <= row['RSI'] <= self.RSI_MAX):
            return False

        # 5. 出来高確認
        if row['Volume_Ratio'] < 1.0:
            return False

        # 6. 直近下落チェック
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
        self.positions = []
        self.closed_trades = []
        self.equity_curve = [self.initial_capital]

        # 指標計算
        stocks = {s: self.calculate_indicators(df) for s, df in stock_data.items()}

        # 日付範囲
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        all_dates = set()
        for df in stocks.values():
            all_dates.update(df.index)
        dates = sorted([d for d in all_dates if start_dt <= d <= end_dt])

        # 日次ループ
        for date in dates:
            self._check_exits(date, stocks)

            if len(self.positions) < self.max_positions:
                self._check_entries(date, stocks)

            # エクイティ
            equity = self.capital + sum(
                p.shares * self._get_price(stocks[p.symbol], date)
                for p in self.positions
                if self._get_price(stocks[p.symbol], date)
            )
            self.equity_curve.append(equity)

        # 残りクローズ
        for pos in self.positions[:]:
            price = self._get_price(stocks[pos.symbol], dates[-1])
            if price:
                self._close_position(pos, dates[-1], price, "end_of_backtest")

        return BacktestResult(
            trades=self.closed_trades,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            equity_curve=self.equity_curve,
        )

    def _check_exits(self, date: datetime, stocks: Dict[str, pd.DataFrame]):
        """エグジット判定"""
        for pos in self.positions[:]:
            df = stocks[pos.symbol]
            price = self._get_price(df, date)
            if not price:
                continue

            # トレーリングストップ更新
            if price > pos.highest_price:
                pos.highest_price = price
                gain_pct = (price - pos.entry_price) / pos.entry_price
                if gain_pct >= self.TRAILING_ACTIVATION:
                    new_stop = price * (1 - self.TRAILING_STOP_PCT)
                    pos.trailing_stop = max(pos.trailing_stop, new_stop)

            # エグジット判定
            if pos.trailing_stop > 0 and price <= pos.trailing_stop:
                self._close_position(pos, date, price, "trailing_stop")
            elif price <= pos.stop_loss:
                self._close_position(pos, date, price, "stop_loss")
            elif price >= pos.target_price:
                self._close_position(pos, date, price, "take_profit")
            elif (date - pos.entry_date).days >= self.MAX_HOLDING_DAYS:
                self._close_position(pos, date, price, "max_holding")

    def _check_entries(self, date: datetime, stocks: Dict[str, pd.DataFrame]):
        """エントリー判定"""
        candidates = []

        for symbol, df in stocks.items():
            if any(p.symbol == symbol for p in self.positions):
                continue
            if date not in df.index:
                continue

            row = df.loc[date]
            if self.check_entry_signal(row):
                candidates.append({
                    'symbol': symbol,
                    'price': row['Close'],
                    'atr': row['ATR'],
                    'volume_ratio': row['Volume_Ratio'],
                })

        # 出来高が高い順
        candidates.sort(key=lambda x: -x['volume_ratio'])

        # エントリー
        slots = self.max_positions - len(self.positions)
        for c in candidates[:slots]:
            self._enter_position(c['symbol'], date, c['price'], c['atr'])

    def _enter_position(self, symbol: str, date: datetime, price: float, atr: float):
        """ポジションエントリー"""
        position_value = self.capital * self.position_size_pct
        shares = int(position_value / price)
        if shares == 0:
            return

        cost = shares * price * (1 + self.commission_rate)
        if cost > self.capital:
            return

        trade = Trade(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
            shares=shares,
            stop_loss=price - (atr * self.STOP_LOSS_ATR),
            target_price=price + (atr * self.TAKE_PROFIT_ATR),
            highest_price=price,
        )

        self.positions.append(trade)
        self.capital -= cost

    def _close_position(self, pos: Trade, date: datetime, price: float, reason: str):
        """ポジションクローズ"""
        pos.exit_date = date
        pos.exit_price = price
        pos.exit_reason = reason

        proceeds = pos.shares * price * (1 - self.commission_rate)
        self.capital += proceeds

        self.positions.remove(pos)
        self.closed_trades.append(pos)

    def _get_price(self, df: pd.DataFrame, date: datetime) -> Optional[float]:
        """価格取得"""
        try:
            if date in df.index:
                return float(df.loc[date, 'Close'])
        except:
            pass
        return None

    def get_today_signals(self, stock_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """本日のシグナルを取得（本番用）"""
        signals = []

        for symbol, df in stock_data.items():
            df = self.calculate_indicators(df)
            if df.empty:
                continue

            latest = df.iloc[-1]
            if self.check_entry_signal(latest):
                atr = latest['ATR']
                price = latest['Close']
                signals.append({
                    'symbol': symbol,
                    'price': price,
                    'stop_loss': price - (atr * self.STOP_LOSS_ATR),
                    'target': price + (atr * self.TAKE_PROFIT_ATR),
                    'rsi': latest['RSI'],
                    'volume_ratio': latest['Volume_Ratio'],
                })

        # 出来高順でソート
        signals.sort(key=lambda x: -x['volume_ratio'])
        return signals[:self.max_positions]
