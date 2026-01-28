#!/usr/bin/env python3
"""
戦略最適化スクリプト
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from config.settings import DATA_DIR


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
        if self.exit_price is None:
            return 0.0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100

    @property
    def profit_loss(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares


@dataclass
class Result:
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


class OptimizedStrategy:
    """
    最適化された戦略 v2

    改良点:
    1. トレーリングストップで利益を伸ばす
    2. 出来高ブレイクアウトでエントリー精度向上
    3. MACD確認でトレンド精度向上
    4. ポジションサイズを動的に調整
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        max_positions: int = 3,
        # エントリー条件
        rsi_min: float = 35,
        rsi_max: float = 60,
        volume_threshold: float = 1.2,
        # リスク管理
        stop_loss_atr: float = 1.5,
        take_profit_atr: float = 5.0,
        trailing_activation: float = 0.03,  # 3%で発動
        trailing_stop_pct: float = 0.02,    # 2%トレール
        max_holding_days: int = 15,
        # ポジションサイズ
        base_position_pct: float = 0.30,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions

        # パラメータ
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.volume_threshold = volume_threshold
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.trailing_activation = trailing_activation
        self.trailing_stop_pct = trailing_stop_pct
        self.max_holding_days = max_holding_days
        self.base_position_pct = base_position_pct

        self.positions: List[Trade] = []
        self.closed: List[Trade] = []
        self.equity: List[float] = []

    def calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # SMA
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_25'] = df['Close'].rolling(25).mean()
        df['SMA_25_Prev'] = df['SMA_25'].shift(1)
        df['SMA_Rising'] = df['SMA_25'] > df['SMA_25_Prev']

        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.0001)
        df['RSI'] = 100 - (100 / (1 + rs))

        # ATR
        hl = df['High'] - df['Low']
        hc = abs(df['High'] - df['Close'].shift())
        lc = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        # Volume
        df['Vol_MA'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA']

        # Recent performance
        df['Ret_5d'] = df['Close'].pct_change(5)

        return df

    def check_entry(self, row: pd.Series) -> bool:
        """エントリー条件（厳選）"""
        required = ['SMA_25', 'SMA_Rising', 'RSI', 'ATR', 'Vol_Ratio', 'Ret_5d', 'Close', 'MACD_Hist', 'SMA_5']
        if any(pd.isna(row.get(col)) for col in required):
            return False

        # 1. トレンド確認: SMA25上向き & 価格>SMA25
        if not row['SMA_Rising'] or row['Close'] <= row['SMA_25']:
            return False

        # 2. 短期も上向き: SMA5 > SMA25
        if row['SMA_5'] <= row['SMA_25']:
            return False

        # 3. MACD確認: ヒストグラムがプラス
        if row['MACD_Hist'] <= 0:
            return False

        # 4. RSIが適正範囲
        if not (self.rsi_min <= row['RSI'] <= self.rsi_max):
            return False

        # 5. 出来高確認
        if row['Vol_Ratio'] < self.volume_threshold:
            return False

        # 6. 直近で大きな下落なし
        if row['Ret_5d'] < -0.05:
            return False

        return True

    def run(self, stocks: Dict[str, pd.DataFrame], start: str, end: str) -> Result:
        self.capital = self.initial_capital
        self.positions = []
        self.closed = []
        self.equity = [self.initial_capital]

        # 指標計算
        data = {s: self.calc_indicators(df) for s, df in stocks.items()}

        # 日付取得
        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        dates = sorted(set().union(*[set(df.index) for df in data.values()]))
        dates = [d for d in dates if start_dt <= d <= end_dt]

        for date in dates:
            self._check_exits(date, data)
            if len(self.positions) < self.max_positions:
                self._check_entries(date, data)

            # エクイティ計算
            eq = self.capital + sum(
                p.shares * self._price(data[p.symbol], date)
                for p in self.positions
                if self._price(data[p.symbol], date)
            )
            self.equity.append(eq)

        # 残りクローズ
        for p in self.positions[:]:
            price = self._price(data[p.symbol], dates[-1])
            if price:
                self._close(p, dates[-1], price, "end")

        return Result(
            trades=self.closed,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            equity_curve=self.equity,
        )

    def _check_exits(self, date, data):
        for p in self.positions[:]:
            price = self._price(data[p.symbol], date)
            if not price:
                continue

            # トレーリングストップ更新
            if price > p.highest_price:
                p.highest_price = price
                gain = (price - p.entry_price) / p.entry_price
                if gain >= self.trailing_activation:
                    p.trailing_stop = max(p.trailing_stop, price * (1 - self.trailing_stop_pct))

            # エグジット判定
            if p.trailing_stop > 0 and price <= p.trailing_stop:
                self._close(p, date, price, "trailing")
            elif price <= p.stop_loss:
                self._close(p, date, price, "stop")
            elif price >= p.target_price:
                self._close(p, date, price, "target")
            elif (date - p.entry_date).days >= self.max_holding_days:
                self._close(p, date, price, "time")

    def _check_entries(self, date, data):
        candidates = []
        for symbol, df in data.items():
            if any(p.symbol == symbol for p in self.positions):
                continue
            if date not in df.index:
                continue

            row = df.loc[date]
            if self.check_entry(row):
                candidates.append({
                    'symbol': symbol,
                    'price': row['Close'],
                    'atr': row['ATR'],
                    'rsi': row['RSI'],
                    'vol_ratio': row['Vol_Ratio'],
                })

        # 出来高が高い順（モメンタム重視）
        candidates.sort(key=lambda x: -x['vol_ratio'])

        slots = self.max_positions - len(self.positions)
        for c in candidates[:slots]:
            self._enter(c['symbol'], date, c['price'], c['atr'])

    def _enter(self, symbol, date, price, atr):
        size = self.capital * self.base_position_pct
        shares = int(size / price)
        if shares == 0 or shares * price > self.capital:
            return

        trade = Trade(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
            shares=shares,
            stop_loss=price - atr * self.stop_loss_atr,
            target_price=price + atr * self.take_profit_atr,
            highest_price=price,
        )
        self.positions.append(trade)
        self.capital -= shares * price * 1.001  # 手数料

    def _close(self, p, date, price, reason):
        p.exit_date = date
        p.exit_price = price
        p.exit_reason = reason
        self.capital += p.shares * price * 0.999  # 手数料
        self.positions.remove(p)
        self.closed.append(p)

    def _price(self, df, date):
        try:
            return float(df.loc[date, 'Close']) if date in df.index else None
        except:
            return None


def load_data():
    stock_data = {}
    for f in (DATA_DIR / "stock_prices").glob("*.csv"):
        symbol = f.stem.replace('_', '.')
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if len(df) > 60:
            stock_data[symbol] = df
    return stock_data


def main():
    print("\n" + "=" * 70)
    print("🔧 戦略パラメータ最適化")
    print("=" * 70)

    data = load_data()
    print(f"銘柄数: {len(data)}")

    dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    start, end = dates[60].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')
    print(f"期間: {start} ~ {end}")

    # パラメータグリッド
    params = [
        # (RSI min, RSI max, SL ATR, TP ATR, Trailing Act, Max Days)
        (35, 60, 1.5, 5.0, 0.03, 15),   # ベース
        (30, 55, 1.5, 6.0, 0.025, 20),  # ワイドRSI、長め保有
        (40, 65, 1.2, 4.0, 0.02, 10),   # タイトRSI、短め保有
        (35, 55, 1.0, 6.0, 0.03, 15),   # タイトSL、高TP
        (30, 60, 2.0, 5.0, 0.04, 12),   # ワイドSL
    ]

    print(f"\n{'#':>2} {'RSI':>10} {'SL/TP':>10} {'Trail':>6} {'Days':>5} | {'取引':>5} {'勝率':>7} {'PF':>6} {'DD':>6} {'Return':>9}")
    print("-" * 85)

    results = []
    for i, (rsi_min, rsi_max, sl, tp, trail, days) in enumerate(params):
        s = OptimizedStrategy(
            rsi_min=rsi_min, rsi_max=rsi_max,
            stop_loss_atr=sl, take_profit_atr=tp,
            trailing_activation=trail,
            max_holding_days=days,
        )
        r = s.run(data, start, end)
        results.append((i, r))

        print(f"{i+1:>2} {rsi_min:>4}-{rsi_max:<4} {sl:>4.1f}/{tp:<4.1f} {trail:>5.1%} {days:>5} | {r.num_trades:>5} {r.win_rate*100:>6.1f}% {r.profit_factor:>6.2f} {r.max_drawdown:>5.1f}% {r.total_return_pct:>+8.1f}%")

    # 最良
    best_idx, best = max(results, key=lambda x: x[1].total_return_pct)
    print("\n" + "=" * 70)
    print(f"🏆 最良パラメータ: #{best_idx+1}")
    print(f"   Return: {best.total_return_pct:+.1f}%")
    print(f"   勝率: {best.win_rate*100:.1f}%")
    print(f"   PF: {best.profit_factor:.2f}")
    print(f"   Sharpe: {best.sharpe_ratio:.2f}")

    # Exit reason分析
    if best.trades:
        print("\n   Exit Reason:")
        reasons = {}
        for t in best.trades:
            r = t.exit_reason
            if r not in reasons:
                reasons[r] = {'n': 0, 'ret': 0}
            reasons[r]['n'] += 1
            reasons[r]['ret'] += t.return_pct
        for r, s in sorted(reasons.items(), key=lambda x: -x[1]['n']):
            avg = s['ret'] / s['n'] if s['n'] else 0
            print(f"     {r:<10}: {s['n']:>3}回, 平均{avg:+.1f}%")


if __name__ == "__main__":
    main()
