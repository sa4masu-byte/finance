#!/usr/bin/env python3
"""最良設定の微調整"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from config.settings import DATA_DIR
from itertools import product


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
    def return_pct(self):
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100 if self.exit_price else 0
    @property
    def profit_loss(self):
        return (self.exit_price - self.entry_price) * self.shares if self.exit_price else 0


@dataclass
class Result:
    trades: List[Trade] = field(default_factory=list)
    initial: float = 1_000_000
    final: float = 1_000_000
    equity: List[float] = field(default_factory=list)
    @property
    def n(self): return len([t for t in self.trades if t.exit_date])
    @property
    def wr(self):
        c = [t for t in self.trades if t.exit_date]
        return len([t for t in c if t.return_pct > 0]) / len(c) if c else 0
    @property
    def pf(self):
        w = sum(t.profit_loss for t in self.trades if t.exit_date and t.return_pct > 0)
        l = abs(sum(t.profit_loss for t in self.trades if t.exit_date and t.return_pct < 0))
        return min(w/l, 10) if l else (10 if w else 0)
    @property
    def ret(self): return ((self.final - self.initial) / self.initial) * 100
    @property
    def dd(self):
        if not self.equity: return 0
        eq = pd.Series(self.equity)
        return abs((eq - eq.cummax()) / eq.cummax()).max() * 100
    @property
    def sharpe(self):
        if len(self.equity) < 2: return 0
        r = pd.Series(self.equity).pct_change().dropna()
        return (r.mean() / r.std()) * np.sqrt(252) if r.std() else 0


class S:
    def __init__(self, rsi_min=35, rsi_max=65, sl=2.0, tp=4.0, trail=0.03, days=10, pos=3, pct=0.30):
        self.rsi_min, self.rsi_max = rsi_min, rsi_max
        self.sl, self.tp, self.trail, self.days = sl, tp, trail, days
        self.pos, self.pct = pos, pct
        self.capital = 1_000_000
        self.positions, self.closed, self.equity = [], [], []

    def calc(self, df):
        df = df.copy()
        df['S5'] = df['Close'].rolling(5).mean()
        df['S25'] = df['Close'].rolling(25).mean()
        df['S25u'] = df['S25'] > df['S25'].shift(1)
        d = df['Close'].diff()
        g, l = d.where(d>0,0).rolling(14).mean(), (-d.where(d<0,0)).rolling(14).mean()
        df['RSI'] = 100 - 100/(1 + g/l.replace(0,0.0001))
        tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        df['VR'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['R5'] = df['Close'].pct_change(5)
        return df

    def ok(self, r):
        for c in ['S25','S25u','RSI','ATR','VR','R5','Close','S5']:
            if pd.isna(r.get(c)): return False
        if not r['S25u'] or r['Close'] <= r['S25'] or r['S5'] <= r['S25']: return False
        if not (self.rsi_min <= r['RSI'] <= self.rsi_max): return False
        if r['VR'] < 1.0 or r['R5'] < -0.05: return False
        return True

    def run(self, stocks, start, end):
        self.capital, self.positions, self.closed, self.equity = 1_000_000, [], [], [1_000_000]
        data = {s: self.calc(df) for s, df in stocks.items()}
        dates = [d for d in sorted(set().union(*[set(df.index) for df in data.values()])) 
                 if pd.to_datetime(start) <= d <= pd.to_datetime(end)]
        
        for dt in dates:
            for p in self.positions[:]:
                if dt not in data[p.symbol].index: continue
                px = data[p.symbol].loc[dt,'Close']
                if px > p.highest_price:
                    p.highest_price = px
                    if (px - p.entry_price)/p.entry_price >= self.trail:
                        p.trailing_stop = max(p.trailing_stop, px*(1-0.02))
                rsn = None
                if p.trailing_stop and px <= p.trailing_stop: rsn = 'trail'
                elif px <= p.stop_loss: rsn = 'stop'
                elif px >= p.target_price: rsn = 'target'
                elif (dt - p.entry_date).days >= self.days: rsn = 'time'
                if rsn:
                    p.exit_date, p.exit_price, p.exit_reason = dt, px, rsn
                    self.capital += p.shares * px * 0.999
                    self.positions.remove(p); self.closed.append(p)
            
            if len(self.positions) < self.pos:
                cands = [(s, data[s].loc[dt,'Close'], data[s].loc[dt,'ATR'], data[s].loc[dt,'VR'])
                         for s, df in data.items() if dt in df.index and self.ok(df.loc[dt])
                         and not any(p.symbol == s for p in self.positions)]
                for sym, px, atr, _ in sorted(cands, key=lambda x:-x[3])[:self.pos-len(self.positions)]:
                    sh = int(self.capital * self.pct / px)
                    if sh and sh*px <= self.capital:
                        self.positions.append(Trade(sym, dt, px, sh, px-atr*self.sl, px+atr*self.tp, highest_price=px))
                        self.capital -= sh * px * 1.001
            
            self.equity.append(self.capital + sum(p.shares * data[p.symbol].loc[dt,'Close'] 
                              for p in self.positions if dt in data[p.symbol].index))
        
        for p in self.positions:
            p.exit_date, p.exit_price, p.exit_reason = dates[-1], data[p.symbol].iloc[-1]['Close'], 'end'
            self.capital += p.shares * p.exit_price * 0.999
            self.closed.append(p)
        return Result(self.closed, 1_000_000, self.capital, self.equity)


def main():
    print("🎯 微調整最適化\n")
    data = {f.stem.replace('_','.'): pd.read_csv(f, index_col=0, parse_dates=True) 
            for f in (DATA_DIR/"stock_prices").glob("*.csv")}
    data = {k:v for k,v in data.items() if len(v)>60}
    dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    start, end = dates[60].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')
    print(f"銘柄:{len(data)} 期間:{start}~{end}\n")
    
    # Grid search around best
    grid = list(product(
        [32, 35, 38],      # RSI min
        [62, 65, 68],      # RSI max  
        [1.8, 2.0, 2.2],   # SL
        [3.5, 4.0, 4.5],   # TP
        [0.025, 0.03, 0.035], # Trail
        [8, 10, 12],       # Days
    ))
    
    print(f"テストパターン数: {len(grid)}")
    print("-"*70)
    
    results = []
    for i, (rsi_min, rsi_max, sl, tp, trail, days) in enumerate(grid):
        r = S(rsi_min, rsi_max, sl, tp, trail, days).run(data, start, end)
        results.append((rsi_min, rsi_max, sl, tp, trail, days, r))
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(grid)} 完了...")
    
    # Top 10
    top = sorted(results, key=lambda x: -x[6].ret)[:10]
    print(f"\n{'#':>2} {'RSI':>8} {'SL':>4} {'TP':>4} {'Tr':>5} {'D':>3} | {'N':>3} {'WR':>5} {'PF':>5} {'DD':>5} {'Sh':>5} {'Ret':>8}")
    print("-"*75)
    for i, (rmin, rmax, sl, tp, tr, d, r) in enumerate(top):
        print(f"{i+1:>2} {rmin:>3}-{rmax:<3} {sl:>4.1f} {tp:>4.1f} {tr:>4.1%} {d:>3} | {r.n:>3} {r.wr*100:>4.0f}% {r.pf:>5.2f} {r.dd:>4.1f}% {r.sharpe:>5.2f} {r.ret:>+7.1f}%")
    
    best = top[0]
    print(f"\n🏆 最良パラメータ:")
    print(f"   RSI: {best[0]}-{best[1]}")
    print(f"   SL: {best[2]} ATR, TP: {best[3]} ATR")
    print(f"   Trail: {best[4]:.1%}, Days: {best[5]}")
    print(f"   Return: {best[6].ret:+.1f}%, Sharpe: {best[6].sharpe:.2f}")

if __name__ == "__main__":
    main()
