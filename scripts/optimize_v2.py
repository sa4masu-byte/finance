#!/usr/bin/env python3
"""広範囲パラメータ最適化"""
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
    def return_pct(self):
        if not self.exit_price: return 0.0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100

    @property
    def profit_loss(self):
        if not self.exit_price: return 0.0
        return (self.exit_price - self.entry_price) * self.shares


@dataclass
class Result:
    trades: List[Trade] = field(default_factory=list)
    initial_capital: float = 1_000_000
    final_capital: float = 1_000_000
    equity_curve: List[float] = field(default_factory=list)

    @property
    def num_trades(self): return len([t for t in self.trades if t.exit_date])
    @property
    def win_rate(self):
        c = [t for t in self.trades if t.exit_date]
        return len([t for t in c if t.return_pct > 0]) / len(c) if c else 0
    @property
    def profit_factor(self):
        w = sum(t.profit_loss for t in self.trades if t.exit_date and t.return_pct > 0)
        l = abs(sum(t.profit_loss for t in self.trades if t.exit_date and t.return_pct < 0))
        return min(w/l, 10) if l else (10 if w else 0)
    @property
    def total_return_pct(self):
        return ((self.final_capital - self.initial_capital) / self.initial_capital) * 100
    @property
    def max_drawdown(self):
        if not self.equity_curve: return 0
        eq = pd.Series(self.equity_curve)
        return abs((eq - eq.cummax()) / eq.cummax()).max() * 100
    @property
    def sharpe_ratio(self):
        if len(self.equity_curve) < 2: return 0
        r = pd.Series(self.equity_curve).pct_change().dropna()
        return (r.mean() / r.std()) * np.sqrt(252) if r.std() else 0


class Strategy:
    def __init__(self, **params):
        self.p = {
            'capital': 1_000_000,
            'max_pos': 3,
            'pos_pct': 0.30,
            'rsi_min': 35,
            'rsi_max': 65,
            'vol_thresh': 1.0,
            'sl_atr': 2.0,
            'tp_atr': 4.0,
            'trail_act': 0.03,
            'trail_pct': 0.02,
            'max_days': 10,
            'use_macd': False,
            **params
        }
        self.capital = self.p['capital']
        self.positions = []
        self.closed = []
        self.equity = []

    def calc(self, df):
        df = df.copy()
        df['SMA5'] = df['Close'].rolling(5).mean()
        df['SMA25'] = df['Close'].rolling(25).mean()
        df['SMA25_up'] = df['SMA25'] > df['SMA25'].shift(1)
        
        # MACD
        e12, e26 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
        df['MACD_H'] = (e12 - e26) - (e12 - e26).ewm(span=9).mean()
        
        # RSI
        d = df['Close'].diff()
        g, l = d.where(d>0,0).rolling(14).mean(), (-d.where(d<0,0)).rolling(14).mean()
        df['RSI'] = 100 - 100/(1 + g/l.replace(0,0.0001))
        
        # ATR
        tr = pd.concat([df['High']-df['Low'], 
                       abs(df['High']-df['Close'].shift()),
                       abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        # Volume
        df['Vol_R'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Ret5'] = df['Close'].pct_change(5)
        return df

    def entry_ok(self, r):
        if any(pd.isna(r.get(c)) for c in ['SMA25','SMA25_up','RSI','ATR','Vol_R','Ret5','Close','SMA5']): 
            return False
        if not r['SMA25_up'] or r['Close'] <= r['SMA25']: return False
        if r['SMA5'] <= r['SMA25']: return False
        if self.p['use_macd'] and r['MACD_H'] <= 0: return False
        if not (self.p['rsi_min'] <= r['RSI'] <= self.p['rsi_max']): return False
        if r['Vol_R'] < self.p['vol_thresh']: return False
        if r['Ret5'] < -0.05: return False
        return True

    def run(self, stocks, start, end):
        self.capital = self.p['capital']
        self.positions, self.closed, self.equity = [], [], [self.capital]
        
        data = {s: self.calc(df) for s, df in stocks.items()}
        dates = sorted(set().union(*[set(df.index) for df in data.values()]))
        dates = [d for d in dates if pd.to_datetime(start) <= d <= pd.to_datetime(end)]
        
        for date in dates:
            # Exit
            for p in self.positions[:]:
                price = data[p.symbol].loc[date,'Close'] if date in data[p.symbol].index else None
                if not price: continue
                
                if price > p.highest_price:
                    p.highest_price = price
                    if (price - p.entry_price)/p.entry_price >= self.p['trail_act']:
                        p.trailing_stop = max(p.trailing_stop, price*(1-self.p['trail_pct']))
                
                reason = None
                if p.trailing_stop and price <= p.trailing_stop: reason = 'trail'
                elif price <= p.stop_loss: reason = 'stop'
                elif price >= p.target_price: reason = 'target'
                elif (date - p.entry_date).days >= self.p['max_days']: reason = 'time'
                
                if reason:
                    p.exit_date, p.exit_price, p.exit_reason = date, price, reason
                    self.capital += p.shares * price * 0.999
                    self.positions.remove(p)
                    self.closed.append(p)
            
            # Entry
            if len(self.positions) < self.p['max_pos']:
                cands = []
                for sym, df in data.items():
                    if any(p.symbol == sym for p in self.positions): continue
                    if date not in df.index: continue
                    row = df.loc[date]
                    if self.entry_ok(row):
                        cands.append((sym, row['Close'], row['ATR'], row['Vol_R']))
                
                cands.sort(key=lambda x: -x[3])
                for sym, price, atr, _ in cands[:self.p['max_pos']-len(self.positions)]:
                    shares = int(self.capital * self.p['pos_pct'] / price)
                    if shares and shares*price <= self.capital:
                        t = Trade(sym, date, price, shares,
                                 price - atr*self.p['sl_atr'],
                                 price + atr*self.p['tp_atr'],
                                 highest_price=price)
                        self.positions.append(t)
                        self.capital -= shares * price * 1.001
            
            eq = self.capital + sum(p.shares * data[p.symbol].loc[date,'Close'] 
                                   for p in self.positions if date in data[p.symbol].index)
            self.equity.append(eq)
        
        # Close remaining
        for p in self.positions[:]:
            price = data[p.symbol].iloc[-1]['Close']
            p.exit_date, p.exit_price, p.exit_reason = dates[-1], price, 'end'
            self.capital += p.shares * price * 0.999
            self.closed.append(p)
        
        return Result(self.closed, self.p['capital'], self.capital, self.equity)


def main():
    print("\n" + "="*70)
    print("🔧 広範囲パラメータ最適化")
    print("="*70)
    
    # Load data
    data = {}
    for f in (DATA_DIR / "stock_prices").glob("*.csv"):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if len(df) > 60: data[f.stem.replace('_','.')] = df
    
    dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    start, end = dates[60].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')
    print(f"銘柄: {len(data)}, 期間: {start} ~ {end}")
    
    # Parameter grid
    configs = [
        # Best from before (baseline)
        {'rsi_min':35, 'rsi_max':65, 'sl_atr':2.0, 'tp_atr':4.0, 'trail_act':0.03, 'max_days':10},
        # Higher TP
        {'rsi_min':35, 'rsi_max':65, 'sl_atr':2.0, 'tp_atr':6.0, 'trail_act':0.03, 'max_days':15},
        # Tighter SL, higher TP
        {'rsi_min':35, 'rsi_max':60, 'sl_atr':1.5, 'tp_atr':6.0, 'trail_act':0.025, 'max_days':12},
        # Very tight SL
        {'rsi_min':40, 'rsi_max':60, 'sl_atr':1.2, 'tp_atr':5.0, 'trail_act':0.02, 'max_days':10},
        # Longer hold
        {'rsi_min':30, 'rsi_max':60, 'sl_atr':2.0, 'tp_atr':8.0, 'trail_act':0.04, 'max_days':20},
        # Very aggressive TP
        {'rsi_min':35, 'rsi_max':55, 'sl_atr':1.5, 'tp_atr':8.0, 'trail_act':0.03, 'max_days':15},
        # Conservative
        {'rsi_min':40, 'rsi_max':55, 'sl_atr':1.5, 'tp_atr':4.0, 'trail_act':0.02, 'max_days':8},
        # Wide RSI, high TP
        {'rsi_min':30, 'rsi_max':70, 'sl_atr':2.0, 'tp_atr':6.0, 'trail_act':0.035, 'max_days':12},
    ]
    
    print(f"\n{'#':>2} {'RSI':>10} {'SL':>5} {'TP':>5} {'Trail':>6} {'Days':>4} | {'N':>4} {'Win':>6} {'PF':>5} {'DD':>5} {'Sharpe':>7} {'Return':>9}")
    print("-"*90)
    
    results = []
    for i, cfg in enumerate(configs):
        s = Strategy(**cfg)
        r = s.run(data, start, end)
        results.append((i, cfg, r))
        print(f"{i+1:>2} {cfg['rsi_min']:>4}-{cfg['rsi_max']:<4} {cfg['sl_atr']:>5.1f} {cfg['tp_atr']:>5.1f} {cfg['trail_act']:>5.1%} {cfg['max_days']:>4} | {r.num_trades:>4} {r.win_rate*100:>5.1f}% {r.profit_factor:>5.2f} {r.max_drawdown:>4.1f}% {r.sharpe_ratio:>7.2f} {r.total_return_pct:>+8.1f}%")
    
    # Best
    best_i, best_cfg, best_r = max(results, key=lambda x: x[2].total_return_pct)
    print("\n" + "="*70)
    print(f"🏆 最良: #{best_i+1}")
    print(f"   RSI: {best_cfg['rsi_min']}-{best_cfg['rsi_max']}")
    print(f"   SL/TP: {best_cfg['sl_atr']}/{best_cfg['tp_atr']} ATR")
    print(f"   Trail: {best_cfg['trail_act']:.1%}")
    print(f"   Max Days: {best_cfg['max_days']}")
    print(f"   Return: {best_r.total_return_pct:+.1f}%")
    print(f"   Sharpe: {best_r.sharpe_ratio:.2f}")

if __name__ == "__main__":
    main()
