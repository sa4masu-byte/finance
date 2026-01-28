#!/usr/bin/env python3
"""
新戦略テストスクリプト
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.settings import DATA_DIR
from backtesting.simple_winning_strategy import SimpleWinningStrategy, run_comparison
from backtesting.backtest_engine import BacktestEngine
from backtesting.enhanced_backtest_engine import EnhancedBacktestEngine


def load_data():
    """データ読み込み"""
    stock_data = {}
    data_dir = DATA_DIR / "stock_prices"

    for csv_file in data_dir.glob("*.csv"):
        symbol = csv_file.stem.replace('_', '.')
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        if len(df) > 60:
            stock_data[symbol] = df

    return stock_data


def main():
    print("\n" + "=" * 70)
    print("🚀 新戦略テスト")
    print("=" * 70)

    # データ読み込み
    stock_data = load_data()
    print(f"銘柄数: {len(stock_data)}")

    if len(stock_data) < 3:
        print("データ不足")
        return

    # 日付範囲
    all_dates = sorted(set().union(*[set(df.index) for df in stock_data.values()]))
    start_date = all_dates[60].strftime('%Y-%m-%d')
    end_date = all_dates[-1].strftime('%Y-%m-%d')
    print(f"期間: {start_date} ~ {end_date}")

    # 比較実行
    print("\n" + "=" * 70)
    print("📊 戦略比較")
    print("=" * 70)

    # 1. 基本版
    print("\n[1/3] 基本版...")
    basic = BacktestEngine(initial_capital=1_000_000, max_positions=3)
    basic_r = basic.run_backtest(stock_data, start_date, end_date)

    # 2. 改善版
    print("[2/3] 改善版...")
    enhanced = EnhancedBacktestEngine(
        initial_capital=1_000_000, max_positions=3,
        min_score=70, min_confidence=0.70
    )
    enhanced_r = enhanced.run_backtest(stock_data, start_date, end_date, {})

    # 3. 新シンプル戦略
    print("[3/3] 新シンプル戦略...")
    simple = SimpleWinningStrategy(initial_capital=1_000_000, max_positions=3)
    simple_r = simple.run_backtest(stock_data, start_date, end_date)

    # 結果表示
    print("\n" + "=" * 70)
    print("📈 結果")
    print("=" * 70)

    results = [
        ("基本版", basic_r),
        ("改善版", enhanced_r),
        ("新戦略", simple_r),
    ]

    print(f"\n{'戦略':<12} {'取引':>6} {'勝率':>8} {'PF':>7} {'Sharpe':>8} {'DD':>7} {'Return':>10} {'最終資産':>12}")
    print("-" * 85)

    for name, r in results:
        print(f"{name:<12} {r.num_trades:>6} {r.win_rate*100:>7.1f}% {r.profit_factor:>7.2f} {r.sharpe_ratio:>8.2f} {r.max_drawdown:>6.1f}% {r.total_return_pct:>+9.1f}% {r.final_capital:>12,.0f}")

    # 勝者判定
    print("\n" + "=" * 70)
    best = max(results, key=lambda x: x[1].total_return_pct)
    print(f"🏆 最良戦略: {best[0]} (Return: {best[1].total_return_pct:+.1f}%)")

    # 新戦略の詳細
    if simple_r.trades:
        print("\n" + "=" * 70)
        print("📋 新戦略 取引詳細")
        print("=" * 70)

        # Exit reason別集計
        reasons = {}
        for t in simple_r.trades:
            if t.exit_reason:
                if t.exit_reason not in reasons:
                    reasons[t.exit_reason] = {'count': 0, 'total': 0}
                reasons[t.exit_reason]['count'] += 1
                reasons[t.exit_reason]['total'] += t.return_pct

        print("\nExit Reason別:")
        for reason, stats in sorted(reasons.items(), key=lambda x: -x[1]['count']):
            avg = stats['total'] / stats['count'] if stats['count'] else 0
            print(f"  {reason:<20}: {stats['count']:>3}回, 平均{avg:+.2f}%")


if __name__ == "__main__":
    main()
