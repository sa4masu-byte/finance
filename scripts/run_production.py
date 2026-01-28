#!/usr/bin/env python3
"""
本番用 - 本日の推奨銘柄を出力

最適化済みパラメータ:
- RSI: 32-68, SL: 2.2 ATR, TP: 3.5 ATR
- Trail: 3.5%, Max Days: 10
- Backtest Return: +32.0%, Sharpe: 2.82
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from src.data.fetcher import StockDataFetcher
from backtesting.optimized_strategy import OptimizedStrategy


def load_watchlist():
    """監視銘柄読み込み"""
    f = DATA_DIR / "watchlist.json"
    if f.exists():
        with open(f) as fp:
            return json.load(fp).get("symbols", [])[:50]
    return ["7203.JP", "6758.JP", "9984.JP", "8306.JP", "9432.JP"]


def main():
    print("\n" + "=" * 60)
    print("📈 本日の推奨銘柄")
    print("=" * 60)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"戦略: 最適化済み (Return +32%, Sharpe 2.82)")
    print()

    # データ取得
    fetcher = StockDataFetcher()
    watchlist = load_watchlist()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)

    print(f"データ取得中... ({len(watchlist)}銘柄)")

    stock_data = {}
    for i, symbol in enumerate(watchlist):
        try:
            df = fetcher.fetch_stock_data(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            if df is not None and len(df) > 30:
                stock_data[symbol] = df
        except Exception as e:
            pass

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(watchlist)}")

    print(f"取得完了: {len(stock_data)}銘柄\n")

    if not stock_data:
        print("⚠️ データ取得失敗")
        return 1

    # シグナル取得
    strategy = OptimizedStrategy(max_positions=5)
    signals = strategy.get_today_signals(stock_data)

    # 結果表示
    print("=" * 60)
    if signals:
        print(f"🎯 推奨銘柄: {len(signals)}件\n")
        print(f"{'銘柄':<12} {'現在値':>10} {'ストップ':>10} {'利確目標':>10} {'RSI':>6} {'出来高比':>8}")
        print("-" * 60)

        for s in signals:
            print(f"{s['symbol']:<12} {s['price']:>10,.0f} {s['stop_loss']:>10,.0f} {s['target']:>10,.0f} {s['rsi']:>6.1f} {s['volume_ratio']:>7.1f}x")

        print("\n📋 エントリールール:")
        print("   • 推奨価格付近で買い")
        print("   • ストップロスは必ず設定")
        print("   • 利確目標に達したら利確")
        print("   • 最大10日保有")
    else:
        print("⚠️ 本日の推奨銘柄はありません")
        print("   エントリー条件を満たす銘柄がありませんでした")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
