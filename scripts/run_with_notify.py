#!/usr/bin/env python3
"""
本番スキャン + 通知送信

使い方:
    # 通常実行（通知あり）
    python scripts/run_with_notify.py

    # 通知なし（テスト）
    python scripts/run_with_notify.py --no-notify

環境変数:
    LINE_NOTIFY_TOKEN    - LINE Notifyのトークン
    SLACK_WEBHOOK_URL    - SlackのWebhook URL
    DISCORD_WEBHOOK_URL  - DiscordのWebhook URL
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from src.data.fetcher import StockDataFetcher
from src.notifications.notifier import Notifier
from backtesting.optimized_strategy import OptimizedStrategy


def load_watchlist():
    f = DATA_DIR / "watchlist.json"
    if f.exists():
        with open(f) as fp:
            return json.load(fp).get("symbols", [])
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-notify", action="store_true", help="通知を送信しない")
    parser.add_argument("--limit", type=int, default=300, help="スキャン銘柄数")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("📈 株式推奨スキャン")
    print("=" * 60)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"通知: {'OFF' if args.no_notify else 'ON'}")
    print()

    # データ取得
    fetcher = StockDataFetcher()
    watchlist = load_watchlist()[:args.limit]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)

    print(f"データ取得中... ({len(watchlist)}銘柄)")

    stock_data = {}
    errors = 0
    for i, symbol in enumerate(watchlist):
        try:
            df = fetcher.fetch_stock_data(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            if df is not None and len(df) > 30:
                stock_data[symbol] = df
        except:
            errors += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(watchlist)} ({len(stock_data)}成功, {errors}エラー)")

    print(f"\n取得完了: {len(stock_data)}/{len(watchlist)}銘柄")

    if not stock_data:
        print("⚠️ データ取得失敗")
        return 1

    # シグナル取得
    strategy = OptimizedStrategy(max_positions=10)
    signals = strategy.get_today_signals(stock_data)

    # 結果表示
    print("\n" + "=" * 60)
    if signals:
        print(f"🎯 推奨銘柄: {len(signals)}件\n")
        print(f"{'銘柄':<12} {'現在値':>10} {'損切り':>10} {'利確':>10} {'RSI':>6}")
        print("-" * 55)
        for s in signals:
            print(f"{s['symbol']:<12} {s['price']:>10,.0f} {s['stop_loss']:>10,.0f} {s['target']:>10,.0f} {s['rsi']:>6.1f}")
    else:
        print("⚠️ 本日の推奨銘柄はありません")

    # 通知送信
    if not args.no_notify:
        print("\n" + "=" * 60)
        print("📤 通知送信中...")

        notifier = Notifier()
        results = notifier.notify_all(signals)

        if results:
            for channel, success in results.items():
                status = "✅" if success else "❌"
                print(f"  {channel}: {status}")
        else:
            print("  通知先が設定されていません")
            print("  環境変数を設定してください:")
            print("    export LINE_NOTIFY_TOKEN='your-token'")
            print("    export SLACK_WEBHOOK_URL='your-url'")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
