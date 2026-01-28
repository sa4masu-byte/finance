#!/usr/bin/env python3
"""
推奨銘柄の詳細分析付き出力
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from src.data.fetcher import StockDataFetcher
from backtesting.optimized_strategy import OptimizedStrategy


def load_watchlist():
    f = DATA_DIR / "watchlist.json"
    if f.exists():
        with open(f) as fp:
            data = json.load(fp)
            return data.get("symbols", [])[:50], data.get("sectors", {})
    return ["7203.JP", "6758.JP", "9984.JP"], {}


def get_sector(symbol, sectors):
    """銘柄のセクターを取得"""
    for sector, symbols in sectors.items():
        if symbol in symbols:
            return sector
    return "不明"


def analyze_stock(symbol, df, strategy):
    """銘柄の詳細分析"""
    df = strategy.calculate_indicators(df)
    if df.empty:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    # トレンド分析
    sma25_trend = "上昇" if latest['SMA_Rising'] else "下降"
    price_vs_sma = ((latest['Close'] - latest['SMA_25']) / latest['SMA_25']) * 100

    # モメンタム
    rsi = latest['RSI']
    rsi_status = "適正" if 32 <= rsi <= 68 else ("過熱" if rsi > 68 else "売られすぎ")

    # 出来高
    vol_ratio = latest['Volume_Ratio']
    vol_status = "急増" if vol_ratio > 2 else ("増加" if vol_ratio > 1.2 else "通常")

    # 直近パフォーマンス
    ret_5d = latest['Return_5d'] * 100 if not pd.isna(latest['Return_5d']) else 0
    ret_20d = ((latest['Close'] / df['Close'].iloc[-20]) - 1) * 100 if len(df) >= 20 else 0

    # ATR%
    atr_pct = (latest['ATR'] / latest['Close']) * 100

    return {
        'symbol': symbol,
        'price': latest['Close'],
        'sma25': latest['SMA_25'],
        'sma25_trend': sma25_trend,
        'price_vs_sma': price_vs_sma,
        'rsi': rsi,
        'rsi_status': rsi_status,
        'vol_ratio': vol_ratio,
        'vol_status': vol_status,
        'ret_5d': ret_5d,
        'ret_20d': ret_20d,
        'atr': latest['ATR'],
        'atr_pct': atr_pct,
        'stop_loss': latest['Close'] - latest['ATR'] * 2.2,
        'target': latest['Close'] + latest['ATR'] * 3.5,
    }


def calculate_correlations(stock_data, target_symbols):
    """銘柄間の相関を計算"""
    # リターンを計算
    returns = {}
    for symbol, df in stock_data.items():
        if len(df) > 20:
            returns[symbol] = df['Close'].pct_change().dropna()

    if not returns:
        return {}

    # DataFrame作成
    ret_df = pd.DataFrame(returns)

    # 相関行列
    corr_matrix = ret_df.corr()

    # ターゲット銘柄との相関
    correlations = {}
    for sym in target_symbols:
        if sym in corr_matrix.columns:
            # 相関が高い銘柄TOP3
            corrs = corr_matrix[sym].drop(sym).sort_values(ascending=False)
            correlations[sym] = corrs.head(3).to_dict()

    return correlations


def main():
    print("\n" + "=" * 70)
    print("📊 推奨銘柄 詳細分析")
    print("=" * 70)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # データ取得
    fetcher = StockDataFetcher()
    watchlist, sectors = load_watchlist()

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
        except:
            pass
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(watchlist)}")

    print(f"取得完了: {len(stock_data)}銘柄\n")

    # シグナル取得
    strategy = OptimizedStrategy(max_positions=5)
    signals = strategy.get_today_signals(stock_data)

    if not signals:
        print("⚠️ 本日の推奨銘柄はありません")
        return

    # 推奨銘柄の詳細分析
    target_symbols = [s['symbol'] for s in signals]

    # 相関分析
    correlations = calculate_correlations(stock_data, target_symbols)

    print("=" * 70)
    print(f"🎯 推奨銘柄: {len(signals)}件")
    print("=" * 70)

    for i, sig in enumerate(signals):
        symbol = sig['symbol']
        analysis = analyze_stock(symbol, stock_data[symbol], strategy)
        sector = get_sector(symbol, sectors)

        print(f"\n{'─'*70}")
        print(f"【{i+1}】{symbol} ({sector})")
        print(f"{'─'*70}")

        print(f"\n  📈 価格情報:")
        print(f"     現在値:     ¥{analysis['price']:,.0f}")
        print(f"     25日SMA:    ¥{analysis['sma25']:,.0f} ({analysis['sma25_trend']}トレンド)")
        print(f"     SMAとの乖離: {analysis['price_vs_sma']:+.1f}%")

        print(f"\n  📊 テクニカル指標:")
        print(f"     RSI:        {analysis['rsi']:.1f} ({analysis['rsi_status']})")
        print(f"     出来高比:   {analysis['vol_ratio']:.1f}x ({analysis['vol_status']})")
        print(f"     ATR:        ¥{analysis['atr']:,.0f} ({analysis['atr_pct']:.1f}%)")

        print(f"\n  📉 パフォーマンス:")
        print(f"     5日リターン:  {analysis['ret_5d']:+.1f}%")
        print(f"     20日リターン: {analysis['ret_20d']:+.1f}%")

        print(f"\n  🎯 推奨売買価格:")
        print(f"     ストップロス: ¥{analysis['stop_loss']:,.0f} ({((analysis['stop_loss']/analysis['price'])-1)*100:+.1f}%)")
        print(f"     利確目標:     ¥{analysis['target']:,.0f} ({((analysis['target']/analysis['price'])-1)*100:+.1f}%)")

        print(f"\n  ✅ 推奨理由:")
        reasons = []
        if analysis['sma25_trend'] == "上昇":
            reasons.append("25日SMAが上昇トレンド")
        if analysis['price_vs_sma'] > 0:
            reasons.append(f"株価がSMA25を{analysis['price_vs_sma']:.1f}%上回る")
        if 32 <= analysis['rsi'] <= 68:
            reasons.append(f"RSI {analysis['rsi']:.0f}で適正範囲")
        if analysis['vol_ratio'] > 1.0:
            reasons.append(f"出来高が平均の{analysis['vol_ratio']:.1f}倍")
        if analysis['ret_5d'] > -5:
            reasons.append("直近5日で急落なし")

        for r in reasons:
            print(f"     • {r}")

        # 相関銘柄
        if symbol in correlations:
            print(f"\n  🔗 相関が高い銘柄:")
            for corr_sym, corr_val in correlations[symbol].items():
                print(f"     • {corr_sym}: {corr_val:.2f}")

    # サマリー
    print(f"\n{'='*70}")
    print("📋 エントリールール:")
    print("   • 推奨価格付近で成行または指値で買い")
    print("   • ストップロスは必ず設定（損切りライン厳守）")
    print("   • 利確目標に達したら利益確定")
    print("   • 最大10日保有、届かなければ手仕舞い")
    print("   • 1銘柄あたりポートフォリオの30%以下")
    print("=" * 70)


if __name__ == "__main__":
    main()
