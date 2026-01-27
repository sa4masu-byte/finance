#!/usr/bin/env python3
"""
ワンクリック データダウンロード＆バックテスト検証スクリプト

使い方:
    python scripts/download_and_backtest.py

機能:
    1. 監視銘柄の株価データをダウンロード（1年分）
    2. CSVファイルとして保存
    3. バックテスト比較を実行
    4. 結果をレポート出力
"""
import sys
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, REPORTS_DIR

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# データ保存ディレクトリ
STOCK_DATA_DIR = DATA_DIR / "stock_prices"
STOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_watchlist() -> List[str]:
    """監視銘柄リストを取得"""
    watchlist_file = DATA_DIR / "watchlist.json"

    if watchlist_file.exists():
        with open(watchlist_file, 'r') as f:
            data = json.load(f)
            # バックテスト用に上位銘柄のみ
            return data.get("symbols", [])[:30]

    return [
        "7203.JP", "6758.JP", "9984.JP", "6861.JP", "8306.JP",
        "9432.JP", "6501.JP", "7267.JP", "8035.JP", "6902.JP",
    ]


def download_stock_data_stooq(symbols: List[str], days: int = 365) -> Dict[str, 'pd.DataFrame']:
    """
    Stooqから株価データをダウンロード
    """
    try:
        import pandas as pd
        from pandas_datareader import data as pdr
    except ImportError:
        logger.error("pandas-datareader が必要です: pip install pandas-datareader")
        return {}

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    stock_data = {}
    failed = []

    logger.info(f"📥 データダウンロード開始: {len(symbols)}銘柄")
    logger.info(f"   期間: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")

    for i, symbol in enumerate(symbols):
        try:
            # Stooq形式に変換
            stooq_symbol = symbol.split('.')[0] + ".JP"

            logger.info(f"  [{i+1}/{len(symbols)}] {symbol} ダウンロード中...")

            df = pdr.DataReader(
                stooq_symbol,
                'stooq',
                start=start_date,
                end=end_date
            )

            if df is not None and len(df) > 60:
                df = df.sort_index()  # Stooqは逆順なのでソート
                stock_data[symbol] = df

                # CSVとして保存
                save_path = STOCK_DATA_DIR / f"{symbol.replace('.', '_')}.csv"
                df.to_csv(save_path)
                logger.info(f"    ✓ {len(df)}日分 → {save_path.name}")
            else:
                failed.append(symbol)
                logger.warning(f"    ✗ データ不足")

            # Rate limiting
            time.sleep(1)

        except Exception as e:
            failed.append(symbol)
            logger.warning(f"    ✗ エラー: {e}")

    logger.info(f"\n📊 ダウンロード完了: {len(stock_data)}/{len(symbols)} 銘柄")

    if failed:
        logger.warning(f"   失敗: {', '.join(failed[:5])}{'...' if len(failed) > 5 else ''}")

    return stock_data


def load_saved_data() -> Dict[str, 'pd.DataFrame']:
    """保存済みのCSVデータを読み込む"""
    import pandas as pd

    stock_data = {}
    csv_files = list(STOCK_DATA_DIR.glob("*.csv"))

    if not csv_files:
        logger.warning("保存済みデータがありません")
        return {}

    logger.info(f"📂 保存済みデータを読み込み中: {len(csv_files)}ファイル")

    for csv_file in csv_files:
        try:
            symbol = csv_file.stem.replace('_', '.')
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

            if len(df) > 60:
                stock_data[symbol] = df
                logger.debug(f"  ✓ {symbol}: {len(df)}日分")
        except Exception as e:
            logger.warning(f"  ✗ {csv_file.name}: {e}")

    logger.info(f"   読み込み完了: {len(stock_data)}銘柄")
    return stock_data


def run_backtest_comparison(stock_data: Dict) -> Dict:
    """バックテスト比較を実行"""
    import pandas as pd
    from backtesting.backtest_engine import BacktestEngine
    from backtesting.enhanced_backtest_engine import EnhancedBacktestEngine

    if len(stock_data) < 3:
        logger.error("バックテストには最低3銘柄必要です")
        return {}

    # 日付範囲を取得
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    start_date = all_dates[60].strftime("%Y-%m-%d")  # 指標計算に60日必要
    end_date = all_dates[-1].strftime("%Y-%m-%d")

    logger.info(f"\n🔬 バックテスト実行")
    logger.info(f"   期間: {start_date} ~ {end_date}")
    logger.info(f"   銘柄数: {len(stock_data)}")

    # セクターマップを読み込み
    sector_map = load_sector_map()

    # 基本バックテスト
    logger.info("\n   [1/2] 基本バックテスト...")
    basic_engine = BacktestEngine(
        initial_capital=1_000_000,
        max_positions=3,
    )
    basic_results = basic_engine.run_backtest(stock_data, start_date, end_date)

    # 改善版バックテスト
    logger.info("   [2/2] 改善版バックテスト...")
    enhanced_engine = EnhancedBacktestEngine(
        initial_capital=1_000_000,
        max_positions=3,
        enable_trailing_stop=True,
        enable_volatility_sizing=True,
        enable_market_regime=True,
        enable_multi_timeframe=True,
        enable_volume_breakout=True,
        enable_swing_low_stop=True,
        enable_additional_filters=True,
        enable_compound=True,
        min_score=55,
        min_confidence=0.60,
    )
    enhanced_results = enhanced_engine.run_backtest(stock_data, start_date, end_date, sector_map)

    return {
        "basic": basic_results,
        "enhanced": enhanced_results,
        "start_date": start_date,
        "end_date": end_date,
    }


def load_sector_map() -> Dict[str, str]:
    """セクターマップを読み込む"""
    watchlist_file = DATA_DIR / "watchlist.json"

    if watchlist_file.exists():
        with open(watchlist_file, 'r') as f:
            data = json.load(f)
            sectors = data.get("sectors", {})

            sector_map = {}
            for sector_name, symbols in sectors.items():
                for symbol in symbols:
                    sector_map[symbol] = sector_name
            return sector_map

    return {}


def print_results(results: Dict):
    """結果を表示"""
    basic = results["basic"]
    enhanced = results["enhanced"]

    print("\n" + "=" * 70)
    print("📊 バックテスト結果比較")
    print("=" * 70)
    print(f"期間: {results['start_date']} ~ {results['end_date']}")

    print(f"\n{'指標':<20} {'基本':>15} {'改善版':>15} {'変化':>15}")
    print("-" * 70)

    metrics = [
        ("取引回数", basic.num_trades, enhanced.num_trades, ""),
        ("勝率", basic.win_rate * 100, enhanced.win_rate * 100, "%"),
        ("平均勝ち", basic.avg_win, enhanced.avg_win, "%"),
        ("平均負け", basic.avg_loss, enhanced.avg_loss, "%"),
        ("プロフィットF", basic.profit_factor, enhanced.profit_factor, ""),
        ("シャープ比", basic.sharpe_ratio, enhanced.sharpe_ratio, ""),
        ("最大DD", basic.max_drawdown, enhanced.max_drawdown, "%"),
        ("総リターン", basic.total_return_pct, enhanced.total_return_pct, "%"),
        ("最終資産", basic.final_capital, enhanced.final_capital, "円"),
    ]

    for name, b_val, e_val, unit in metrics:
        if unit == "円":
            change = e_val - b_val
            print(f"{name:<20} {b_val:>14,.0f} {e_val:>14,.0f} {change:>+14,.0f}")
        else:
            change = e_val - b_val
            if name == "最大DD":
                indicator = "✓" if change < 0 else "✗"
            else:
                indicator = "✓" if change > 0 else "✗"
            print(f"{name:<20} {b_val:>13.2f}{unit:>2} {e_val:>13.2f}{unit:>2} {change:>+13.2f}{unit:>2} {indicator}")

    # サマリー
    print("\n" + "=" * 70)
    print("📈 改善効果サマリー")
    print("=" * 70)

    improvements = [
        ("勝率", enhanced.win_rate > basic.win_rate,
         f"{(enhanced.win_rate - basic.win_rate) * 100:+.1f}%"),
        ("リターン", enhanced.total_return_pct > basic.total_return_pct,
         f"{enhanced.total_return_pct - basic.total_return_pct:+.1f}%"),
        ("シャープ比", enhanced.sharpe_ratio > basic.sharpe_ratio,
         f"{enhanced.sharpe_ratio - basic.sharpe_ratio:+.2f}"),
        ("リスク(DD)", enhanced.max_drawdown < basic.max_drawdown,
         f"{basic.max_drawdown - enhanced.max_drawdown:+.1f}%"),
    ]

    for name, improved, change in improvements:
        status = "✅" if improved else "❌"
        print(f"  {name:<15}: {status} ({change})")


def save_results_json(results: Dict):
    """結果をJSONで保存"""
    basic = results["basic"]
    enhanced = results["enhanced"]

    output_file = REPORTS_DIR / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def serialize(r):
        return {
            "num_trades": r.num_trades,
            "win_rate": float(r.win_rate),
            "avg_win": float(r.avg_win),
            "avg_loss": float(r.avg_loss),
            "profit_factor": float(r.profit_factor),
            "sharpe_ratio": float(r.sharpe_ratio),
            "max_drawdown": float(r.max_drawdown),
            "total_return_pct": float(r.total_return_pct),
            "final_capital": float(r.final_capital),
        }

    output_data = {
        "date": datetime.now().isoformat(),
        "period": {
            "start": results["start_date"],
            "end": results["end_date"],
        },
        "basic": serialize(basic),
        "enhanced": serialize(enhanced),
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n📁 結果保存: {output_file}")


def main():
    """メイン処理"""
    print("\n" + "=" * 70)
    print("🚀 ワンクリック データダウンロード＆バックテスト検証")
    print("=" * 70)

    # 1. 保存済みデータを確認
    stock_data = load_saved_data()

    # 2. データがなければダウンロード
    if len(stock_data) < 5:
        logger.info("\n保存済みデータが不足しています。ダウンロードを開始します...")

        try:
            symbols = get_watchlist()
            stock_data = download_stock_data_stooq(symbols, days=365)
        except Exception as e:
            logger.error(f"ダウンロードエラー: {e}")
            logger.info("\n💡 手動でデータを配置する場合:")
            logger.info(f"   CSVファイルを {STOCK_DATA_DIR} に配置してください")
            logger.info("   形式: Date,Open,High,Low,Close,Volume")
            return 1

    if len(stock_data) < 3:
        logger.error("バックテストに必要なデータが不足しています")
        logger.info(f"\n💡 CSVファイルを {STOCK_DATA_DIR} に配置してください")
        return 1

    # 3. バックテスト実行
    try:
        results = run_backtest_comparison(stock_data)

        if results:
            # 4. 結果表示
            print_results(results)

            # 5. 結果保存
            save_results_json(results)

            print("\n✅ 検証完了!")
        else:
            logger.error("バックテストに失敗しました")
            return 1

    except Exception as e:
        logger.error(f"バックテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
