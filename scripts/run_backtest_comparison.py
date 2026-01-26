#!/usr/bin/env python3
"""
バックテスト比較スクリプト

基本バックテスト vs 改善版バックテストを比較して
各改善施策の効果を検証する

改善施策:
1. ADX (トレンド強度)
2. マルチタイムフレーム分析
3. 決算フィルター
4. アンサンブルML
5. 動的ストップロス
"""
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, REPORTS_DIR
from src.data.fetcher import StockDataFetcher
from backtesting.backtest_engine import BacktestEngine, BacktestResults
from backtesting.enhanced_backtest_engine import EnhancedBacktestEngine, EnhancedBacktestResults

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_watchlist() -> List[str]:
    """監視銘柄リストを読み込む"""
    watchlist_file = DATA_DIR / "watchlist.json"

    if watchlist_file.exists():
        with open(watchlist_file, 'r') as f:
            data = json.load(f)
            symbols = data.get("symbols", [])
            # バックテスト用に銘柄数を制限（高速化）
            return symbols[:30]  # 上位30銘柄

    return [
        "7203.JP", "6758.JP", "9984.JP", "6861.JP", "8306.JP",
        "9432.JP", "6501.JP", "7267.JP", "8035.JP", "6902.JP",
        "4063.JP", "6098.JP", "4519.JP", "6367.JP", "7741.JP",
    ]


def load_sector_map() -> Dict[str, str]:
    """セクターマップを読み込む"""
    watchlist_file = DATA_DIR / "watchlist.json"

    if watchlist_file.exists():
        with open(watchlist_file, 'r') as f:
            data = json.load(f)
            sectors = data.get("sectors", {})

            # Invert the sector map
            sector_map = {}
            for sector_name, symbols in sectors.items():
                for symbol in symbols:
                    sector_map[symbol] = sector_name
            return sector_map

    return {}


def fetch_historical_data(symbols: List[str], days: int = 365) -> Dict:
    """過去データを取得"""
    fetcher = StockDataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    stock_data = {}
    failed = []

    logger.info(f"データ取得中... ({len(symbols)}銘柄)")

    for i, symbol in enumerate(symbols):
        try:
            df = fetcher.fetch_stock_data(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            if df is not None and len(df) > 60:  # 最低60日分
                stock_data[symbol] = df
                logger.debug(f"  ✓ {symbol}: {len(df)} days")
            else:
                failed.append(symbol)
        except Exception as e:
            logger.warning(f"  ✗ {symbol}: {e}")
            failed.append(symbol)

        # 進捗表示
        if (i + 1) % 10 == 0:
            logger.info(f"  進捗: {i + 1}/{len(symbols)}")

    logger.info(f"データ取得完了: {len(stock_data)}/{len(symbols)} 銘柄")
    return stock_data


def run_basic_backtest(
    stock_data: Dict,
    start_date: str,
    end_date: str,
) -> BacktestResults:
    """基本バックテストを実行"""
    engine = BacktestEngine(
        initial_capital=1_000_000,
        max_positions=3,
    )
    return engine.run_backtest(stock_data, start_date, end_date)


def run_enhanced_backtest(
    stock_data: Dict,
    start_date: str,
    end_date: str,
    sector_map: Dict[str, str],
    min_score: float = 55,
    min_confidence: float = 0.60,
) -> EnhancedBacktestResults:
    """改善版バックテストを実行（全機能有効）"""
    engine = EnhancedBacktestEngine(
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
        min_score=min_score,
        min_confidence=min_confidence,
    )
    return engine.run_backtest(stock_data, start_date, end_date, sector_map)


def run_minimal_backtest(
    stock_data: Dict,
    start_date: str,
    end_date: str,
    sector_map: Dict[str, str],
) -> EnhancedBacktestResults:
    """最小限の改善（ADXのみ）"""
    engine = EnhancedBacktestEngine(
        initial_capital=1_000_000,
        max_positions=3,
        enable_trailing_stop=False,
        enable_volatility_sizing=False,
        enable_market_regime=False,
        enable_multi_timeframe=False,
        enable_volume_breakout=False,
        enable_swing_low_stop=False,
        enable_additional_filters=False,
        enable_compound=False,
        min_score=55,
        min_confidence=0.60,
    )
    return engine.run_backtest(stock_data, start_date, end_date, sector_map)


def print_results(name: str, results):
    """結果を表示"""
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print(f"{'='*60}")
    print(f"  取引回数:     {results.num_trades:,}")
    print(f"  勝率:         {results.win_rate:.1%}")
    print(f"  平均勝ち:     {results.avg_win:+.2f}%")
    print(f"  平均負け:     {results.avg_loss:+.2f}%")
    print(f"  プロフィット: {results.profit_factor:.2f}")
    print(f"  シャープ比:   {results.sharpe_ratio:.2f}")
    print(f"  最大DD:       {results.max_drawdown:.1f}%")
    print(f"  総リターン:   {results.total_return_pct:+.2f}%")
    print(f"  最終資産:     ¥{results.final_capital:,.0f}")
    print(f"  平均保有日数: {results.avg_holding_days:.1f}日")


def print_comparison(basic, enhanced):
    """比較結果を表示"""
    print(f"\n{'='*60}")
    print("📈 改善効果の比較")
    print(f"{'='*60}")

    metrics = [
        ("勝率", basic.win_rate * 100, enhanced.win_rate * 100, "%"),
        ("プロフィット", basic.profit_factor, enhanced.profit_factor, ""),
        ("シャープ比", basic.sharpe_ratio, enhanced.sharpe_ratio, ""),
        ("最大DD", basic.max_drawdown, enhanced.max_drawdown, "%"),
        ("総リターン", basic.total_return_pct, enhanced.total_return_pct, "%"),
    ]

    print(f"\n{'指標':<15} {'基本':>12} {'改善版':>12} {'改善幅':>12}")
    print("-" * 55)

    for name, basic_val, enhanced_val, unit in metrics:
        if name == "最大DD":
            # DDは低い方が良い
            change = basic_val - enhanced_val
            indicator = "✓" if change > 0 else "✗"
        else:
            change = enhanced_val - basic_val
            indicator = "✓" if change > 0 else "✗"

        print(f"{name:<15} {basic_val:>10.2f}{unit:>2} {enhanced_val:>10.2f}{unit:>2} {change:>+10.2f}{unit:>2} {indicator}")


def save_results(basic_results, enhanced_results, output_dir: Path):
    """結果をJSONで保存"""
    output_file = output_dir / f"backtest_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    def serialize_results(results, name):
        return {
            "name": name,
            "num_trades": results.num_trades,
            "win_rate": results.win_rate,
            "avg_win": results.avg_win,
            "avg_loss": results.avg_loss,
            "profit_factor": results.profit_factor,
            "sharpe_ratio": results.sharpe_ratio,
            "max_drawdown": results.max_drawdown,
            "total_return_pct": results.total_return_pct,
            "final_capital": results.final_capital,
            "avg_holding_days": results.avg_holding_days,
        }

    output_data = {
        "date": datetime.now().isoformat(),
        "basic": serialize_results(basic_results, "基本バックテスト"),
        "enhanced": serialize_results(enhanced_results, "改善版バックテスト"),
        "improvement": {
            "win_rate_change": enhanced_results.win_rate - basic_results.win_rate,
            "profit_factor_change": enhanced_results.profit_factor - basic_results.profit_factor,
            "return_change": enhanced_results.total_return_pct - basic_results.total_return_pct,
            "max_dd_reduction": basic_results.max_drawdown - enhanced_results.max_drawdown,
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"結果を保存: {output_file}")


def main():
    """メイン処理"""
    print("\n" + "=" * 60)
    print("🔬 バックテスト比較分析")
    print("=" * 60)

    # 1. データ準備
    watchlist = load_watchlist()
    sector_map = load_sector_map()

    logger.info(f"対象銘柄数: {len(watchlist)}")

    # 過去1年分のデータを取得
    stock_data = fetch_historical_data(watchlist, days=365)

    if len(stock_data) < 5:
        logger.error("十分なデータがありません")
        return 1

    # バックテスト期間を設定
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=300)).strftime("%Y-%m-%d")

    logger.info(f"バックテスト期間: {start_date} ~ {end_date}")

    # 2. 基本バックテスト実行
    logger.info("\n基本バックテスト実行中...")
    basic_results = run_basic_backtest(stock_data, start_date, end_date)
    print_results("基本バックテスト", basic_results)

    # 3. 改善版バックテスト実行
    logger.info("\n改善版バックテスト実行中...")
    enhanced_results = run_enhanced_backtest(
        stock_data, start_date, end_date, sector_map
    )
    print_results("改善版バックテスト（全機能）", enhanced_results)

    # 4. 比較分析
    print_comparison(basic_results, enhanced_results)

    # 5. 結果保存
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results(basic_results, enhanced_results, REPORTS_DIR)

    # サマリー
    print(f"\n{'='*60}")
    print("✅ バックテスト完了")
    print(f"{'='*60}")

    # 改善効果の判定
    win_rate_improved = enhanced_results.win_rate > basic_results.win_rate
    return_improved = enhanced_results.total_return_pct > basic_results.total_return_pct
    dd_reduced = enhanced_results.max_drawdown < basic_results.max_drawdown

    print(f"\n改善効果サマリー:")
    print(f"  勝率向上:       {'✅' if win_rate_improved else '❌'}")
    print(f"  リターン向上:   {'✅' if return_improved else '❌'}")
    print(f"  リスク低減:     {'✅' if dd_reduced else '❌'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
