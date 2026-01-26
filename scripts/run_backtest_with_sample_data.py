#!/usr/bin/env python3
"""
バックテスト比較スクリプト（サンプルデータ版）

サンプルの株価データを生成してバックテストを実行
実際の運用ではStooqからデータを取得する
"""
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, REPORTS_DIR
from backtesting.backtest_engine import BacktestEngine, BacktestResults
from backtesting.enhanced_backtest_engine import EnhancedBacktestEngine, EnhancedBacktestResults

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def generate_sample_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    base_price: float = 1000,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seed: int = None
) -> pd.DataFrame:
    """
    サンプル株価データを生成

    リアルな株価の特性をシミュレート:
    - ランダムウォーク + トレンド
    - 出来高の変動
    - ギャップ（始値と前日終値の差）
    """
    if seed is not None:
        np.random.seed(seed)

    # 日付範囲を作成
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    n_days = len(dates)

    # 価格シミュレーション（幾何ブラウン運動）
    returns = np.random.normal(trend, volatility, n_days)
    prices = base_price * np.exp(np.cumsum(returns))

    # OHLC生成
    daily_volatility = volatility * np.sqrt(252)
    highs = prices * (1 + np.random.uniform(0, daily_volatility, n_days))
    lows = prices * (1 - np.random.uniform(0, daily_volatility, n_days))
    opens = np.roll(prices, 1) * (1 + np.random.normal(0, volatility * 0.3, n_days))
    opens[0] = base_price

    # 出来高（ランダム + トレンド相関）
    base_volume = 1_000_000
    volume_factor = 1 + np.abs(returns) * 50  # 価格変動が大きい日は出来高増加
    volumes = (base_volume * volume_factor * np.random.uniform(0.5, 1.5, n_days)).astype(int)

    # DataFrameを作成
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    }, index=dates)

    # High/Low調整（Close, Openを含むように）
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)

    return df


def generate_multiple_stocks(
    symbols: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """複数銘柄のサンプルデータを生成"""
    stock_data = {}

    # 各銘柄に異なる特性を付与
    stock_params = {
        # 成長株（上昇トレンド、高ボラ）
        "growth": {"trend": 0.0003, "volatility": 0.025, "base_price": 3000},
        # バリュー株（安定、低ボラ）
        "value": {"trend": 0.0001, "volatility": 0.015, "base_price": 1500},
        # 大型株（安定）
        "large_cap": {"trend": 0.00015, "volatility": 0.018, "base_price": 2000},
        # 高ボラ株
        "high_vol": {"trend": 0.0001, "volatility": 0.035, "base_price": 500},
    }

    for i, symbol in enumerate(symbols):
        # 銘柄タイプをローテーション
        types = list(stock_params.keys())
        stock_type = types[i % len(types)]
        params = stock_params[stock_type]

        df = generate_sample_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            base_price=params["base_price"],
            volatility=params["volatility"],
            trend=params["trend"],
            seed=hash(symbol) % (2**32)  # 再現性のためseedを固定
        )
        stock_data[symbol] = df
        logger.debug(f"Generated {symbol}: {len(df)} days, type={stock_type}")

    return stock_data


def get_sample_symbols() -> List[str]:
    """サンプル銘柄リスト"""
    return [
        "7203.JP", "6758.JP", "9984.JP", "6861.JP", "8306.JP",
        "9432.JP", "6501.JP", "7267.JP", "8035.JP", "6902.JP",
        "4063.JP", "6098.JP", "4519.JP", "6367.JP", "7741.JP",
        "4755.JP", "9433.JP", "9434.JP", "6594.JP", "6857.JP",
    ]


def get_sample_sector_map() -> Dict[str, str]:
    """サンプルセクターマップ"""
    return {
        "7203.JP": "auto", "7267.JP": "auto", "6902.JP": "auto",
        "6758.JP": "tech", "6501.JP": "tech", "8035.JP": "tech",
        "6857.JP": "tech", "6594.JP": "tech",
        "9984.JP": "telecom", "9432.JP": "telecom", "9433.JP": "telecom", "9434.JP": "telecom",
        "6861.JP": "precision", "6098.JP": "service", "4063.JP": "chemical",
        "4519.JP": "pharma",
        "8306.JP": "finance",
        "6367.JP": "machinery", "7741.JP": "precision",
        "4755.JP": "internet",
    }


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
            change = basic_val - enhanced_val
            indicator = "✓" if change > 0 else "✗"
        else:
            change = enhanced_val - basic_val
            indicator = "✓" if change > 0 else "✗"

        print(f"{name:<15} {basic_val:>10.2f}{unit:>2} {enhanced_val:>10.2f}{unit:>2} {change:>+10.2f}{unit:>2} {indicator}")


def analyze_trades(results, name: str):
    """取引の詳細分析"""
    print(f"\n{'='*60}")
    print(f"📋 {name} - 取引分析")
    print(f"{'='*60}")

    if not results.trades:
        print("  取引なし")
        return

    # Exit reason別の集計
    if hasattr(results.trades[0], 'exit_reason'):
        exit_reasons = {}
        for trade in results.trades:
            if trade.exit_reason:
                reason = trade.exit_reason
                if reason not in exit_reasons:
                    exit_reasons[reason] = {"count": 0, "total_return": 0}
                exit_reasons[reason]["count"] += 1
                exit_reasons[reason]["total_return"] += trade.return_pct

        print("\n  Exit Reason別:")
        for reason, stats in sorted(exit_reasons.items(), key=lambda x: x[1]["count"], reverse=True):
            avg_return = stats["total_return"] / stats["count"] if stats["count"] > 0 else 0
            print(f"    {reason:<20}: {stats['count']:>3}回, 平均{avg_return:+.2f}%")


def save_results(basic_results, enhanced_results, output_dir: Path):
    """結果をJSONで保存"""
    output_file = output_dir / f"backtest_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    def serialize_results(results, name):
        return {
            "name": name,
            "num_trades": results.num_trades,
            "win_rate": float(results.win_rate),
            "avg_win": float(results.avg_win),
            "avg_loss": float(results.avg_loss),
            "profit_factor": float(results.profit_factor),
            "sharpe_ratio": float(results.sharpe_ratio),
            "max_drawdown": float(results.max_drawdown),
            "total_return_pct": float(results.total_return_pct),
            "final_capital": float(results.final_capital),
            "avg_holding_days": float(results.avg_holding_days),
        }

    output_data = {
        "date": datetime.now().isoformat(),
        "note": "サンプルデータによるバックテスト結果",
        "basic": serialize_results(basic_results, "基本バックテスト"),
        "enhanced": serialize_results(enhanced_results, "改善版バックテスト"),
        "improvement": {
            "win_rate_change": float(enhanced_results.win_rate - basic_results.win_rate),
            "profit_factor_change": float(enhanced_results.profit_factor - basic_results.profit_factor),
            "return_change": float(enhanced_results.total_return_pct - basic_results.total_return_pct),
            "max_dd_reduction": float(basic_results.max_drawdown - enhanced_results.max_drawdown),
        }
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"結果を保存: {output_file}")
    return output_file


def main():
    """メイン処理"""
    print("\n" + "=" * 60)
    print("🔬 バックテスト比較分析（サンプルデータ版）")
    print("=" * 60)
    print("\n注意: 実際の株価データではなく、シミュレーションデータを使用")

    # バックテスト期間を設定
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=300)).strftime("%Y-%m-%d")

    # 1. サンプルデータ生成
    symbols = get_sample_symbols()
    sector_map = get_sample_sector_map()

    logger.info(f"対象銘柄数: {len(symbols)}")
    logger.info(f"バックテスト期間: {start_date} ~ {end_date}")

    logger.info("サンプルデータ生成中...")
    stock_data = generate_multiple_stocks(
        symbols,
        start_date,
        end_date
    )

    # 2. 基本バックテスト実行
    logger.info("\n基本バックテスト実行中...")
    basic_results = run_basic_backtest(stock_data, start_date, end_date)
    print_results("基本バックテスト", basic_results)
    analyze_trades(basic_results, "基本バックテスト")

    # 3. 改善版バックテスト実行
    logger.info("\n改善版バックテスト実行中...")
    enhanced_results = run_enhanced_backtest(
        stock_data, start_date, end_date, sector_map
    )
    print_results("改善版バックテスト（全機能）", enhanced_results)
    analyze_trades(enhanced_results, "改善版バックテスト")

    # 4. 比較分析
    print_comparison(basic_results, enhanced_results)

    # 5. 結果保存
    output_file = save_results(basic_results, enhanced_results, REPORTS_DIR)

    # サマリー
    print(f"\n{'='*60}")
    print("✅ バックテスト完了")
    print(f"{'='*60}")

    # 改善効果の判定
    win_rate_improved = enhanced_results.win_rate > basic_results.win_rate
    return_improved = enhanced_results.total_return_pct > basic_results.total_return_pct
    dd_reduced = enhanced_results.max_drawdown < basic_results.max_drawdown
    sharpe_improved = enhanced_results.sharpe_ratio > basic_results.sharpe_ratio

    print(f"\n改善効果サマリー:")
    print(f"  勝率向上:       {'✅' if win_rate_improved else '❌'} ({(enhanced_results.win_rate - basic_results.win_rate)*100:+.1f}%)")
    print(f"  リターン向上:   {'✅' if return_improved else '❌'} ({enhanced_results.total_return_pct - basic_results.total_return_pct:+.1f}%)")
    print(f"  リスク低減:     {'✅' if dd_reduced else '❌'} (DD {basic_results.max_drawdown - enhanced_results.max_drawdown:+.1f}%)")
    print(f"  シャープ向上:   {'✅' if sharpe_improved else '❌'} ({enhanced_results.sharpe_ratio - basic_results.sharpe_ratio:+.2f})")

    print(f"\n結果ファイル: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
