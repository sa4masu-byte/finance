#!/usr/bin/env python3
"""
日次推奨スクリプト（ローカルデータ最小化版）

機能:
- 監視銘柄の最新データを取得（直近60日分のみ）
- スコアリングを実行
- 推奨銘柄をJSON/CLIで出力
- 通知送信（設定時）

データ最小化:
- 必要最小限の日数のみ取得
- メモリ上で処理、不要なキャッシュは作成しない
"""
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    SCREENING_CRITERIA, SCORING_WEIGHTS, DATA_DIR, REPORTS_DIR,
)
from src.analysis.indicators import TechnicalIndicators
from src.data.fetcher import StockDataFetcher
from backtesting.scoring_engine import ScoringEngine

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# 監視銘柄リスト（最小限）
# =============================================================================
# 流動性の高い主要銘柄のみに絞る
WATCHLIST_MINIMAL = [
    # 大型株（流動性高）
    "7203.JP",  # トヨタ自動車
    "6758.JP",  # ソニーグループ
    "9984.JP",  # ソフトバンクグループ
    "6861.JP",  # キーエンス
    "8306.JP",  # 三菱UFJ
    "9432.JP",  # NTT
    "6501.JP",  # 日立製作所
    "7267.JP",  # ホンダ
    "8035.JP",  # 東京エレクトロン
    "6902.JP",  # デンソー
    # 成長株
    "4063.JP",  # 信越化学
    "6098.JP",  # リクルート
    "4519.JP",  # 中外製薬
    "6367.JP",  # ダイキン
    "7741.JP",  # HOYA
    # IT/ハイテク
    "4755.JP",  # 楽天グループ
    "9433.JP",  # KDDI
    "9434.JP",  # ソフトバンク
    "6594.JP",  # 日本電産
    "6857.JP",  # アドバンテスト
]


def load_watchlist() -> List[str]:
    """監視銘柄リストを読み込む"""
    watchlist_file = DATA_DIR / "watchlist.json"

    if watchlist_file.exists():
        with open(watchlist_file, 'r') as f:
            data = json.load(f)
            return data.get("symbols", WATCHLIST_MINIMAL)

    return WATCHLIST_MINIMAL


def fetch_minimal_data(symbols: List[str], days: int = 60) -> Dict:
    """
    必要最小限のデータを取得

    Args:
        symbols: 銘柄リスト
        days: 取得日数（デフォルト60日、テクニカル指標計算に必要な最小限）

    Returns:
        Dict[symbol, DataFrame]
    """
    fetcher = StockDataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    stock_data = {}
    failed = []

    for symbol in symbols:
        try:
            df = fetcher.fetch_stock_data(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            if df is not None and len(df) > 20:
                stock_data[symbol] = df
                logger.debug(f"  ✓ {symbol}: {len(df)} days")
            else:
                failed.append(symbol)
        except Exception as e:
            logger.warning(f"  ✗ {symbol}: {e}")
            failed.append(symbol)

    logger.info(f"データ取得完了: {len(stock_data)}/{len(symbols)} 銘柄")
    return stock_data


def calculate_recommendations(stock_data: Dict) -> List[Dict]:
    """
    各銘柄のスコアを計算し、推奨リストを生成

    Args:
        stock_data: Dict[symbol, DataFrame]

    Returns:
        推奨銘柄リスト（スコア降順）
    """
    indicator_engine = TechnicalIndicators()
    scoring_engine = ScoringEngine()

    recommendations = []

    for symbol, df in stock_data.items():
        try:
            # テクニカル指標を計算
            df_with_ind = indicator_engine.calculate_all(df)

            # 最新の指標値を取得
            latest = indicator_engine.get_latest_indicators(df_with_ind)
            if not latest:
                continue

            # スコアを計算
            result = scoring_engine.calculate_score(latest)

            # 推奨基準をチェック（基準を緩和）
            # 元の基準: min_score=65, min_confidence=0.70
            # 緩和版: min_score=55, min_confidence=0.65
            if result["total_score"] >= 55 and result["confidence"] >= 0.65:
                recommendations.append({
                    "symbol": symbol,
                    "price": latest["Close"],
                    "total_score": result["total_score"],
                    "trend_score": result["trend_score"],
                    "momentum_score": result["momentum_score"],
                    "volume_score": result["volume_score"],
                    "volatility_score": result["volatility_score"],
                    "confidence": result["confidence"],
                    "rsi": latest.get("RSI_14"),
                    "adx": latest.get("ADX"),
                    "volume_ratio": latest.get("Volume_Ratio"),
                })
        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")

    # スコア順にソート
    recommendations.sort(key=lambda x: x["total_score"], reverse=True)

    return recommendations[:5]  # 上位5銘柄


def output_recommendations(recommendations: List[Dict]):
    """推奨銘柄を出力"""
    print("\n" + "=" * 60)
    print("📊 本日の推奨銘柄")
    print("=" * 60)

    if not recommendations:
        print("\n⚠️  本日の推奨銘柄はありません")
        print("   市場状況を確認してください\n")
        return

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['symbol']}")
        print(f"   価格: ¥{rec['price']:,.0f}")
        print(f"   総合スコア: {rec['total_score']:.1f}/100")
        print(f"   信頼度: {rec['confidence']*100:.0f}%")
        print(f"   ├─ トレンド: {rec['trend_score']:.1f}")
        print(f"   ├─ モメンタム: {rec['momentum_score']:.1f}")
        print(f"   ├─ 出来高: {rec['volume_score']:.1f}")
        print(f"   └─ ボラティリティ: {rec['volatility_score']:.1f}")
        if rec.get('adx'):
            print(f"   ADX: {rec['adx']:.1f} {'(強いトレンド)' if rec['adx'] > 25 else ''}")
        if rec.get('rsi'):
            print(f"   RSI: {rec['rsi']:.1f}")

    print("\n" + "=" * 60)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60 + "\n")


def save_recommendations(recommendations: List[Dict]):
    """推奨結果をJSONで保存（最小限）"""
    output_file = REPORTS_DIR / f"recommendation_{datetime.now().strftime('%Y%m%d')}.json"

    # 古いレポートを削除（3日以上前）
    for old_file in REPORTS_DIR.glob("recommendation_*.json"):
        try:
            file_date = datetime.strptime(old_file.stem.split("_")[1], "%Y%m%d")
            if (datetime.now() - file_date).days > 3:
                old_file.unlink()
        except:
            pass

    output_data = {
        "date": datetime.now().isoformat(),
        "recommendations": recommendations,
        "criteria": {
            "min_score": SCREENING_CRITERIA["min_score"],
            "min_confidence": SCREENING_CRITERIA["min_confidence"],
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"結果を保存: {output_file}")


def send_notification(recommendations: List[Dict]):
    """通知を送信（設定されている場合）"""
    # Slack通知
    slack_webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if slack_webhook and recommendations:
        try:
            import urllib.request

            message = "📊 *本日の推奨銘柄*\n\n"
            for rec in recommendations:
                message += f"• {rec['symbol']}: スコア {rec['total_score']:.1f}\n"

            payload = json.dumps({"text": message}).encode('utf-8')
            req = urllib.request.Request(
                slack_webhook,
                data=payload,
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(req)
            logger.info("Slack通知を送信しました")
        except Exception as e:
            logger.warning(f"Slack通知エラー: {e}")


def main():
    """メイン処理"""
    logger.info("=" * 50)
    logger.info("日次推奨システム開始")
    logger.info("=" * 50)

    # 1. 監視銘柄を読み込む
    watchlist = load_watchlist()
    logger.info(f"監視銘柄数: {len(watchlist)}")

    # 2. 最小限のデータを取得
    logger.info("データ取得中...")
    stock_data = fetch_minimal_data(watchlist, days=60)

    if not stock_data:
        logger.error("データを取得できませんでした")
        return 1

    # 3. 推奨銘柄を計算
    logger.info("スコアリング実行中...")
    recommendations = calculate_recommendations(stock_data)

    # 4. 結果を出力
    output_recommendations(recommendations)

    # 5. 結果を保存
    save_recommendations(recommendations)

    # 6. 通知送信
    send_notification(recommendations)

    logger.info("日次推奨システム完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
