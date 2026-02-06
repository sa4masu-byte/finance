"""
推奨システムの診断スクリプト
なぜ推奨銘柄が生成されないかを調査
"""
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.fetcher import StockDataFetcher
from src.analysis.indicators import TechnicalIndicators
from backtesting.scoring_engine import ScoringEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose():
    """推奨システムを診断"""

    logger.info("=" * 80)
    logger.info("推奨システム診断開始")
    logger.info("=" * 80)

    # 銘柄リスト読み込み
    universe_file = PROJECT_ROOT / "config" / "stock_universe.json"
    with open(universe_file, 'r') as f:
        universe = json.load(f)

    symbols = universe.get("symbols", [])[:5]  # 最初の5銘柄でテスト
    logger.info(f"テスト銘柄: {symbols}")

    # データ取得
    fetcher = StockDataFetcher()
    indicator_engine = TechnicalIndicators()
    scoring_engine = ScoringEngine()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    logger.info("\n1. データ取得テスト")
    logger.info("-" * 80)

    stock_data = {}
    for symbol in symbols:
        try:
            df = fetcher.fetch_stock_data(
                symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            if df is not None and len(df) > 0:
                logger.info(f"✅ {symbol}: {len(df)}行のデータ取得成功")
                stock_data[symbol] = df
            else:
                logger.warning(f"⚠️  {symbol}: データなし")
        except Exception as e:
            logger.error(f"❌ {symbol}: {e}")

    if not stock_data:
        logger.error("データが取得できませんでした")
        return

    logger.info(f"\n取得成功: {len(stock_data)}/{len(symbols)} 銘柄")

    # テクニカル指標計算
    logger.info("\n2. テクニカル指標計算テスト")
    logger.info("-" * 80)

    for symbol, df in stock_data.items():
        try:
            df = indicator_engine.calculate_all(df)
            stock_data[symbol] = df

            # 最新の指標を表示
            latest = df.iloc[-1]
            logger.info(f"\n{symbol} の最新指標:")
            logger.info(f"  Close: ¥{latest['Close']:.2f}")
            logger.info(f"  RSI: {latest.get('RSI', 0):.2f}")
            logger.info(f"  MACD: {latest.get('MACD', 0):.2f}")
            logger.info(f"  Volume: {latest.get('Volume', 0):.0f}")

        except Exception as e:
            logger.error(f"❌ {symbol} 指標計算エラー: {e}")

    # スコア計算
    logger.info("\n3. スコア計算テスト")
    logger.info("-" * 80)

    all_scores = []

    for symbol, df in stock_data.items():
        try:
            latest = df.iloc[-1]
            indicators = latest.to_dict()

            # スコア計算
            score_result = scoring_engine.calculate_score(indicators)

            logger.info(f"\n{symbol} のスコア:")
            logger.info(f"  総合スコア: {score_result['total_score']:.1f}")
            logger.info(f"  トレンド: {score_result['trend_score']:.1f}")
            logger.info(f"  モメンタム: {score_result['momentum_score']:.1f}")
            logger.info(f"  出来高: {score_result['volume_score']:.1f}")
            logger.info(f"  ボラティリティ: {score_result['volatility_score']:.1f}")
            logger.info(f"  信頼度: {score_result['confidence']:.2%}")

            should_recommend = scoring_engine.should_recommend(score_result)
            logger.info(f"  推奨判定: {'✅ 推奨' if should_recommend else '❌ 非推奨'}")

            all_scores.append({
                'symbol': symbol,
                'score': score_result['total_score'],
                'confidence': score_result['confidence'],
                'recommend': should_recommend
            })

        except Exception as e:
            logger.error(f"❌ {symbol} スコア計算エラー: {e}")

    # サマリー
    logger.info("\n" + "=" * 80)
    logger.info("診断結果サマリー")
    logger.info("=" * 80)

    df_scores = pd.DataFrame(all_scores)

    logger.info(f"\n銘柄数: {len(df_scores)}")
    logger.info(f"平均スコア: {df_scores['score'].mean():.1f}")
    logger.info(f"最高スコア: {df_scores['score'].max():.1f}")
    logger.info(f"最低スコア: {df_scores['score'].min():.1f}")
    logger.info(f"推奨銘柄数: {df_scores['recommend'].sum()}")

    logger.info("\n現在の推奨基準:")
    logger.info(f"  最小スコア: 65")
    logger.info(f"  最小信頼度: 0.70")

    # 推奨
    logger.info("\n" + "=" * 80)
    logger.info("推奨事項")
    logger.info("=" * 80)

    if df_scores['recommend'].sum() == 0:
        max_score = df_scores['score'].max()
        max_confidence = df_scores['confidence'].max()

        logger.warning("\n⚠️  推奨銘柄が0件です。基準を緩和することをお勧めします:")
        logger.info(f"\n提案1: スコア閾値を下げる")
        logger.info(f"  現在の最高スコア: {max_score:.1f}")
        logger.info(f"  推奨: 閾値を 50-60 に設定")

        logger.info(f"\n提案2: 信頼度閾値を下げる")
        logger.info(f"  現在の最高信頼度: {max_confidence:.2%}")
        logger.info(f"  推奨: 閾値を 0.60 に設定")

        logger.info(f"\n提案3: 両方を緩和")
        logger.info(f"  スコア閾値: 55")
        logger.info(f"  信頼度閾値: 0.65")
    else:
        logger.info(f"✅ {df_scores['recommend'].sum()}件の推奨銘柄が見つかりました")

    logger.info("\n" + "=" * 80)

if __name__ == "__main__":
    diagnose()
