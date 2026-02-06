"""
過去の推奨銘柄をバックフィルするスクリプト
指定期間の各営業日に対して推奨銘柄を生成し、実際のパフォーマンスを追跡
"""
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.fetcher import StockDataFetcher
from src.analysis.indicators import TechnicalIndicators
from backtesting.scoring_engine import ScoringEngine
from config.settings import RISK_PARAMS

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 出力ディレクトリ
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class RecommendationBackfiller:
    """過去の推奨銘柄をバックフィルするクラス"""

    def __init__(self, days: int = 30):
        """
        Args:
            days: バックフィルする日数
        """
        self.days = days
        self.fetcher = StockDataFetcher()
        self.indicator_engine = TechnicalIndicators()
        self.scoring_engine = ScoringEngine()

        # 最適化された重みを読み込み
        weights_file = REPORTS_DIR / "best_weights.json"
        if weights_file.exists():
            with open(weights_file, 'r') as f:
                weights_data = json.load(f)
                self.scoring_engine = ScoringEngine(weights=weights_data.get('weights'))
                logger.info("最適化された重みを読み込みました")
        else:
            logger.warning("最適化された重みが見つかりません。デフォルト重みを使用します")

    def backfill(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        """
        指定期間の推奨銘柄をバックフィル

        Args:
            start_date: 開始日（Noneの場合は days 日前から）
            end_date: 終了日（Noneの場合は昨日まで）
        """
        # 日付範囲設定
        if end_date is None:
            end_date = datetime.now() - timedelta(days=1)  # 昨日まで

        if start_date is None:
            start_date = end_date - timedelta(days=self.days)

        logger.info("=" * 80)
        logger.info("推奨銘柄バックフィル開始")
        logger.info(f"期間: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
        logger.info("=" * 80)

        # 銘柄リスト読み込み
        universe_file = PROJECT_ROOT / "config" / "stock_universe.json"
        with open(universe_file, 'r') as f:
            universe = json.load(f)

        symbols = universe.get("symbols", [])
        logger.info(f"対象銘柄数: {len(symbols)}")

        # データ取得期間（指標計算のため、開始日の90日前から）
        data_start = start_date - timedelta(days=90)

        # 全銘柄のデータを取得
        logger.info("株価データ取得中...")
        stock_data = {}
        for symbol in symbols:
            try:
                df = self.fetcher.fetch_stock_data(
                    symbol,
                    start_date=data_start.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                if df is not None and len(df) > 0:
                    # テクニカル指標計算
                    df = self.indicator_engine.calculate_all(df)
                    stock_data[symbol] = df
            except Exception as e:
                logger.warning(f"{symbol} のデータ取得失敗: {e}")

        logger.info(f"データ取得完了: {len(stock_data)}/{len(symbols)} 銘柄")

        # 各営業日ごとに推奨銘柄を生成
        current_date = start_date
        generated_count = 0
        skipped_count = 0

        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            output_file = REPORTS_DIR / f"recommendation_{current_date.strftime('%Y%m%d')}.json"

            # 既に存在する場合はスキップ
            if output_file.exists():
                logger.info(f"⏭️  {date_str}: 既に存在（スキップ）")
                skipped_count += 1
                current_date += timedelta(days=1)
                continue

            # その日の推奨銘柄を生成
            recommendations = self._generate_recommendations_for_date(
                current_date,
                stock_data
            )

            if recommendations:
                # パフォーマンス追跡（その後のN日間の実績）
                recommendations_with_performance = self._add_performance_tracking(
                    recommendations,
                    current_date,
                    stock_data,
                    tracking_days=15
                )

                # JSON保存
                output = {
                    "date": current_date.isoformat(),
                    "recommendations": recommendations_with_performance,
                    "criteria": {
                        "min_score": 55,
                        "min_confidence": 0.65
                    },
                    "backfilled": True
                }

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)

                logger.info(f"✅ {date_str}: {len(recommendations)} 銘柄を生成")
                generated_count += 1
            else:
                logger.info(f"⚠️  {date_str}: 推奨銘柄なし")

            current_date += timedelta(days=1)

        logger.info("=" * 80)
        logger.info("バックフィル完了")
        logger.info(f"生成: {generated_count}日分")
        logger.info(f"スキップ: {skipped_count}日分")
        logger.info("=" * 80)

    def _generate_recommendations_for_date(
        self,
        date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> List[Dict]:
        """
        特定日付の推奨銘柄を生成

        Args:
            date: 対象日付
            stock_data: 株価データ辞書

        Returns:
            推奨銘柄リスト
        """
        candidates = []

        for symbol, df in stock_data.items():
            # その日のデータが存在するか確認
            if date not in df.index:
                continue

            # その日の指標を取得
            try:
                row = df.loc[date]
                indicators = row.to_dict()

                # スコア計算
                score_result = self.scoring_engine.calculate_score(indicators)

                # 推奨基準チェック（基準を緩和）
                # 元の基準: min_score=65, min_confidence=0.70
                # 緩和版: min_score=55, min_confidence=0.65
                total_score = score_result["total_score"]
                confidence = score_result["confidence"]

                if total_score >= 55 and confidence >= 0.65:
                    candidates.append({
                        "symbol": symbol,
                        "price": float(indicators.get("Close", 0)),
                        "total_score": score_result["total_score"],
                        "trend_score": score_result["trend_score"],
                        "momentum_score": score_result["momentum_score"],
                        "volume_score": score_result["volume_score"],
                        "volatility_score": score_result["volatility_score"],
                        "confidence": score_result["confidence"],
                        "rsi": float(indicators.get("RSI", 0)),
                        "adx": float(indicators.get("ADX", 0)),
                        "volume_ratio": float(indicators.get("Volume", 0) / indicators.get("Volume_MA20", 1))
                    })
            except Exception as e:
                logger.debug(f"{symbol} のスコア計算エラー ({date}): {e}")
                continue

        # スコア順にソート
        candidates.sort(key=lambda x: x["total_score"], reverse=True)

        # 上位5銘柄を返す
        return candidates[:5]

    def _add_performance_tracking(
        self,
        recommendations: List[Dict],
        rec_date: datetime,
        stock_data: Dict[str, pd.DataFrame],
        tracking_days: int = 15
    ) -> List[Dict]:
        """
        推奨銘柄の実際のパフォーマンスを追跡

        Args:
            recommendations: 推奨銘柄リスト
            rec_date: 推奨日
            stock_data: 株価データ
            tracking_days: 追跡日数

        Returns:
            パフォーマンス情報を追加した推奨銘柄リスト
        """
        for rec in recommendations:
            symbol = rec["symbol"]
            entry_price = rec["price"]

            if symbol not in stock_data:
                continue

            df = stock_data[symbol]

            # 推奨日以降のデータを取得
            future_data = df[df.index > rec_date].head(tracking_days)

            if len(future_data) == 0:
                continue

            # パフォーマンス計算
            max_price = future_data["High"].max()
            min_price = future_data["Low"].min()
            final_price = future_data["Close"].iloc[-1] if len(future_data) > 0 else entry_price

            max_gain = ((max_price - entry_price) / entry_price) * 100
            max_loss = ((min_price - entry_price) / entry_price) * 100
            final_return = ((final_price - entry_price) / entry_price) * 100

            # パフォーマンス情報を追加
            rec["performance"] = {
                "max_gain_pct": round(max_gain, 2),
                "max_loss_pct": round(max_loss, 2),
                "final_return_pct": round(final_return, 2),
                "max_price": round(max_price, 2),
                "min_price": round(min_price, 2),
                "final_price": round(final_price, 2),
                "tracking_days": len(future_data)
            }

        return recommendations


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="推奨銘柄バックフィルスクリプト")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="バックフィルする日数（デフォルト: 30日）"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="開始日 (YYYY-MM-DD形式)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="終了日 (YYYY-MM-DD形式)"
    )

    args = parser.parse_args()

    # 日付パース
    start_date = None
    end_date = None

    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')

    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    # バックフィル実行
    backfiller = RecommendationBackfiller(days=args.days)
    backfiller.backfill(start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    main()
