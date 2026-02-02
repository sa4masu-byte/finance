"""
データアクセス層
推奨履歴の読み込みとパフォーマンス計算を担当
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


class DataManager:
    """データアクセスマネージャー"""

    def __init__(self):
        self.reports_dir = REPORTS_DIR
        self.cache = {}

    def get_latest_recommendations(self) -> Optional[Dict]:
        """
        最新の推奨銘柄を取得

        Returns:
            最新の推奨データ、または None
        """
        try:
            # recommendation_*.json ファイルを検索
            recommendation_files = sorted(
                self.reports_dir.glob("recommendation_*.json"),
                reverse=True
            )

            if not recommendation_files:
                logger.warning("推奨履歴が見つかりません")
                return None

            # 最新ファイルを読み込み
            latest_file = recommendation_files[0]
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"最新推奨を読み込みました: {latest_file.name}")
            return data

        except Exception as e:
            logger.error(f"推奨データ読み込みエラー: {e}")
            return None

    def get_recommendation_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        期間内の推奨履歴を取得

        Args:
            start_date: 開始日（Noneの場合は全期間）
            end_date: 終了日（Noneの場合は現在まで）

        Returns:
            推奨データのリスト（日付降順）
        """
        try:
            recommendation_files = sorted(
                self.reports_dir.glob("recommendation_*.json"),
                reverse=True
            )

            if not recommendation_files:
                return []

            history = []

            for file_path in recommendation_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 日付フィルタリング
                rec_date = pd.to_datetime(data['date'])

                if start_date and rec_date < start_date:
                    continue
                if end_date and rec_date > end_date:
                    continue

                history.append(data)

            logger.info(f"推奨履歴を読み込みました: {len(history)}件")
            return history

        except Exception as e:
            logger.error(f"履歴読み込みエラー: {e}")
            return []

    def get_portfolio_statistics(self) -> Optional[Dict]:
        """
        ポートフォリオ統計情報を取得

        Returns:
            統計情報の辞書、またはNone
        """
        try:
            # ポートフォリオパフォーマンスファイルを検索
            perf_files = sorted(
                self.reports_dir.glob("portfolio_performance_*.json"),
                reverse=True
            )

            if perf_files:
                with open(perf_files[0], 'r', encoding='utf-8') as f:
                    return json.load(f)

            # JSONファイルがない場合、履歴から計算
            return self._calculate_statistics_from_history()

        except Exception as e:
            logger.error(f"統計情報取得エラー: {e}")
            return None

    def _calculate_statistics_from_history(self) -> Dict:
        """履歴から統計情報を計算"""
        history = self.get_recommendation_history()

        if not history:
            return {
                'total_recommendations': 0,
                'avg_score': 0,
                'total_symbols': 0
            }

        total_recs = 0
        total_score = 0
        symbols = set()

        for rec in history:
            recommendations = rec.get('recommendations', [])
            total_recs += len(recommendations)

            for r in recommendations:
                total_score += r.get('total_score', 0)
                symbols.add(r.get('symbol'))

        return {
            'total_recommendations': total_recs,
            'avg_score': total_score / total_recs if total_recs > 0 else 0,
            'total_symbols': len(symbols),
            'total_days': len(history)
        }

    def get_best_weights(self) -> Optional[Dict]:
        """
        最適化された重みを取得

        Returns:
            重み情報の辞書、またはNone
        """
        try:
            weights_file = self.reports_dir / "best_weights.json"

            if not weights_file.exists():
                logger.warning("重みファイルが見つかりません")
                return None

            with open(weights_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"重み読み込みエラー: {e}")
            return None

    def export_to_csv(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> str:
        """
        推奨履歴をCSVにエクスポート

        Args:
            start_date: 開始日
            end_date: 終了日

        Returns:
            CSVファイルのパス
        """
        history = self.get_recommendation_history(start_date, end_date)

        # データを平坦化
        rows = []
        for rec in history:
            rec_date = rec['date']
            for r in rec.get('recommendations', []):
                rows.append({
                    'date': rec_date,
                    'symbol': r.get('symbol'),
                    'price': r.get('price'),
                    'total_score': r.get('total_score'),
                    'trend_score': r.get('trend_score'),
                    'momentum_score': r.get('momentum_score'),
                    'volume_score': r.get('volume_score'),
                    'volatility_score': r.get('volatility_score'),
                    'confidence': r.get('confidence'),
                    'rsi': r.get('rsi'),
                    'adx': r.get('adx'),
                    'volume_ratio': r.get('volume_ratio')
                })

        df = pd.DataFrame(rows)

        # ファイル名生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.reports_dir / f"recommendations_export_{timestamp}.csv"

        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"CSVエクスポート完了: {csv_path}")

        return str(csv_path)

    def get_performance_chart_path(self) -> Optional[str]:
        """
        最新のパフォーマンスグラフのパスを取得

        Returns:
            グラフファイルのパス、またはNone
        """
        try:
            chart_files = sorted(
                self.reports_dir.glob("portfolio_performance_*.png"),
                reverse=True
            )

            if not chart_files:
                return None

            return str(chart_files[0])

        except Exception as e:
            logger.error(f"グラフパス取得エラー: {e}")
            return None
