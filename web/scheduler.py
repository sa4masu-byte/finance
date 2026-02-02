"""
定期実行スケジューラー
APSchedulerを使用してバックグラウンドタスクを管理
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import subprocess

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'scheduler.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class TaskScheduler:
    """タスクスケジューラー"""

    def __init__(self):
        self.scheduler = BlockingScheduler(timezone='Asia/Tokyo')
        self.project_root = PROJECT_ROOT

    def run_daily_recommendation(self):
        """日次推奨銘柄生成タスク"""
        try:
            logger.info("=" * 80)
            logger.info("日次推奨銘柄生成タスク開始")
            logger.info("=" * 80)

            script_path = self.project_root / "scripts" / "run_daily_recommendation.py"

            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=600  # 10分タイムアウト
            )

            if result.returncode == 0:
                logger.info("✅ 日次推奨銘柄生成完了")
                logger.info(result.stdout)
            else:
                logger.error("❌ 日次推奨銘柄生成失敗")
                logger.error(result.stderr)

        except subprocess.TimeoutExpired:
            logger.error("❌ タスクがタイムアウトしました")
        except Exception as e:
            logger.error(f"❌ タスク実行エラー: {e}", exc_info=True)

    def run_performance_visualization(self):
        """パフォーマンス可視化タスク"""
        try:
            logger.info("=" * 80)
            logger.info("パフォーマンス可視化タスク開始")
            logger.info("=" * 80)

            script_path = self.project_root / "scripts" / "visualize_portfolio_performance.py"

            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                logger.info("✅ パフォーマンス可視化完了")
                logger.info(result.stdout)
            else:
                logger.error("❌ パフォーマンス可視化失敗")
                logger.error(result.stderr)

        except subprocess.TimeoutExpired:
            logger.error("❌ タスクがタイムアウトしました")
        except Exception as e:
            logger.error(f"❌ タスク実行エラー: {e}", exc_info=True)

    def run_optimization(self):
        """週次最適化タスク（オプション）"""
        try:
            logger.info("=" * 80)
            logger.info("週次最適化タスク開始")
            logger.info("=" * 80)

            script_path = self.project_root / "scripts" / "run_optimization.py"

            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=3600  # 1時間タイムアウト
            )

            if result.returncode == 0:
                logger.info("✅ 最適化完了")
                logger.info(result.stdout)
            else:
                logger.error("❌ 最適化失敗")
                logger.error(result.stderr)

        except subprocess.TimeoutExpired:
            logger.error("❌ タスクがタイムアウトしました")
        except Exception as e:
            logger.error(f"❌ タスク実行エラー: {e}", exc_info=True)

    def setup_jobs(self):
        """ジョブをセットアップ"""

        # 毎日 09:00 JST - 推奨銘柄生成
        self.scheduler.add_job(
            self.run_daily_recommendation,
            trigger=CronTrigger(hour=9, minute=0, timezone='Asia/Tokyo'),
            id='daily_recommendation',
            name='日次推奨銘柄生成',
            replace_existing=True
        )
        logger.info("✅ ジョブ登録: 日次推奨銘柄生成 (毎日 09:00 JST)")

        # 毎日 18:00 JST - パフォーマンス可視化
        self.scheduler.add_job(
            self.run_performance_visualization,
            trigger=CronTrigger(hour=18, minute=0, timezone='Asia/Tokyo'),
            id='daily_performance',
            name='日次パフォーマンス可視化',
            replace_existing=True
        )
        logger.info("✅ ジョブ登録: 日次パフォーマンス可視化 (毎日 18:00 JST)")

        # 毎週月曜 10:00 JST - 最適化（オプション、コメントアウト）
        # 最適化は計算コストが高いため、必要に応じて有効化
        # self.scheduler.add_job(
        #     self.run_optimization,
        #     trigger=CronTrigger(day_of_week='mon', hour=10, minute=0, timezone='Asia/Tokyo'),
        #     id='weekly_optimization',
        #     name='週次最適化',
        #     replace_existing=True
        # )
        # logger.info("✅ ジョブ登録: 週次最適化 (毎週月曜 10:00 JST)")

    def start(self):
        """スケジューラー起動"""
        logger.info("=" * 80)
        logger.info("🚀 タスクスケジューラー起動")
        logger.info("=" * 80)

        self.setup_jobs()

        # 登録済みジョブを表示
        logger.info("\n📅 登録済みジョブ:")
        for job in self.scheduler.get_jobs():
            logger.info(f"  - {job.name} (ID: {job.id})")
            logger.info(f"    次回実行: {job.next_run_time}")

        logger.info("\n⏰ スケジューラー開始...")
        logger.info("停止するには Ctrl+C を押してください")
        logger.info("=" * 80)

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("\n🛑 スケジューラー停止")
            self.scheduler.shutdown()


def main():
    """メイン関数"""
    scheduler = TaskScheduler()
    scheduler.start()


if __name__ == "__main__":
    main()
