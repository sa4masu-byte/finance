"""
設定ページ: システム設定と管理
"""
import streamlit as st
import json
from pathlib import Path
from web.data_manager import DataManager


def render():
    """設定ページをレンダリング"""

    st.markdown('<div class="main-header">⚙️ 設定</div>', unsafe_allow_html=True)
    st.markdown("システム設定と管理")
    st.markdown("---")

    # データマネージャー初期化
    dm = DataManager()

    # タブ分け
    tab1, tab2, tab3 = st.tabs(["📊 最適化設定", "⏰ スケジュール設定", "ℹ️ システム情報"])

    with tab1:
        render_optimization_settings(dm)

    with tab2:
        render_schedule_settings()

    with tab3:
        render_system_info(dm)


def render_optimization_settings(dm: DataManager):
    """最適化設定"""

    st.markdown("## 📊 スコアリング重み設定")

    # 現在の重みを取得
    weights = dm.get_best_weights()

    if weights:
        st.success("✅ 最適化済みの重みが設定されています")

        current_weights = weights.get('weights', {})

        st.markdown("### 現在の重み")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("トレンド", f"{current_weights.get('trend', 0):.1%}")
            st.metric("モメンタム", f"{current_weights.get('momentum', 0):.1%}")
            st.metric("出来高", f"{current_weights.get('volume', 0):.1%}")

        with col2:
            st.metric("ボラティリティ", f"{current_weights.get('volatility', 0):.1%}")
            st.metric("パターン", f"{current_weights.get('pattern', 0):.1%}")

        # パフォーマンス情報
        if 'validation_performance' in weights:
            st.markdown("### 最適化パフォーマンス")

            perf = weights['validation_performance']

            perf_cols = st.columns(4)

            with perf_cols[0]:
                st.metric("勝率", f"{perf.get('avg_win_rate', 0):.1%}")
            with perf_cols[1]:
                st.metric("平均リターン", f"{perf.get('avg_return', 0):.1f}%")
            with perf_cols[2]:
                st.metric("シャープレシオ", f"{perf.get('avg_sharpe', 0):.2f}")
            with perf_cols[3]:
                st.metric("プロフィットファクター", f"{perf.get('avg_profit_factor', 0):.2f}")

    else:
        st.warning("⚠️ 最適化された重みがありません")

        st.info("""
        **重みの最適化を実行:**
        ```bash
        python scripts/run_optimization.py
        ```

        最適化には30分〜1時間かかります。
        """)

    st.markdown("---")

    # 手動実行ボタン
    st.markdown("## 🔄 手動実行")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎯 推奨銘柄生成を実行"):
            st.info("推奨銘柄生成を開始します...")
            st.code("python scripts/run_daily_recommendation.py")
            st.warning("⚠️ この機能は現在手動実行のみ対応しています。")

    with col2:
        if st.button("📊 パフォーマンス可視化を実行"):
            st.info("パフォーマンス可視化を開始します...")
            st.code("python scripts/visualize_portfolio_performance.py")
            st.warning("⚠️ この機能は現在手動実行のみ対応しています。")


def render_schedule_settings():
    """スケジュール設定"""

    st.markdown("## ⏰ 定期実行スケジュール")

    st.info("""
    定期実行はスケジューラーサービスで管理されます。
    Dockerコンテナで起動している場合、自動的に実行されます。
    """)

    # 現在のスケジュール表示
    st.markdown("### 📅 実行スケジュール")

    schedule_data = {
        "タスク": ["推奨銘柄生成", "パフォーマンス可視化", "重み最適化（オプション）"],
        "頻度": ["毎日", "毎日", "毎週月曜"],
        "実行時刻": ["09:00 JST", "18:00 JST", "10:00 JST"],
        "ステータス": ["⏸️ 停止中", "⏸️ 停止中", "⏸️ 停止中"]
    }

    st.table(schedule_data)

    st.markdown("---")

    # スケジューラー起動方法
    st.markdown("### 🚀 スケジューラー起動方法")

    st.markdown("#### Docker Composeを使用:")
    st.code("""
# スケジューラーサービスを起動
docker-compose up -d scheduler

# ログ確認
docker-compose logs -f scheduler
""", language="bash")

    st.markdown("#### 手動起動:")
    st.code("""
# バックグラウンドで実行
python web/scheduler.py &

# または nohup で実行
nohup python web/scheduler.py > scheduler.log 2>&1 &
""", language="bash")


def render_system_info(dm: DataManager):
    """システム情報"""

    st.markdown("## ℹ️ システム情報")

    # データ統計
    stats = dm.get_portfolio_statistics()

    if stats:
        st.markdown("### 📊 データ統計")

        info_cols = st.columns(3)

        with info_cols[0]:
            st.metric("総推奨回数", f"{stats.get('total_recommendations', 0)}回")
        with info_cols[1]:
            st.metric("推奨銘柄数", f"{stats.get('total_symbols', 0)}銘柄")
        with info_cols[2]:
            st.metric("データ保存日数", f"{stats.get('total_days', 0)}日")

    st.markdown("---")

    # ファイル情報
    st.markdown("### 📁 データファイル")

    reports_dir = Path(__file__).parent.parent.parent / "data" / "reports"

    if reports_dir.exists():
        # 推奨ファイル数
        rec_files = list(reports_dir.glob("recommendation_*.json"))
        perf_files = list(reports_dir.glob("portfolio_performance_*.png"))
        csv_files = list(reports_dir.glob("recommendations_export_*.csv"))

        file_cols = st.columns(3)

        with file_cols[0]:
            st.metric("推奨履歴ファイル", f"{len(rec_files)}件")
        with file_cols[1]:
            st.metric("パフォーマンスグラフ", f"{len(perf_files)}件")
        with file_cols[2]:
            st.metric("エクスポートCSV", f"{len(csv_files)}件")

    st.markdown("---")

    # システム要件
    st.markdown("### 💻 システム要件")

    st.info("""
    **必要なPythonパッケージ:**
    - streamlit
    - pandas
    - numpy
    - matplotlib
    - APScheduler
    - pandas-datareader

    **推奨環境:**
    - Python 3.9+
    - メモリ: 2GB以上
    - ディスク: 1GB以上（データ保存用）
    """)

    st.markdown("---")

    # バージョン情報
    st.markdown("### 📌 バージョン情報")

    st.code("""
株式推奨システム v1.0.0

- バックテストエンジン: v1.0
- スコアリングエンジン: v1.0
- Webダッシュボード: v1.0
- スケジューラー: v1.0
""")

    st.markdown("---")

    # ドキュメント
    st.markdown("### 📚 ドキュメント")

    st.markdown("""
    - [README](../README.md)
    - [アーキテクチャ設計書](../docs/web_service_architecture.md)
    - [指標選定根拠](../docs/indicator_selection_rationale.md)
    """)
