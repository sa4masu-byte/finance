"""
パフォーマンスページ: ポートフォリオパフォーマンス表示
"""
import streamlit as st
from datetime import datetime
import pandas as pd
from pathlib import Path
from PIL import Image
from web.data_manager import DataManager


def render():
    """パフォーマンスページをレンダリング"""

    st.markdown('<div class="main-header">📈 ポートフォリオパフォーマンス</div>', unsafe_allow_html=True)
    st.markdown("推奨銘柄を実際に購入していた場合の資金推移")
    st.markdown("---")

    # データマネージャー初期化
    dm = DataManager()

    # 統計情報取得
    stats = dm.get_portfolio_statistics()

    if not stats:
        st.warning("⚠️ パフォーマンスデータがありません。")
        st.info("""
        **パフォーマンス可視化の実行:**
        ```bash
        python scripts/visualize_portfolio_performance.py
        ```

        推奨履歴が蓄積されると、自動的にポートフォリオシミュレーションを実行できます。
        """)
        return

    # パフォーマンスグラフ表示
    chart_path = dm.get_performance_chart_path()

    if chart_path and Path(chart_path).exists():
        st.markdown("## 📊 資金推移グラフ")

        # 画像読み込み
        image = Image.open(chart_path)
        st.image(image, use_container_width=True)

        # グラフ更新日時
        chart_date = datetime.fromtimestamp(Path(chart_path).stat().st_mtime)
        st.caption(f"最終更新: {chart_date.strftime('%Y年%m月%d日 %H:%M:%S')}")

    else:
        st.warning("グラフファイルが見つかりません。")

    st.markdown("---")

    # 統計情報表示
    st.markdown("## 📈 パフォーマンス統計")

    # 基本統計が辞書にある場合
    if 'total_recommendations' in stats:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("総推奨回数", f"{stats['total_recommendations']}回")
        with col2:
            st.metric("平均スコア", f"{stats['avg_score']:.1f}")
        with col3:
            st.metric("推奨銘柄数", f"{stats['total_symbols']}銘柄")

    # 推奨履歴から詳細統計を計算
    st.markdown("### 📊 推奨履歴統計")

    history = dm.get_recommendation_history()

    if history:
        # 日付ごとの推奨数
        daily_counts = [len(h.get('recommendations', [])) for h in history]

        # 全推奨をフラット化
        all_recommendations = []
        for h in history:
            all_recommendations.extend(h.get('recommendations', []))

        if all_recommendations:
            # スコア分布
            scores = [r['total_score'] for r in all_recommendations]
            confidences = [r.get('confidence', 0) for r in all_recommendations]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### スコア分布")
                score_df = pd.DataFrame({'スコア': scores})
                st.bar_chart(score_df['スコア'].value_counts().sort_index())

                st.metric("平均スコア", f"{sum(scores) / len(scores):.1f}")
                st.metric("最高スコア", f"{max(scores):.1f}")
                st.metric("最低スコア", f"{min(scores):.1f}")

            with col2:
                st.markdown("#### 信頼度分布")
                conf_df = pd.DataFrame({'信頼度': confidences})
                st.bar_chart(conf_df['信頼度'].value_counts().sort_index())

                st.metric("平均信頼度", f"{sum(confidences) / len(confidences):.1%}")
                st.metric("最高信頼度", f"{max(confidences):.1%}")
                st.metric("最低信頼度", f"{min(confidences):.1%}")

            # 銘柄別推奨回数
            st.markdown("#### 銘柄別推奨回数（トップ10）")
            symbols = [r['symbol'] for r in all_recommendations]
            symbol_counts = pd.Series(symbols).value_counts().head(10)

            st.bar_chart(symbol_counts)

    st.markdown("---")

    # エクスポート機能
    st.markdown("## 💾 データエクスポート")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### CSV エクスポート")
        if st.button("📥 推奨履歴をCSVでダウンロード"):
            csv_path = dm.export_to_csv()
            st.success(f"✅ エクスポート完了: {csv_path}")

            # ダウンロードボタン
            with open(csv_path, 'rb') as f:
                st.download_button(
                    label="📥 ダウンロード",
                    data=f,
                    file_name=Path(csv_path).name,
                    mime='text/csv'
                )

    with col2:
        st.markdown("### 期間指定エクスポート")

        # 日付範囲選択
        date_range = st.date_input(
            "期間を選択",
            value=(datetime.now().date(), datetime.now().date()),
            key="export_date_range"
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
            if st.button("📥 期間指定でエクスポート"):
                csv_path = dm.export_to_csv(
                    pd.to_datetime(start_date),
                    pd.to_datetime(end_date)
                )
                st.success(f"✅ エクスポート完了: {csv_path}")

                with open(csv_path, 'rb') as f:
                    st.download_button(
                        label="📥 ダウンロード",
                        data=f,
                        file_name=Path(csv_path).name,
                        mime='text/csv'
                    )

    st.markdown("---")

    # 手動更新
    st.markdown("## 🔄 手動更新")
    st.info("""
    パフォーマンスグラフを手動で更新する場合:
    ```bash
    python scripts/visualize_portfolio_performance.py
    ```
    """)

    if st.button("🔄 ページを再読み込み"):
        st.rerun()
