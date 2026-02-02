"""
ホームページ: 今日の推奨銘柄表示
"""
import streamlit as st
from datetime import datetime
import pandas as pd
from web.data_manager import DataManager


def render():
    """ホームページをレンダリング"""

    st.markdown('<div class="main-header">📈 株式推奨システム</div>', unsafe_allow_html=True)

    # データマネージャー初期化
    dm = DataManager()

    # 現在日時
    now = datetime.now()
    st.markdown(f"### 📅 {now.strftime('%Y年%m月%d日 (%A)')}")
    st.markdown("---")

    # 最新の推奨を取得
    latest = dm.get_latest_recommendations()

    if not latest:
        st.warning("⚠️ 推奨データがありません。定期実行を有効にしてください。")
        st.info("""
        **初回セットアップ:**
        1. バックグラウンドでスケジューラーを起動
        2. 手動で推奨生成を実行:
           ```bash
           python scripts/run_daily_recommendation.py
           ```
        3. このページを更新
        """)
        return

    # 推奨日付
    rec_date = pd.to_datetime(latest['date'])
    st.success(f"✅ 最新推奨: {rec_date.strftime('%Y年%m月%d日 %H:%M')}")

    # 推奨基準
    criteria = latest.get('criteria', {})
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("最小スコア", f"{criteria.get('min_score', 0):.0f}")
    with col2:
        st.metric("最小信頼度", f"{criteria.get('min_confidence', 0):.1%}")
    with col3:
        st.metric("推奨銘柄数", len(latest.get('recommendations', [])))

    st.markdown("---")

    # 推奨銘柄リスト
    recommendations = latest.get('recommendations', [])

    if not recommendations:
        st.info("本日の推奨銘柄はありません。")
        return

    st.markdown("## 🎯 本日の推奨銘柄")

    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"**#{i} {rec['symbol']}** - スコア: {rec['total_score']:.1f}", expanded=(i <= 3)):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"### {rec['symbol']}")
                st.markdown(f"**現在価格:** ¥{rec['price']:,.0f}")

                # スコア詳細
                st.markdown("#### 📊 スコア詳細")

                score_data = {
                    'カテゴリ': ['トレンド', 'モメンタム', '出来高', 'ボラティリティ'],
                    'スコア': [
                        rec.get('trend_score', 0),
                        rec.get('momentum_score', 0),
                        rec.get('volume_score', 0),
                        rec.get('volatility_score', 0)
                    ]
                }
                score_df = pd.DataFrame(score_data)
                st.bar_chart(score_df.set_index('カテゴリ'))

            with col2:
                # メトリクス
                st.markdown("#### 📈 テクニカル指標")

                confidence = rec.get('confidence', 0)
                confidence_color = "normal"
                if confidence >= 0.8:
                    confidence_color = "off"  # Green
                elif confidence >= 0.7:
                    confidence_color = "normal"  # Yellow

                st.metric("信頼度", f"{confidence:.1%}")
                st.metric("RSI", f"{rec.get('rsi', 0):.1f}")
                st.metric("ADX", f"{rec.get('adx', 0):.1f}")
                st.metric("出来高比率", f"{rec.get('volume_ratio', 0):.2f}x")

            # 推奨アクション
            st.markdown("#### 💡 推奨アクション")

            total_score = rec['total_score']

            if total_score >= 75:
                st.success("🟢 **強い買い推奨**: 高スコア・高信頼度")
            elif total_score >= 70:
                st.info("🔵 **買い推奨**: 良好なスコア")
            else:
                st.warning("🟡 **慎重に検討**: スコアは基準値近辺")

            # リスク情報
            st.markdown("#### ⚠️ リスク管理")
            st.markdown(f"""
            - **推奨ポジションサイズ**: 資金の20%
            - **ストップロス**: 価格の約8-10%下
            - **利益目標**: +10-15%
            - **最大保有期間**: 15日間
            """)

    st.markdown("---")

    # サマリー統計
    st.markdown("## 📊 本日の推奨サマリー")

    summary_cols = st.columns(4)

    with summary_cols[0]:
        avg_score = sum(r['total_score'] for r in recommendations) / len(recommendations)
        st.metric("平均スコア", f"{avg_score:.1f}")

    with summary_cols[1]:
        avg_confidence = sum(r.get('confidence', 0) for r in recommendations) / len(recommendations)
        st.metric("平均信頼度", f"{avg_confidence:.1%}")

    with summary_cols[2]:
        avg_rsi = sum(r.get('rsi', 0) for r in recommendations) / len(recommendations)
        st.metric("平均RSI", f"{avg_rsi:.1f}")

    with summary_cols[3]:
        avg_volume_ratio = sum(r.get('volume_ratio', 0) for r in recommendations) / len(recommendations)
        st.metric("平均出来高比率", f"{avg_volume_ratio:.2f}x")

    # 注意事項
    st.markdown("---")
    st.info("""
    **📌 重要な注意事項:**
    - 本推奨は過去データに基づく統計的分析です
    - 必ず自身でファンダメンタルズ分析を実施してください
    - リスク管理を徹底し、損切りルールを守ってください
    - 投資判断は自己責任で行ってください
    """)
