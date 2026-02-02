"""
履歴ページ: 過去の推奨履歴表示
"""
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from web.data_manager import DataManager


def render():
    """履歴ページをレンダリング"""

    st.markdown('<div class="main-header">📜 推奨履歴</div>', unsafe_allow_html=True)
    st.markdown("過去の推奨銘柄とパフォーマンス")
    st.markdown("---")

    # データマネージャー初期化
    dm = DataManager()

    # フィルター設定
    st.markdown("## 🔍 フィルター")

    col1, col2, col3 = st.columns(3)

    with col1:
        # 期間選択
        period_option = st.selectbox(
            "表示期間",
            ["全期間", "過去7日間", "過去30日間", "過去90日間", "カスタム"]
        )

    with col2:
        # ソート順
        sort_option = st.selectbox(
            "並び順",
            ["日付（新しい順）", "日付（古い順）", "スコア（高い順）", "スコア（低い順）"]
        )

    with col3:
        # 表示件数
        limit = st.number_input(
            "表示件数",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )

    # 日付範囲設定
    start_date = None
    end_date = None

    if period_option == "過去7日間":
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
    elif period_option == "過去30日間":
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
    elif period_option == "過去90日間":
        start_date = datetime.now() - timedelta(days=90)
        end_date = datetime.now()
    elif period_option == "カスタム":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("開始日", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("終了日", datetime.now())

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

    st.markdown("---")

    # 履歴取得
    history = dm.get_recommendation_history(start_date, end_date)

    if not history:
        st.warning("⚠️ 履歴データがありません。")
        st.info("""
        **推奨銘柄の生成:**
        ```bash
        python scripts/run_daily_recommendation.py
        ```
        """)
        return

    # データを平坦化
    rows = []
    for rec in history:
        rec_date = pd.to_datetime(rec['date'])
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

    if df.empty:
        st.warning("該当する履歴がありません。")
        return

    # ソート
    if sort_option == "日付（新しい順）":
        df = df.sort_values('date', ascending=False)
    elif sort_option == "日付（古い順）":
        df = df.sort_values('date', ascending=True)
    elif sort_option == "スコア（高い順）":
        df = df.sort_values('total_score', ascending=False)
    elif sort_option == "スコア（低い順）":
        df = df.sort_values('total_score', ascending=True)

    # 表示件数制限
    df_display = df.head(limit)

    # サマリー統計
    st.markdown("## 📊 サマリー統計")

    summary_cols = st.columns(4)

    with summary_cols[0]:
        st.metric("総推奨回数", f"{len(df)}回")
    with summary_cols[1]:
        st.metric("推奨銘柄数", f"{df['symbol'].nunique()}銘柄")
    with summary_cols[2]:
        st.metric("平均スコア", f"{df['total_score'].mean():.1f}")
    with summary_cols[3]:
        st.metric("平均信頼度", f"{df['confidence'].mean():.1%}")

    st.markdown("---")

    # データテーブル表示
    st.markdown(f"## 📋 推奨履歴（{len(df_display)}件表示）")

    # 表示用にフォーマット
    display_df = df_display.copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df['price'] = display_df['price'].apply(lambda x: f"¥{x:,.0f}")
    display_df['total_score'] = display_df['total_score'].apply(lambda x: f"{x:.1f}")
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    display_df['rsi'] = display_df['rsi'].apply(lambda x: f"{x:.1f}")
    display_df['adx'] = display_df['adx'].apply(lambda x: f"{x:.1f}")
    display_df['volume_ratio'] = display_df['volume_ratio'].apply(lambda x: f"{x:.2f}x")

    # カラム名を日本語に
    display_df = display_df.rename(columns={
        'date': '日付',
        'symbol': '銘柄',
        'price': '価格',
        'total_score': 'スコア',
        'trend_score': 'トレンド',
        'momentum_score': 'モメンタム',
        'volume_score': '出来高',
        'volatility_score': 'ボラティリティ',
        'confidence': '信頼度',
        'rsi': 'RSI',
        'adx': 'ADX',
        'volume_ratio': '出来高比'
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # グラフ表示
    st.markdown("## 📈 推移グラフ")

    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["スコア推移", "銘柄別推奨回数", "期間別推奨数"])

    with chart_tab1:
        st.markdown("### スコア推移")
        chart_df = df.copy()
        chart_df = chart_df.set_index('date')
        st.line_chart(chart_df[['total_score']])

    with chart_tab2:
        st.markdown("### 銘柄別推奨回数（トップ15）")
        symbol_counts = df['symbol'].value_counts().head(15)
        st.bar_chart(symbol_counts)

    with chart_tab3:
        st.markdown("### 期間別推奨数")
        daily_counts = df.groupby(df['date'].dt.date).size()
        st.bar_chart(daily_counts)

    st.markdown("---")

    # 詳細分析
    st.markdown("## 🔬 詳細分析")

    analysis_tab1, analysis_tab2 = st.tabs(["スコア分布", "テクニカル指標"])

    with analysis_tab1:
        st.markdown("### スコア分布")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### トータルスコア")
            st.bar_chart(df['total_score'].value_counts().sort_index())

        with col2:
            st.markdown("#### 信頼度")
            st.bar_chart(df['confidence'].value_counts().sort_index())

    with analysis_tab2:
        st.markdown("### テクニカル指標分布")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### RSI分布")
            st.bar_chart(df['rsi'].value_counts().sort_index())

        with col2:
            st.markdown("#### ADX分布")
            st.bar_chart(df['adx'].value_counts().sort_index())
